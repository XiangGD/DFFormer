import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .dfformer_utils import merge_tokens, token2map,map2token, cluster_dpc_knn
#debug
# from tcformer_utils import merge_tokens,token2map, map2token, cluster_dpc_knn

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# Mlp for dynamic tokens
class TCMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = TokenConv(in_channels=hidden_features,
                                out_channels=hidden_features,
                                kernel_size=3, padding=1, stride=1,
                                bias=True,
                                groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, token_dict):
        token_dict['x'] = self.fc1(token_dict['x'])
        x = self.dwconv(token_dict)#[]
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# The first conv layer
class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm(x)

        return x, H, W

class LowPassModule(nn.Module):
    def __init__(self, in_channel, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(size) for size in sizes])
        self.relu = nn.ReLU()
        ch = in_channel // 4
        self.channel_splits = [ch, ch, ch, ch]

    def _make_stage(self, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        return nn.Sequential(prior)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        feats = torch.split(feats, self.channel_splits, dim=1)#feats:tuple type [b,128,32,32]->[b,32,32,32]x4
        priors = [F.interpolate(input=self.stages[i](feats[i]), size=(h, w), mode='bilinear',align_corners=True) for i in range(4)]
        bottle = torch.cat(priors, 1) #bottle->[b,128,32,32]
        return self.relu(bottle)

class Filters_Conv2d(nn.Module):
    def __init__(self, dim, kernel_size, stride=1):
        self.in_channels = dim
        self.out_channels = dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2
        #self.minus1 = (torch.ones(self.in_channels, self.out_channels, 1) * -1.000)
        super(Filters_Conv2d, self).__init__()
        
        self.register_buffer('minus1', torch.full((self.in_channels, self.out_channels, 1), -1.0))

        self.kernel = nn.Parameter(self.base_kernel(kernel_size=kernel_size), requires_grad=True)

    def base_kernel(self, kernel_size):
        kernel = torch.rand(self.in_channels, self.out_channels, kernel_size ** 2 - 1)\
                 .permute(2, 0, 1).contiguous()
        kernel = torch.div(kernel, kernel.sum(dim=0, keepdim=True))
        return kernel.permute(1, 2, 0).contiguous()

    def constrain_kernel(self, kernel, size):
        #kernel = kernel.permute(2, 0, 1)
        #kernel = torch.div(kernel, kernel.sum(dim=0, keepdim=True))
        #kernel = kernel.permute(1, 2, 0)

        ctr = size ** 2 // 2
        real_kernel = torch.cat([
            kernel[:, :, :ctr], 
            self.minus1.to(kernel.device),  #
            kernel[:, :, ctr:]
        ], dim=2)

        return real_kernel.reshape(self.out_channels, self.in_channels, size, size)

    def forward(self, x):
        real_kernel = self.constrain_kernel(self.kernel, self.kernel_size)
        return F.conv2d(
            x, real_kernel,
            stride=self.stride,
            padding=self.padding,
            groups=1
        )

class ConvRelPosEnc(nn.Module):
    """Convolutional relative position encoding."""
    def __init__(self, dim, window):
        """Initialization.

        Ch: Channels per head.
        h: Number of heads.
        window: Window size(s) in convolutional relative positional encoding.
                It can have two forms:
                1. An integer of window size, which assigns all attention heads
                   with the same window size in ConvRelPosEnc.
                2. A dict mapping window size to #attention head splits
                   (e.g. {window size 1: #attention head split 1, window size
                                      2: #attention head split 2})
                   It will apply different window size to
                   the attention head splits.
        """
        super().__init__()

        if isinstance(window, int):
            # Set the same window size for all attention heads.
            window = {window: dim}
            self.window = window
            ch = 1
        elif isinstance(window, dict):
            self.window = window
            ch = dim // (sum(window.values())+2)
        else:
            raise ValueError()


        self.conv_list = nn.ModuleList()
        self.conv_list.append(nn.Identity())
        self.head_splits = [2]
        for cur_window, cur_head_split in window.items():
            cur_conv = Filters_Conv2d(dim=cur_head_split*ch,
                                      kernel_size=cur_window)
            self.conv_list.append(cur_conv)
            self.head_splits.append(cur_head_split)
        self.channel_splits = [x * ch for x in self.head_splits]
        self.LP = LowPassModule(dim)

    def forward(self, q, v, size):
        #foward function
        B, h, N, Ch = q.shape
        H, W = size

        # We don't use CLS_TOKEN
        # Shape: [B, h, H*W, Ch] -> [B, h*Ch, H, W].
        v = rearrange(v, "B (H W) C -> B C H W", H=H, W=W).contiguous() #v:[b,1024,128]->[b,128,32,32]
        lp = self.LP(v) #[b,128,32,32]
        # Split according to channels.
        v_list = torch.split(v, self.channel_splits, dim=1) #v_list:[b,128,32,32]->([b,32,32,32],[b,48,32,32],[b,48,32,32])
        conv_v_list = [conv(x) for conv, x in zip(self.conv_list, v_list)] #conv_v_list: [[b,32,32,32],[b,48,32,32],[b,48,32,32]]
        conv_v = torch.cat(conv_v_list, dim=1) # conv_v:[[b,32,32,32],[b,32,32,32],[b,32,32,32],[b,32,32,32]]->[b,128,32,32]
        # Shape: [B, h*Ch, H, W] -> [B, h, H*W, Ch].
        conv_v = rearrange(conv_v, "B (h Ch) H W -> B h (H W) Ch", h=h,Ch=Ch).contiguous()#conv_v:[b,128,32,32]->[b,2,1024,64], num_heads=2
        lp = rearrange(lp, "B (h Ch) H W -> B h (H W) Ch", h=h,Ch=Ch).contiguous()#lp:[b,128,32,32]->[b,2,1024,64]  num_heads=2

        return q * conv_v + lp


# Attention module with spatial reduction layer
class Attention(nn.Module):
    def __init__(self, dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 sr_ratio=1,
                 crpe_windows={ 3: 2,
                                5: 2,
                                7: 2 },):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.res = nn.Linear(dim, dim, bias=qkv_bias)
        self.res_active = nn.ReLU()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.crpe = ConvRelPosEnc(dim, crpe_windows)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).contiguous().reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1).contiguous()
            x_= self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()

        k, v = kv[0], kv[1]

        # attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = (q * self.scale) @ k.transpose(-2, -1).contiguous()
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        res = self.res_active(self.res(x))
        crpe = self.crpe(q, res, (H,W))

        x = (attn @ v + crpe).transpose(1, 2).contiguous().reshape(B, N, C).contiguous()
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


# Transformer blocks
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

# Attention for dynamic tokens
class TCAttention(nn.Module):
    def __init__(self, dim, num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 sr_ratio=1,
                 crpe_windows={ 3: 2,
                                5: 2,
                                7: 2 },
                 ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.res = nn.Linear(dim, dim, bias=qkv_bias)
        self.res_active = nn.ReLU()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.crpe = ConvRelPosEnc(dim, crpe_windows)

        self.sr_ratio = sr_ratio

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, q_dict, kv_dict):
        q = q_dict['x']# q:[b,1024,128],kv:[b,4096,128]
        kv = kv_dict['x']
        B, Nq, C = q.shape
        Nkv = kv.shape[1]

        conf_kv = kv_dict['token_score'] if 'token_score' in kv_dict.keys() else kv.new_zeros(B, Nkv, 1) #conf_kv:[b,4096,1]
        q = self.q(q).reshape(B, Nq, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous() #[b,2,1024,64] num_heads=2

        if self.sr_ratio > 1:
            tmp = torch.cat([kv, conf_kv], dim=-1)#[b,4096,129]
            tmp_dict = kv_dict.copy()
            tmp_dict['x'] = tmp
            tmp_dict['map_size'] = q_dict['map_size']#[32,32]
            tmp = token2map(tmp_dict) # H*W = N/4 #[b,129,32,32]
            kv_ = tmp[:, :C] #[B,128,32,32]
            conf_kv = tmp[:, C:] #[2,1,32,32]

            kv = self.sr(kv_).reshape(B, C, -1).permute(0, 2, 1).contiguous()#kv:[b,64,128]
            kv = self.norm(kv)

            conf_kv = F.avg_pool2d(conf_kv, kernel_size=self.sr_ratio, stride=self.sr_ratio) #[b,1,8,8]
            conf_kv = conf_kv.reshape(B, 1, -1).permute(0, 2, 1).contiguous() #[b,64,1]
        else:
            kv_dict['map_size'] = q_dict['map_size']
            kv_ = token2map(kv_dict)  # [B,1024,512]

        res = self.res_active(self.res(kv_.reshape(B, C, -1).permute(0, 2, 1).contiguous()))#kv_:[b,128,32,32]->[b,128,1024]->[b,1024,128]->x_
        crpe = self.crpe(q, res, q_dict['map_size']) # q:[b,2,1024,64], x_:[b,1024,128] q_dict['map_size']=(32,32),crpe:[b,2,1024,64]
        kv = self.kv(kv).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous() #kv:[2,2,2,64,64]
        k, v = kv[0], kv[1]#k:[b,2,64,64],v:[b,2,64,64]

        attn = (q * self.scale) @ k.transpose(-2, -1).contiguous()#attn:[b,2,1024,64]@[b,2,64,64]->[b,2,1024,64]

        conf_kv = conf_kv.squeeze(-1)[:, None, None, :]#con_kv:[b,64,1]->[b,64]->[b,1,1,64]
        attn = attn + conf_kv #  attn:[b,2,1024,64]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        #x = (attn @ v).transpose(1, 2).reshape(B, Nq, C).contiguous()
        x = (attn @ v + crpe).transpose(1, 2).contiguous().reshape(B, Nq, C).contiguous() # x:[b,1024,128]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# Transformer block for dynamic tokens
class TCBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, use_sr_layer=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = TCAttention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = TCMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, inputs):
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            q_dict, kv_dict = inputs  # 2tcattention: q:[b,1024,128],#kv:[b,1024,128]
        else:
            q_dict, kv_dict = inputs, None

        x = q_dict['x'] #[b,1024,128]
        # norm1
        q_dict['x'] = self.norm1(q_dict['x'])
        if kv_dict is None:
            kv_dict = q_dict
        else:
            kv_dict['x'] = self.norm1(kv_dict['x'])
        # attn
        x = x + self.drop_path(self.attn(q_dict, kv_dict))
        # mlp
        q_dict['x'] = self.norm2(x)
        x = x + self.drop_path(self.mlp(q_dict))#[b,1024,128]
        q_dict['x'] = x

        return q_dict

# depth-wise conv
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x

class TokenConv(nn.Conv2d):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        groups = kwargs['groups'] if 'groups' in kwargs.keys() else 1
        self.skip = nn.Conv1d(in_channels=kwargs['in_channels'],
                              out_channels=kwargs['out_channels'],
                              kernel_size=1, bias=False,
                              groups=groups)

    def forward(self, token_dict):
        x = token_dict['x'] #[b,1024,1024]
        x = self.skip(x.permute(0, 2, 1)).contiguous().permute(0, 2, 1).contiguous() #[[b,1024,1024]
        x_map = token2map(token_dict) #[b,1024,1024]-->[b,1024,32,32]
        x_map = super().forward(x_map)
        x = x + map2token(x_map, token_dict) # [b,4096,128]
        return x

# conv layer for dynamic tokens
class Aggregate(nn.Module):
    def __init__(self, in_channels, out_channels,):
        super().__init__()
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv = nn.Conv2d(in_channels=in_channels*2,
                               out_channels=out_channels,
                               kernel_size=3, stride=1,
                               padding=1,bias=False,
                              )

    def forward(self,inputs):
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            q_dict, kv_dict = inputs  # 2tcattention: q:[b,1024,128],#kv:[b,1024,128]
        else:
            q_dict, kv_dict = inputs, inputs
        q_map = token2map(q_dict) #[b,4096,64]
        kv_dict['map_size'] = q_dict['map_size']
        kv_map = token2map(kv_dict) # [b,64,64,64]
        x_map = torch.cat((kv_map, q_map),dim=1)
        x_map = self.conv(x_map)

        return x_map

# CTM block
class CTM(nn.Module):
    def __init__(self, sample_ratio, embed_dim, dim_out, k=5):
        super().__init__()
        self.sample_ratio = sample_ratio
        self.dim_out = dim_out
        self.conv = TokenConv(in_channels=embed_dim, out_channels=dim_out, kernel_size=3, stride=2, padding=1)
        self.norm = nn.LayerNorm(self.dim_out)
        self.score = nn.Linear(self.dim_out, 1)
        self.k = k

    def forward(self, token_dict):
        token_dict = token_dict.copy()#token_dict['x']->[b,4096,64]
        x = self.conv(token_dict)# x->[b,4096,128], res->[b,128,32,32]
        x = self.norm(x)
        token_score = self.score(x) #token_score->[b,4096,1]
        token_weight = token_score.exp() #token_weight->[b,4096,1]

        token_dict['x'] = x #[b,4096,128]
        B, N, C = x.shape
        token_dict['token_score'] = token_score

        cluster_num = max(math.ceil(N * self.sample_ratio), 1)#cluster_num:1024
        idx_cluster, cluster_num = cluster_dpc_knn(
            token_dict, cluster_num, self.k)
        down_dict = merge_tokens(token_dict, idx_cluster, cluster_num, token_weight)

        H, W = token_dict['map_size']
        H = math.floor((H - 1) / 2 + 1)
        W = math.floor((W - 1) / 2 + 1)
        down_dict['map_size'] = [H, W]

        return down_dict, token_dict


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Conv2d_BN(nn.Module):
    """Convolution with BN module."""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, pad=1, act_layer=None, norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=pad,
                               stride=stride, bias=False,groups=1,dilation=1)
        self.bn1 = norm_layer(out_ch)
        self.act_layer = act_layer() if act_layer is not None else nn.Identity( )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        """foward function"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act_layer(x)

        return x

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
