import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


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
        

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

class CBAM_FUSE_Head(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(self, img_size, feature_strides, in_channels, embedding_dim,
                     num_classes, dropout_ratio=0, **kwargs):
        super(CBAM_FUSE_Head, self).__init__()
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.dropout_ratio = dropout_ratio
        self.img_size = [img_size, img_size // 4, img_size // 8, img_size // 16, img_size // 32]
        self.proj4 = Conv2d_BN(in_ch=in_channels[3], out_ch=in_channels[2], stride=1, act_layer=nn.Hardswish)
        self.proj3 = Conv2d_BN(in_ch=in_channels[2] * 2, out_ch=in_channels[1], stride=1, act_layer=nn.Hardswish)
        self.proj2 = Conv2d_BN(in_ch=in_channels[1] * 2, out_ch=in_channels[0], stride=1, act_layer=nn.Hardswish)
        self.proj1 = Conv2d_BN(in_ch=in_channels[0] * 2, out_ch=embedding_dim, stride=1, act_layer=nn.Hardswish)
        self.cbma = CBAM(embedding_dim, 16)
        self.linear_pred = nn.Conv2d(in_channels=embedding_dim, out_channels=num_classes, kernel_size=1, padding=0)
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()

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

    def forward(self, x1, x2 ):
        c1, c2, c3, c4 = x1
        r1, r2, r3, r4 = x2

        ############## fpn decoder on C1-C4 ###########
        c4 = c4 + r4
        c4_up = F.interpolate(c4, size=self.img_size[3], mode='bilinear', align_corners=False)
        c4_up = self.proj4(c4_up)

        c3 = c3 + r3
        c3 = torch.cat((c3, c4_up), dim=1)
        c3_up = F.interpolate(c3, size=self.img_size[2], mode='bilinear', align_corners=False)
        c3_up = self.proj3(c3_up)


        c2 = c2 + r2
        c2 = torch.cat((c2, c3_up), dim=1)
        c2_up = F.interpolate(c2, size=self.img_size[1], mode='bilinear', align_corners=False)
        c2_up = self.proj2(c2_up)

        c1 = c1 + r1
        c1 = torch.cat((c1, c2_up), dim=1)
        c1_up = F.interpolate(c1, size=self.img_size[0]//2, mode='bilinear', align_corners=False)
        c1_up = self.proj1(c1_up)

        x = F.interpolate(c1_up, size=self.img_size[0], mode='bilinear', align_corners=False)
        x = self.cbma(x) 
        x = self.dropout(x)
        x = self.linear_pred(x)
        return x


