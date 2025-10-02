import math
import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import trunc_normal_
from .seg_heads import CBAM_FUSE_Head
from .dfformer_layers import Block, TCBlock, OverlapPatchEmbed, CTM, BasicBlock, Aggregate
from .dfformer_utils import map2token,token2map,load_checkpoint
from . import dfformer_configs as configs
import sys
sys.path.append('./modules')

from dfformer.registry import MODELS

CONFIGS = {
    'dfformer-light': configs.get_dfformer_light_config(),
    'dfformer-small': configs.get_dfformer_small_config(),
    'dfformer-large': configs.get_dfformer_large_config()
}

class DiceLoss(nn.Module):
    def __init__(self, ):
        super(DiceLoss, self).__init__()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target):
        bs = target.shape[0]
        inputs = torch.sigmoid(inputs.view(bs, -1))
        target = target.view(bs, -1)
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        dice = self._dice_loss(inputs, target)
        return dice

@MODELS.register_module()
class DFFormer(nn.Module):
    def __init__(
            self, model_name, img_size=512, sample_ratios=[0.25, 0.25, 0.25],
            qkv_bias=True, qk_scale=None,drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=0.,norm_layer = partial(nn.LayerNorm, eps=1e-6),
            num_stages=4
    ):
        super().__init__()
        config = CONFIGS[model_name]
        cur = 0
        in_chans = config.in_chans[0]
        embed_dims = config.embed_dims[0]
        num_heads = config.num_heads[0]
        mlp_ratios =config.mlp_ratios[0]
        depths = config.depths[0]
        sr_ratios = config.sr_ratios[0]
        pretrained = config.pretrained[0]
        feature_strides = config.feature_strides[0]
        segment_dim = config.segment_dim[0]
        num_classes = config.num_classes
        self.num_stages = num_stages
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule


        # In stage 1, use the standard transformer blocks
        for i in range(1):
            patch_embed = OverlapPatchEmbed(img_size=img_size,
                                            patch_size=7,
                                            stride=4,
                                            in_chans=in_chans,
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i])
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            res_block = BasicBlock(inplanes=embed_dims[i], planes=embed_dims[i])

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)
            setattr(self, f"res_block{i + 1}", res_block)

        # In stage 2~4, use TCBlock for dynamic tokens
        for i in range(1, num_stages):
            ctm = CTM(sample_ratios[i-1], embed_dims[i-1], embed_dims[i])

            block = nn.ModuleList([TCBlock(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i])
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]
            aggregate = Aggregate(in_channels=embed_dims[i], out_channels=embed_dims[i])
            res_block = BasicBlock(inplanes=embed_dims[i], planes=embed_dims[i])

            setattr(self, f"ctm{i}", ctm)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"res_block{i + 1}", res_block)
            setattr(self, f"aggregate{i + 1}", aggregate)
            setattr(self, f"norm{i + 1}", norm)

        self.seg_head = CBAM_FUSE_Head(img_size=img_size,
                                feature_strides=feature_strides,
                                in_channels=embed_dims,
                                embedding_dim=segment_dim,
                                num_classes=num_classes)

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.apply(self._init_weights)
        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            load_checkpoint(self, pretrained)
            print('load pretrained weights successfully!')

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

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    def forward_feature(self, inputs):
        logits = []
        res_logits = []

        i = 0
        patch_embed = getattr(self, f"patch_embed{i + 1}")
        block = getattr(self, f"block{i + 1}")
        res_block = getattr(self, f"res_block{i + 1}")
        norm = getattr(self, f"norm{i + 1}")
        res_block = getattr(self, f"res_block{i + 1}", res_block)
        x, H, W = patch_embed(inputs)
        B, N, _ = x.shape
        res = res_block(x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous())
        for blk in block:
            x = blk(x, H, W)
        x = norm(x)

        # init token dict
        device = x.device
        idx_token = torch.arange(N)[None, :].repeat(B, 1).to(device)
        agg_weight = x.new_ones(B, N, 1)
        token_dict = {'x': x,
                      'token_num': N,
                      'map_size': [H, W],
                      'init_grid_size': [H, W],
                      'idx_token': idx_token,
                      'agg_weight': agg_weight,
                      }
        logits.append(token_dict)
        res_logits.append(res)

        # stage 2~4
        for i in range(1, self.num_stages):
            ctm = getattr(self, f"ctm{i}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            aggregate = getattr(self,f'aggregate{i+1}')
            res_block = getattr(self, f"res_block{i + 1}")


            token_dict = ctm(token_dict)  # down sample tuple q_dict[b,1024,128],kv_dict[b,4096,128],res[b,128,32,32]
            res = res_block(aggregate(token_dict))

            for j, blk in enumerate(block):
                token_dict = blk(token_dict)
            token_dict['x'] = norm(token_dict['x'])
            logits.append(token_dict)
            res_logits.append(res)

        logits = [token2map(token_dict) for token_dict in logits]
        logit = self.seg_head(logits, res_logits)

        return logit

    def forward(self, image, mask, edge_masks, if_predcit_label=None, *args, **kwargs):
        if type(image) == tuple:
            image = image[0]
        if type(mask) == tuple:
            mask = mask[0]
        logit = self.forward_feature(image)
        mask = torch.squeeze(mask)

        loss_bce = self.bce_loss(logit.squeeze(), mask)
        loss_dice = self.dice_loss(logit.squeeze(), mask)
        loss = 0.5 * loss_bce + 0.5 * loss_dice

        output_dict = {
            # loss for backward
            "backward_loss": loss,
            # predicted mask, will calculate for metrics automatically
            "pred_mask": logit,
            "pred_label": None,
            "visual_loss": {
                "predict_loss": loss,
                },
                "visual_image": {
                "pred_mask": logit,
                }

            }
        return output_dict


if __name__ == '__main__':
    img = torch.randn(2, 3, 256, 256).cuda()
    #img = torch.randn(2, 3, 512, 512).cuda()
    config = CONFIGS['tc-small']
    net = DFFormer(config, img_size=256).cuda()

    for name,param in net.named_parameters():   #查看可训练参数
        if param.requires_grad:
            #print( name)
            print('{}:{}:'.format(name,param.shape))
    n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(n_parameters)
    out = net(img)
    print(out.shape)
