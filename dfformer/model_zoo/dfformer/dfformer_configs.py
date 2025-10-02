import ml_collections

def get_dfformer_light_config():
    "Returns the tcformer_light configuration"
    config = ml_collections.ConfigDict()
    config.in_chans = 3,
    config.embed_dims = [64, 128, 320, 512],
    config.num_heads = [1, 2, 5, 8],
    config.mlp_ratios = [8, 8, 4, 4],
    config.depths = [2, 2, 2, 2],
    config.sr_ratios = [8, 4, 2, 1],
    config.num_stages = 4,
    config.pretrained = '/home/user/DFFormer/pretrained/DFFormer/dfformer_light.pth',
    config.sample_ratios = [0.25, 0.25, 0.25],
    config.feature_strides = [4, 8, 16, 32],
    config.segment_dim = 64,
    config.num_classes = 1
    
    return config

def get_dfformer_small_config():
    "Returns the tcformer_small configuration"
    config = ml_collections.ConfigDict()
    config.img_size = 256,
    config.in_chans = 3,
    config.embed_dims = [64, 128, 320, 512],
    config.num_heads = [1, 2, 5, 8],
    config.mlp_ratios = [8, 8, 4, 4],
    config.depths = [3, 4, 6, 3],
    config.sr_ratios = [8, 4, 2, 1],
    config.num_stages = 4,
    config.qkv_bias = True,
    config.pretrained = '/home/eva/DFFormer/pretrained/DFFormer/dfformer_small.pth',
    config.sample_ratios = [0.25, 0.25, 0.25],
    config.feature_strides = [4, 8, 16, 32],
    config.segment_dim = 64,
    config.num_classes = 1
    return config

def get_dfformer_large_config():
    "Returns the tcformer_large configuration"
    config = ml_collections.ConfigDict()
    config.img_size = 256,
    config.in_chans = 3,
    config.embed_dims = [64, 128, 320, 512],
    config.num_heads = [1, 2, 5, 8],
    config.mlp_ratios = [8, 8, 4, 4],
    config.depths = [3, 8, 27, 3],
    config.sr_ratios = [8, 4, 2, 1],
    config.num_stages = 4,
    config.pretrained = '/home/eva/DFFormer/pretrained/DFFormer/dfformer_large.pth',
    config.sample_ratios = [0.25, 0.25, 0.25],
    config.feature_strides = [4, 8, 16, 32],
    config.segment_dim = 64,
    config.num_classes = 1
    return config

