import math
import logging
import warnings
import re
import os
import torch
import torch.nn.functional as F

'''
Note:
    B: batch size
    N: token number
    C: channel number
    N_init: initial token number
    H_init: height of initial grid
    W_init: width of initilal grid
    H: height of feature map
    W: width of feature map

    We represent the dynamic tokens by a dict with the following keys:
        x (torch.Tensor[B, N, C]): token features.
        token_num(int): token number.
        map_size(list[int] or tuple[int]): feature map resolution in format
            [H, W].
        init_grid_size(list[int] or tuple[int]): initial grid resolution in 
            format [H_init, W_init].
        idx_token(torch.LongTensor[B, N_init]): indicates which token the initial
            grid belongs to. 
        agg_weight(torch.LongTensor[B, N_init] or None): weight for aggregation. 
            Indicates the weight of each token in its cluster. If set to None,
            uniform weight is used.
'''

def get_grid_index(init_size, map_size, device):
    """For each initial grid, get its index in the feature map.
    Returns:
        idx (LongTensor[B, N_init]): index in flattened feature map.

    Args:
        init_grid_size(list[int] or tuple[int]): initial grid resolution in
            format [H_init, W_init].
        map_size(list[int] or tuple[int]): feature map resolution in format
            [H, W].
        device: the device of output
    """
    H_init, W_init = init_size
    H, W = map_size
    idx = torch.arange(H * W, device=device).reshape(1, 1, H, W)  #[1,1,64,64]#[[[0,1,...63],\n[64,...,127],...[4032,...4095]]]
    idx = F.interpolate(idx.float(), [H_init, W_init], mode='nearest').long()  #[1,1,128,128]#[[[0,0,1,1,...,63,63],\n[64,...,127],...[4032,...4095]]]
    return idx.flatten()


def index_points(points, idx):

    """Sample features following the index.
      
   Returns:
        new_points:, indexed points data, [B, S, C]

    Args:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def token2map(token_dict):
    """Transform vision tokens to feature map. This function only
    works when the resolution of the feature map is not higher than
    the initial grid structure.
    Returns:
        x_out (Tensor[B, C, H, W]): feature map.

    Args:
        token_dict (dict): dict for token information.
    """
    x = token_dict['x']
    H, W = token_dict['map_size']
    H_init, W_init = token_dict['init_grid_size']#init_grid_size is fixed
    idx_token = token_dict['idx_token']
    B, N, C = x.shape
    N_init = H_init * W_init
    device = x.device

    if N_init == N and N == H * W:
        # for the initial tokens with grid structure, just reshape
        return x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

    # for each initial grid, get the corresponding index in
    # the flattened feature map.
    idx_hw = get_grid_index([H_init, W_init], [H, W], device=device)[None, :].expand(B, -1)
    idx_batch = torch.arange(B, device=device)[:, None].expand(B, N_init)


    # choose the way with fewer flops.
    if N_init < N * H * W:
        # use sparse matrix multiplication
        # Flops: B * N_init * (C+2)
        idx_hw = idx_hw + idx_batch * H * W
        idx_token = idx_token + idx_batch * N
        indices = torch.stack([idx_hw, idx_token], dim=0).reshape(2, B * N_init)

        with torch.cuda.amp.autocast(enabled=False):
            # torch.sparse do not support gradient for
            # sparse tensor, so we detach it
            value = x.new_ones(B * N_init).detach().float()#[B*16384,]#[1,1,...,1,1]

            # build a sparse matrix with the shape [B * H * W, B * N]
            A = torch.sparse.FloatTensor(indices, value, torch.Size([B * H * W, B * N]))

            # normalize the weight for each rowk[
            all_weight = torch.sparse.mm(A, x.new_ones(B * N, 1).type(torch.float32)) + 1e-6
            value = value / all_weight[idx_hw.reshape(-1), 0]

            # update the matrix with normalize weight
            A = torch.sparse.FloatTensor(indices, value, torch.Size([B * H * W, B * N]))

            # sparse matrix multiplication
            x_out = torch.sparse.mm(A, x.reshape(B * N, C).type(torch.float32))  # [B*H*W, C]

    else:
        # use dense matrix multiplication
        # Flops: B * N * H * W * (C+2)
        indices = torch.stack([idx_batch, idx_hw, idx_token], dim=0).reshape(3, B * N_init)

        # build a matrix with shape [B, H*W, N]
        A = torch.sparse.FloatTensor(indices, x.new_ones(B * N_init), torch.Size([B, H * W, N])).to_dense()
        # normalize the weight
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-6)

        x_out = A @ x  # [B, H*W, C]

    x_out = x_out.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
    
    return x_out

def map2token(feature_map, token_dict):
    """Transform feature map to vision tokens. This function only
    works when the resolution of the feature map is not higher than
    the initial grid structure.

    Returns:
        out (Tensor[B, N, C]): token features.

    Args:
        feature_map (Tensor[B, C, H, W]): feature map.
        token_dict (dict): dict for token information.
    """
    idx_token = token_dict['idx_token']
    N = token_dict['token_num']
    H_init, W_init = token_dict['init_grid_size']
    N_init = H_init * W_init

    # agg_weight = token_dict['agg_weight'] if 'agg_weight' in token_dict.keys() else None
    #agg_weight = None  # we do not use the weight value here

    B, C, H, W = feature_map.shape
    device = feature_map.device

    if N_init == N and N == H * W:
        # for the initial tokens with grid structure, just reshape
        return feature_map.flatten(2).permute(0, 2, 1).contiguous()

    idx_hw = get_grid_index(
        [H_init, W_init], [H, W], device=device)[None, :].expand(B, -1)#[B,16384]

    idx_batch = torch.arange(B, device=device)[:, None].expand(B, N_init)#[B,16384]


    # choose the way with fewer flops.
    if N_init < N * H * W:
        # use sparse matrix multiplication
        # Flops: B * N_init * (C+2)
        idx_token = idx_token + idx_batch * N
        idx_hw = idx_hw + idx_batch * H * W 
        indices = torch.stack([idx_token, idx_hw], dim=0).reshape(2, -1)

        # torch.sparse do not support fp16
        with torch.cuda.amp.autocast(enabled=False):
            # sparse mm do not support gradient for sparse matrix
            value = feature_map.new_ones(B * N_init).detach().float()

            # build a sparse matrix with shape [B*N, B*H*W]
            A = torch.sparse_coo_tensor(indices, value, (B * N, B * H * W))

            # normalize the matrix
            all_weight = torch.sparse.mm(A,torch.ones([B * H * W, 1], device=device, dtype=torch.float32) ) + 1e-6 
            value = value / all_weight[idx_token.reshape(-1), 0]
            #A = A / (torch.sum(value)/B*N_init)
            A = torch.sparse_coo_tensor(indices, value, (B * N, B * H * W))
            # out: [B*N, C]
            out = torch.sparse.mm(A, feature_map.permute(0, 2, 3, 1).contiguous().reshape(B * H * W, C).float())
    else:
        # use dense matrix multiplication
        indices = torch.stack([idx_batch, idx_token, idx_hw], dim=0).reshape(3, -1)
        value = feature_map.new_ones(B * N_init).detach()  
        A = torch.sparse_coo_tensor(indices, value, (B, N, H * W)).to_dense()
        # normalize the matrix
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-6)

        out = A @ feature_map.permute(0, 2, 3, 1).reshape(B, H * W, C).contiguous()

    out = out.type(feature_map.dtype)
    out = out.reshape(B, N, C)
    return out


def token_downup(target_dict, source_dict):
    """Transform token features between different distribution.

    Returns:
        x_out (Tensor[B, N, C]): token features.

    Args:
        target_dict (dict): dict for target token information
        source_dict (dict): dict for source token information.
    """

    x_s = source_dict['x']
    idx_token_s = source_dict['idx_token']
    idx_token_t = target_dict['idx_token']
    T = target_dict['token_num']
    B, S, C = x_s.shape
    N_init = idx_token_s.shape[1]

    weight = target_dict['agg_weight'] if 'agg_weight' in target_dict.keys() else None
    if weight is None:
        weight = x_s.new_ones(B, N_init, 1)
    weight = weight.reshape(-1)

    # choose the way with fewer flops.
    if N_init < T * S:
        # use sparse matrix multiplication
        # Flops: B * N_init * (C+2)
        idx_token_t = idx_token_t + torch.arange(B, device=x_s.device)[:, None] * T
        idx_token_s = idx_token_s + torch.arange(B, device=x_s.device)[:, None] * S
        coor = torch.stack([idx_token_t, idx_token_s], dim=0).reshape(2, B * N_init)

        # torch.sparse.spmm does not support fp16
        with torch.cuda.amp.autocast(enabled=False):
            # torch.sparse does not support grad for sparse matrix
            weight = weight.float().detach()
            # build a matrix with shape [B*T, B*S]
            A = torch.sparse.FloatTensor(coor, weight, torch.Size([B * T, B * S]))
            # normalize the matrix
            all_weight = torch.sparse.mm(A.type(torch.float32), x_s.new_ones(B * S, 1).type(torch.float32)) + 1e-6
            weight = weight / all_weight[(idx_token_t).reshape(-1), 0]
            A = torch.sparse.FloatTensor(coor, weight, torch.Size([B * T, B * S]))
            # sparse matmul
            x_out = torch.sparse.mm(A.type(torch.float32),  x_s.reshape(B * S, C).type(torch.float32))
    else:
        # use dense matrix multiplication
        # Flops: B * T * S * (C+2)
        idx_batch = torch.arange(B, device=x_s.device)[:, None].expand(B, N_init)
        coor = torch.stack([idx_batch, idx_token_t, idx_token_s], dim=0).reshape(3, B * N_init)
        weight = weight.detach()  # detach to reduce training time
        # build a matrix with shape [B, T, S]
        A = torch.sparse.FloatTensor(coor, weight, torch.Size([B, T, S])).to_dense()
        # normalize the matrix
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-6)
        # dense matmul
        x_out = A @ x_s

    x_out = x_out.reshape(B, T, C).type(x_s.dtype)
    return x_out


def map2token_flops(N_init, C):
    return N_init * (2 + 1 + 1 + C)


def token2map_flops(N_init, C):
    return N_init * (2 + 1 + 1 + C)


def downup_flops(N_init, C):
    return N_init * (2 + 1 + 1 + C)


def cluster_and_merge_flops(num_tokens, dim, k):
    flops = 0
    flops += num_tokens * num_tokens * dim  # distance matrix
    flops += num_tokens * k                 # local density
    flops += num_tokens * num_tokens        # distance indicator
    flops += num_tokens * dim               # token merge
    return flops



def sra_flops(h, w, r, dim):
    return 2 * h * w * (h // r) * (w // r) * dim

def cluster_dpc_knn(token_dict, cluster_num, k=5, token_mask=None):
    """Cluster tokens with DPC-KNN algorithm.
    Return:
        idx_cluster (Tensor[B, N]): cluster index of each token.
        cluster_num (int): actual cluster number. The same with
            input cluster number
    Args:
        token_dict (dict): dict for token information
        cluster_num (int): cluster number
        k (int): number of the nearest neighbor used for local density.
        token_mask (Tensor[B, N]): mask indicate the whether the token is
            padded empty token. Non-zero value means the token is meaningful,
            zero value means the token is an empty token. If set to None, all
            tokens are regarded as meaningful.
    """
    with torch.no_grad():
        x = token_dict['x']
        B, N, C = x.shape

        dist_matrix = torch.cdist(x, x) / (C ** 0.5)# [B,N,C],[B,N,C]-->[B,N,N]
        if token_mask is not None:
            token_mask = token_mask > 0
            # in order to not affect the local density, the distance between empty tokens
            # and any other tokens should be the maximal distance.
            dist_matrix = dist_matrix * token_mask[:, None, :] + \
                          (dist_matrix.max() + 1) * (~token_mask[:, None, :])

        # get local density
        dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1, largest=False) #dist_nearest-->[B,N,K],index_nearest-->[B,N,K]
        density = (-(dist_nearest ** 2).mean(dim=-1)).exp() # density-->[B,N]
        # add a little noise to ensure no tokens have the same density.
        density = density + torch.rand(
            density.shape, device=density.device, dtype=density.dtype) * 1e-6

        if token_mask is not None:
            # the density of empty token should be 0
            density = density * token_mask

        # get distance indicator
        mask = density[:, None, :] > density[:, :, None]#mask-->[B,N,N]。
        mask = mask.type(x.dtype)#（0.0，1.=0）mask-->[B,N,N]
        dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
        dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)# #dist-->[B,N],index_parent-->[B,N]

        # select clustering center according to score
        score = dist * density # 
        _, index_down = torch.topk(score, k=cluster_num, dim=-1)# index_down-->[B,cluster_num]

        # assign tokens to the nearest center
        dist_matrix = index_points(dist_matrix, index_down) #dist_matrix-->[B,cluster_num,N]

        idx_cluster = dist_matrix.argmin(dim=1) # idx_cluster-->[B,N]

        # make sure cluster center merge to itself
        idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
        idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
        idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

    return idx_cluster, cluster_num


def merge_tokens(token_dict, idx_cluster, cluster_num, token_weight=None):
    """Merge tokens in the same cluster to a single cluster.
    Implemented by torch.index_add(). Flops: B*N*(C+2)
    Return:
        out_dict (dict): dict for output token information

    Args:
        token_dict (dict): dict for input token information
        idx_cluster (Tensor[B, N]): cluster index of each token.
        cluster_num (int): cluster number
        token_weight (Tensor[B, N, 1]): weight for each token.
    """

    x = token_dict['x']
    idx_token = token_dict['idx_token']
    agg_weight = token_dict['agg_weight']

    B, N, C = x.shape
    if token_weight is None:
        token_weight = x.new_ones(B, N, 1)

    idx_batch = torch.arange(B, device=x.device)[:, None]
    idx = idx_cluster + idx_batch * cluster_num

    all_weight = token_weight.new_zeros(B * cluster_num, 1)
    all_weight.index_add_(dim=0, index=idx.reshape(B * N),
                          source=token_weight.reshape(B * N, 1))
    all_weight = all_weight + 1e-6
    norm_weight = token_weight / all_weight[idx]

    # average token features
    x_merged = x.new_zeros(B * cluster_num, C)
    source = x * norm_weight
    x_merged.index_add_(dim=0, index=idx.reshape(B * N),
                        source=source.reshape(B * N, C).type(x.dtype))
    x_merged = x_merged.reshape(B, cluster_num, C)

    idx_token_new = index_points(idx_cluster[..., None], idx_token).squeeze(-1)
    weight_t = index_points(norm_weight, idx_token)
    agg_weight_new = agg_weight * weight_t
    agg_weight_new / agg_weight_new.max(dim=1, keepdim=True)[0]

    out_dict = {}
    out_dict['x'] = x_merged
    out_dict['token_num'] = cluster_num
    out_dict['map_size'] = token_dict['map_size']
    out_dict['init_grid_size'] = token_dict['init_grid_size']
    out_dict['idx_token'] = idx_token_new
    out_dict['agg_weight'] = agg_weight_new
    return out_dict



def vis_tokens(img, token_dict, edge_color=[1.0, 1.0, 1.0], edge_width=1):
    """Visualize tokens
    Return:
        vis_img (Tensor[B, 3, H, W]): visualize result.

    Args:
        img (Tensor[B, 3, H, W]): input image.
        token_dict (dict): dict for input token information
        edge_color (float[int]): color for edges
        edge_width (int): width for edges
    """

    N = token_dict['token_num']
    device, dtype = img.device, img.dtype

    # color_map = torch.tensor(img, device=device, dtype=float) / 255.0
    # color_map = color_map.permute(2, 0, 1)[None, ...]
    color_map = F.avg_pool2d(img, kernel_size=4)
    B, C, H, W = color_map.shape

    token_color = map2token(color_map, token_dict)
    tmp_dict = token_dict.copy()
    tmp_dict['map_size'] = [H, W]
    tmp_dict['x'] = token_color
    vis_img = token2map(tmp_dict)

    token_idx = torch.arange(N, device=device)[None, :, None].float() / N
    tmp_dict['x'] = token_idx
    idx_map = token2map(tmp_dict)  # [B, 1, H, W]

    vis_img = F.interpolate(vis_img, [H * 8, W * 8], mode='nearest')
    idx_map = F.interpolate(idx_map, [H * 8, W * 8], mode='nearest')

    kernel = idx_map.new_zeros([4, 1, 3, 3])
    kernel[:, :, 1, 1] = 1
    kernel[0, :, 0, 1] = -1
    kernel[1, :, 2, 1] = -1
    kernel[2, :, 1, 0] = -1
    kernel[3, :, 1, 2] = -1

    for i in range(edge_width):
        edge_map = F.conv2d(F.pad(idx_map, [1, 1, 1, 1], mode='replicate'), kernel)
        edge_map = (edge_map != 0).max(dim=1, keepdim=True)[0]
        idx_map = idx_map * (~edge_map) + torch.rand(idx_map.shape, device=device, dtype=dtype) * edge_map

    edge_color = torch.tensor(edge_color, device=device, dtype=dtype)[None, :, None, None]
    vis_img = vis_img * (~edge_map) + edge_color * edge_map
    return vis_img


def get_token_density_map(token_dict):
    N = token_dict['token_num']
    idx_token = token_dict['idx_token']
    B, N_init = idx_token.shape
    device = idx_token.device
    idx_batch = torch.arange(B, device=device)[:, None].expand(B, N_init)

    coor = torch.stack([idx_batch, idx_token], dim=0).reshape(2, B * N_init)
    tmp = torch.ones(B * N_init, device=device)
    token_density = 1 / torch.sparse.FloatTensor(coor, tmp, torch.Size([B, N])).to_dense()
    tmp_dict = token_dict.copy()
    tmp_dict['x'] = token_density[..., None]
    density_map = token2map(tmp_dict)
    return density_map

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def load_checkpoint(model,
                    filename,
                    revise_keys=[(r'^module\.', '')]):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.
        revise_keys (list): A list of customized keywords to modify the
            state_dict in checkpoint. Each item is a (pattern, replacement)
            pair of the regular expression operations. Default: strip
            the prefix 'module.' by [(r'^module\\.', '')].


    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    assert os.path.isfile(filename), '{} is not a check file'.format(filename)
    checkpoint = torch.load(filename)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    # get state_dict from checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    # strip prefix of state_dict
    for p, r in revise_keys:
        state_dict = {re.sub(p, r, k): v for k, v in state_dict.items()}
    # load state_dict
    model.load_state_dict(state_dict)
    return checkpoint

