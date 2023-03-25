import torch
import numpy as np

def get_multires_hash_encoding(args, encoder=HashEncoder):
    '''
    Returns a multiresolutional hash encoding and output dimension.
    '''

    embedded = encoder(bounding_box=args.bounding_box, \
                       log2_hashmap_size=args.log2_hashmap_size, \
                       finest_resolution=args.finest_resolution)

    out_dim = embedded.out_dim

    return embedded, out_dim


def hierarchical_sampling(bins, w_i, num_samples, use_uniform=False):
    pdf = (w_i + 1e-5) / torch.sum(w_i, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

    if use_uniform:
        u = torch.linspace(0., 1., num_samples)
        u = u.expand(list(cdf.shape[:-1]) + [num_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [num_samples])

    u = u.contiguous()

    indices = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(indices - 1), indices - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(indices), indices)

    indices_gathered = torch.stack([below, above], -1)

    #
    matched_shape = [indices_gathered.shape[0], indices_gathered.shape[1], cdf.shape[-1]]
    cdf_gathered = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, indices_gathered)
    bins_gathered = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, indices_gathered)

    denom = (cdf_gathered[..., 1] - cdf_gathered[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)

    #
    t = (u - cdf_gathered[..., 0]) / denom
    samples = bins_gathered[..., 0] + t * (bins_gathered[..., 1] - bins_gathered[..., 0])

    return samples


def get_rays(height, width, K, cam2world):
    i, j = torch.meshgrid(torch.linspace(0, width - 1, width), torch.linspace(0, height - 1, height))
    i = i.t()
    j = j.t()

    dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j-K[1][2]) / K[1][1], -torch.ones_like(i)], -1)

    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., None, :] * cam2world[:3, :3], -1)
    # Translate camera frame's origin
    rays_o = cam2world[:3, -1].expand(rays_d.shape)

    return rays_o, rays_d


def ndc_rays(height, width, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -2. * focal / width * rays_o[..., 0] / rays_o[..., 2]
    o1 = -2. * focal / height * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -2. * focal / width * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -2. * focal / height * (rays_d[..., 1] / rays_d[...,2 ] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


def get_vertices(xyz, bounding_box, resolution, log2_hashmap_size):
    '''
    xyz: B X 3. 3D coordinates of samples
    bounding_box: min and max x,y,z coordinates of object bbox
    '''

    box_min, box_max = bounding_box

    keep_mask = (xyz == torch.max(torch.min(xyz, box_max), box_min))

    if not torch.all(xyz <= box_max) or not torch.all(xyz >= box_min):
        xyz = torch.clamp(xyz, min = box_min, max = box_max)

    grid_size = (box_max - box_min) / resolution
    bottom_left = torch.floor((xyz - box_min) / grid_size).int()

    min_vertex = bottom_left * grid_size + box_min
    max_vertex = min_vertex + torch.tensor([1.0, 1.0, 1.0]) * grid_size

    voxel_indices = bottom_left.unsqueeze(1) + \
        torch.tensor([[[i, j, k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]], device = 'cuda')
    hash_indices = hash(voxel_indices, log2_hashmap_size)

    return min_vertex, max_vertex, hash_indices, keep_mask

