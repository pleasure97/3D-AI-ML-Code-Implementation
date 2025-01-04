import torch
import torch.nn as nn
import numpy as np
from kornia import create_meshgrid

class HashEncoder(nn.Module):
    def __init__(self, bounding_box, resolution_levels = 16, features_per_level = 2,
                 log2_hashmap_size = 2 ** 19, coarse_resolution = 16, fine_resolution = 512):
        super().__init__()

        self.bounding_box = bounding_box
        self.resolution_levels = resolution_levels
        self.features_per_level = features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.coarse_resolution = torch.tensor(coarse_resolution)
        self.fine_resolution = torch.tensor(fine_resolution)

        self.out_dim = self.resolution_levels + self.features_per_level
        self.b \
            = torch.exp((torch.log(self.fine_resolution)) - torch.log(self.coarse_resolution)) / (resolution_levels - 1)

        ### fix it to suit for tiny cuda nn !!!!!!!!!!!!!
        self.embedding = nn.ModuleList([nn.Embedding(self.log2_hashmap_size, self.features_per_level) \
                                        for i in range(len(resolution_levels))])

        # uniform initialization
        for i in range(resolution_levels):
            nn.init.uniform_(self.embedding[i].weight, a = -0.0001, b = 0.0001)

    def interpolate(self, x, min_vertex, max_vertex, voxel_embedding):

        '''
        x: B X 3
        min_vertex: B X 3
        max_vertex: B X 3
        voxel_embedding: B X 8 X2
        '''

        weights = (x - min_vertex) / (max_vertex - min_vertex)

        c00 = voxel_embedding[:, 0] * (1 - weights[:, 0][:, None]) + voxel_embedding[:, 4] * weights[:, 0][:, None]
        c01 = voxel_embedding[:, 1] * (1 - weights[:, 0][:, None]) + voxel_embedding[:, 5] * weights[:, 0][:, None]
        c10 = voxel_embedding[:, 2] * (1 - weights[:, 0][:, None]) + voxel_embedding[:, 6] * weights[:, 0][:, None]
        c11 = voxel_embedding[:, 3] * (1 - weights[:, 0][:, None]) + voxel_embedding[:, 7] * weights[:, 0][:, None]

        c0 = c00 * (1 - weights[:, 1][:, None]) + c10 * weights[:, 1][:, None]
        c1 = c01 * (1 - weights[:, 1][:, None]) + c11 * weights[:, 1][:, None]

        c = c0 * (1 - weights[:, 2][:, None]) + c1 * weights[:, 2][:, None]


        return c

    def forward(self, x):
        # x : B X 3
        x_embedded_list = []

        for i in range(self.resolution_levels):
            resolution = torch.floor(self.coarse_resolution * self.b ** i)
            min_vertex, max_vertex, hash_indices, keep_mask \
                = get_vertices(x, self.bounding_box, resolution, self.log2_hashmap_size)

            voxel_embedding = self.embeding[i](hash_indices)

            x_embedded = self.interpolate(x, min_vertex, max_vertex, voxel_embedding)
            x_embedded_list.append(x_embedded)

        keep_mask = (keep_mask.sum(dim = -1) == keep_mask.shape[-1])
        return torch.cat(x_embedded_list, dim = -1), keep_mask


def get_multires_hash_encoding(args, encoder=HashEncoder):
    '''
    Returns a multiresolutional hash encoding and output dimension.
    '''

    embedded = encoder(bounding_box=args.bounding_box, \
                       log2_hashmap_size=args.log2_hashmap_size, \
                       finest_resolution=args.finest_resolution)

    out_dim = embedded.out_dim

    return embedded, out_dim


def hierarchical_sampling(bins, w_i, num_coarse_samples, use_uniform=False):
    pdf = (w_i + 1e-5) / torch.sum(w_i, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

    if use_uniform:
        u = torch.linspace(0., 1., num_coarse_samples)
        u = u.expand(list(cdf.shape[:-1]) + [num_coarse_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [num_coarse_samples])

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

    dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1)

    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., None, :] * cam2world[:3, :3], -1)
    # Translate camera frame's origin
    rays_o = cam2world[:3, -1].expand(rays_d.shape)

    return rays_o, rays_d


def get_rays_np(height, width, K, c2w):
    i, j = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i - K[0][2]), - (j - K[1][2]) / K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    # Translate camera frame's origin to he world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))

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
    d1 = -2. * focal / height * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
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
        xyz = torch.clamp(xyz, min=box_min, max=box_max)

    grid_size = (box_max - box_min) / resolution
    bottom_left = torch.floor((xyz - box_min) / grid_size).int()

    min_vertex = bottom_left * grid_size + box_min
    max_vertex = min_vertex + torch.tensor([1.0, 1.0, 1.0]) * grid_size

    voxel_indices = bottom_left.unsqueeze(1) + \
                    torch.tensor([[[i, j, k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]], device='cuda')
    hash_indices = hash(voxel_indices, log2_hashmap_size)

    return min_vertex, max_vertex, hash_indices, keep_mask


def img2mse(rgb, target):
    return torch.mean((rgb - target) ** 2)


def mse2psnr(mse):
    return -10. * torch.log(mse) / torch.log(torch.Tensor([10.]))


def hash(coords, log2_hashmap_size):
    primes = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]

    xor_result = torch.zeros_like(coords)[..., 0]

    for i in range(coords.shape[-1]):
        xor_result ^= coords[..., i] * primes[i]

    return torch.tensor((1 << log2_hashmap_size) - 1).to(xor_result.device) & xor_result


def to8bit(x):
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)


def get_ray_directions(height, width, focal):
    grid = create_meshgrid(height, width, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)
    directions = \
        torch.stack([(i - width / 2) / focal, -(j - height / 2) / focal, -torch.ones_like(i)], -1)

    direction_bounds = directions.view(-1, 3)

    return directions


def get_rays_origin_and_direction(directions, c2w):
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:3, :3].T  # (height, width, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:3, -1].expand(rays_d.shape)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d


def get_ndc_rays_origin_and_direction(height, width, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Store some intermediate homogeneous results
    ox_oz = rays_o[..., 0] / rays_o[..., 2]
    oy_oz = rays_o[..., 1] / rays_o[..., 2]

    # Projection
    o0 = -1. / (width / (2. * focal)) * ox_oz
    o1 = -1. / (height / (2. * focal)) * oy_oz
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (width / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - ox_oz)
    d1 = -1. / (height / (2. * focal)) * (rays_d[..., 1] / rays_d[...,2 ] - oy_oz)
    d2 = 1 - o2

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d

