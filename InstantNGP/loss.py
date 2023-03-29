import math
import torch
from utils import hash


def total_variation_loss(embeddings, min_resolution, max_resolution, level, log2_hashmap_size, num_levels=16):
    # Get resolution
    b = math.exp(math.log(max_resolution) - math.log(min_resolution) / (num_levels - 1))
    resolution = torch.tensor(math.floor(min_resolution * b ** level))

    # Get cube size to apply loss
    min_cube_size = min_resolution - 1
    max_cube_size = 50

    assert (min_cube_size <= max_cube_size)
    cube_size = torch.floor(torch.clip(resolution / 10., min_cube_size, max_cube_size)).int()

    # Sample cuboid
    min_vertex = torch.randint(0, resolution - cube_size, (3,))
    idx = min_vertex + torch.stack([torch.arange(cube_size + 1) for _ in range(3)], dim=-1)
    cube_indices = torch.stack(torch.meshgrid(idx[:, 0], idx[:, 1], idx[:, 2]), dim=-1)

    hashed_indices = hash(cube_indices, log2_hashmap_size)
    cube_embeddings = embeddings(hashed_indices)

    # Compute loss
    tv_x = torch.pow(cube_embeddings[1:, :, :, :] - cube_embeddings[:-1, :, :, :], 2).sum()
    tv_y = torch.pow(cube_embeddings[:, 1:, :, :] - cube_embeddings[:, :-1, :, :], 2).sum()
    tv_z = torch.pow(cube_embeddings[:, :, 1:, :] - cube_embeddings[:, :, :-1, :], 2).sum()

    return (tv_x + tv_y + tv_z) / cube_size


def sigma_sparsity_loss(sigmas):
    return torch.log(1. + 2 * sigmas ** 2).sum(dim=-1)
