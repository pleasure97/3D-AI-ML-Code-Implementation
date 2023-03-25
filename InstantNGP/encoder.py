import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_vertices


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


