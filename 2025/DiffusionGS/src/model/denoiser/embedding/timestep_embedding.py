from dataclasses import dataclass
from typing import Literal
import torch
import torch.nn as nn
import math
from src.model import ModuleWithConfig

@dataclass
class TimestepEmbeddingConfig:
    name: Literal["timestep_embedding"]
    time_dim: int
    max_period: int

@dataclass
class TimestepMLPConfig:
    name: Literal["timestep_mlp"]
    embedding: TimestepEmbeddingConfig
    out_dim: int

class TimestepEmbedding(ModuleWithConfig[TimestepEmbeddingConfig]):
    def __init__(self, config: TimestepEmbeddingConfig):
        super().__init__(config)
        self.config = config
        self.time_dim = self.config.time_dim
        self.max_period = self.config.max_period

    def forward(self, x):
        device = x.device
        half_dim = self.time_dim // 2
        embedding = math.log(self.max_period) / (half_dim - 1)
        embedding = torch.exp(torch.arange(half_dim, device=device) * -embedding)
        # x.shape : [batch_size, num_timesteps]
        # embedding.shape : [half_dim]
        embedding = x[..., None] * embedding[None, None, :]
        embedding = torch.cat((embedding.sin(), embedding.cos()), dim=-1)

        return embedding


class TimestepMLP(ModuleWithConfig[TimestepMLPConfig]):
    def __init__(self, config: TimestepMLPConfig):
        super().__init__(config)

        self.config = config
        self.time_mlp = nn.Sequential(
            TimestepEmbedding(self.config.embedding),
            nn.Linear(self.config.embedding.time_dim, self.config.out_dim),
            nn.GELU(),
            nn.Linear(self.config.out_dim, self.config.out_dim))

    def forward(self, x):
        return self.time_mlp(x)
