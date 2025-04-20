from dataclasses import dataclass
from typing import Literal
import torch
from torch import nn

@dataclass
class PositionalEmbeddingConfig:
    name: Literal["positional_embedding"]
    num_patches: int
    embedding_dim: int

class PositionalEmbedding(nn.Module, PositionalEmbeddingConfig):
    def __init__(self, config: PositionalEmbeddingConfig, num_patches: int, embedding_dim: int, device: torch.device):
        super().__init__()
        self.config = config
        self.positional_embedding = nn.Parameter(
            torch.ones(1, self.config.num_patches, self.config.embedding_dim), requires_grad=True
        ).to(device)

    def forward(self):
        return self.positional_embedding
