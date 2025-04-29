from dataclasses import dataclass
from typing import Literal
import torch
from torch import nn
from src.model import ModuleWithConfig

@dataclass
class PositionalEmbeddingConfig:
    name: Literal["positional_embedding"]
    num_patches: int
    embedding_dim: int

class PositionalEmbedding(ModuleWithConfig[PositionalEmbeddingConfig]):
    def __init__(self, config: PositionalEmbeddingConfig):
        super().__init__()
        self.config = config
        self.positional_embedding = nn.Parameter(
            torch.ones(1, self.config.num_patches, self.config.embedding_dim), requires_grad=True)

    def forward(self):
        return self.positional_embedding
