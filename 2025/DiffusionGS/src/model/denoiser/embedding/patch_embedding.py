from dataclasses import dataclass
from typing import Literal
import torch.nn as nn
from src.model import ModuleWithConfig

@dataclass
class PatchEmbeddingConfig:
    in_channels: int
    patch_size: int
    embedding_dim: int

@dataclass
class PatchMLPConfig:
    name: Literal["patch_mlp"]
    embedding: PatchEmbeddingConfig
    out_dim: int

class PatchEmbedding(ModuleWithConfig[PatchEmbeddingConfig]):
    """Turns a 2D input image into a 1D sequence learnable embedding vector.

    Args:
        in_channels (int): Number of color channels for the input images. Defaults to 3.
        patch_size (int): Size of patches to convert input image into. Defaults to 16.
        embedding_dim (int): Size of embedding to turn image into. Defaults to 768.
    """

    def __init__(self,
                 config: PatchEmbeddingConfig):
        super().__init__()

        self.config = config

        self.in_channels = self.config.in_channels
        self.patch_size = self.config.patch_size
        self.embedding_dim = self.config.embedding_dim

        self.patcher = nn.Conv2d(in_channels=self.in_channels,
                                 out_channels=self.embedding_dim,
                                 kernel_size=self.patch_size,
                                 stride=self.patch_size,
                                 padding=0)

        self.flatten = nn.Flatten(start_dim=2, end_dim=-1)

    def forward(self, x):
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)
        return x_flattened.permute(0, 2, 1)

class PatchMLP(ModuleWithConfig[PatchMLPConfig]):
    def __init__(self, config: PatchMLPConfig):
        super().__init__()

        self.config = config

        self.patch_mlp = nn.Sequential(
            PatchEmbedding(self.config.embedding),
            nn.Linear(self.config.embedding.embedding_dim, self.config.out_dim),
            nn.GELU(),
            nn.Linear(self.config.out_dim, self.config.out_dim))

    def forward(self, x):
        return self.patch_mlp(x)
