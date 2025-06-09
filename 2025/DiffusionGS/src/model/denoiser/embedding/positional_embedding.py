import torch
from torch import nn

class PositionalEmbedding(nn.Module):
    def __init__(self, num_patches: int, embedding_dim: int):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.ones(1, num_patches, embedding_dim), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.positional_embedding
