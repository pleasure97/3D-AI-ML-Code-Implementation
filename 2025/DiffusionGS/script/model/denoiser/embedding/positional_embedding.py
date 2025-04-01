import torch
from torch import nn


class PositionalEmbedding(nn.Module):
    def __init__(self, num_patches: int, embedding_dim: int, device: torch.device):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.ones(1, num_patches, embedding_dim), requires_grad=True
        ).to(device)

    def forward(self):
        return self.positional_embedding
