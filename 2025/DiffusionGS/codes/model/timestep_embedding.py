import torch
import torch.nn as nn
import math

class TimestepEmbedding(nn.Module):
  def __init__(self, time_dim: int, max_period: int=10_000):
    super().__init__()
    self.time_dim = time_dim
    self.max_period = max_period

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

class TimestepMLP(nn.Module):
  def __init__(self, fourier_dim: int, time_dim: int):
    super().__init__()

    self.time_mlp = nn.Sequential(
      TimestepEmbedding(fourier_dim),
      nn.Linear(fourier_dim, time_dim),
      nn.GELU(),
      nn.Linear(time_dim, time_dim))

  def forward(self, x):
    return self.time_mlp(x)
