from dataclasses import dataclass
import torch
import torch.nn as nn
from ..types import Gaussians


@dataclass
class GaussianDecoderConfig:
    u_near: float
    u_far: float
    d_int: int
    d_hidden: int
    weight: float
    d_out: int


class GaussianDecoder(nn.Module):
    def __init__(self, transformer_output_tensor: torch.Tensor, u_near: float, u_far: float,
                 k: int=100, input_dim: int=768, hidden_dim: int=768, output_dim: int=14, weight: float=0.5):
        super().__init__()

        self.transformer_output_tensor = transformer_output_tensor

        self.u_near = torch.tensor(u_near, dtype=torch.float32)
        self.u_far = torch.tensor(u_far, dtype=torch.float32)

        self.k = k

        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU())

        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim))

        # weight to control center
        self.weight = torch.full((k,), weight, dtype=torch.float32)

    def forward(self, x) -> Gaussians:
        output1 = self.mlp1(x)

        output1 = output1 + self.transformer_output_tensor.mean(dim=1, keepdim=True)

        output2 = self.mlp2(output1)

        # Positions are clipped to [-1, 1]^3
        position = torch.tanh(output2[:, :, :3])

        # Depth Transition
        depth = self.weight * self.u_near + (1 - self.weight) * self.u_far
        depth = depth.view(1, self.k, 1)
        position[:, :, 2:3] = depth

        # Color is in [0, 1]
        color = torch.sigmoid(output2[:, :, 3:7])

        # Scale > 0
        scale = torch.exp(output2[:, :, 7:10])

        # Rotation
        rotation = output2[:, :, 10:13]

        # Opacity
        opacity = torch.sigmoid(output2[:, :, 13:14])

        return Gaussians(position, rotation, )
