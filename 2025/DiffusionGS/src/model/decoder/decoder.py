from dataclasses import dataclass
from typing import Literal
from jaxtyping import Float
from src.model import ModuleWithConfig
import torch
import torch.nn as nn
from torch import Tensor
from src.model.types import Gaussians
from src.utils.geometry_util import multiply_scaling_rotation


@dataclass
class GaussianDecoderConfig:
    name: Literal["object_decoder"] | Literal["scene_decoder"]
    u_near: float
    u_far: float
    input_dim: int
    hidden_dim: int
    output_dim: int
    weight: float
    num_points: int


class GaussianDecoder(ModuleWithConfig[GaussianDecoderConfig]):
    def __init__(self, config: GaussianDecoderConfig):
        super().__init__(config)

        self.config = config

        self.register_buffer("u_near", torch.tensor(self.config.u_near, dtype=torch.float32))
        self.register_buffer("u_far", torch.tensor(self.config.u_far, dtype=torch.float32))

        self.num_points = self.config.num_points

        self.mlp1 = nn.Sequential(
            nn.Linear(self.config.input_dim, self.config.hidden_dim),
            nn.ReLU())

        self.mlp2 = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.output_dim))

        # weight to control center
        self.weight = self.config.weight

    def forward(self,
                x : Float[Tensor, "batch num_tokens input_dim"],
                timestep_embedding: Float[Tensor, "batch time_dim"]) -> Gaussians:
        output1 = self.mlp1(x)
        output1 = output1 + timestep_embedding.mean(dim=1, keepdim=True)

        output2 = self.mlp2(output1)

        # Positions are clipped to [-1, 1]^3
        positions = torch.tanh(output2[:, :, :3])

        # Depth Transition
        depth = (self.weight * self.u_near + (1 - self.weight) * self.u_far).to(positions.device)
        if depth.dim() == 0:
            depth = depth.view(1, 1).expand(positions.shape[0], positions.shape[1])

        depth = depth.unsqueeze(-1)
        positions = torch.cat([positions[..., :2], depth], dim=-1)

        # Color is in [0, 1]
        colors = torch.sigmoid(output2[:, :, 3:6])

        # Scale > 0
        scale = torch.exp(output2[:, :, 6:9])

        # Rotation
        quaternion = output2[:, :, 9:13]

        # Adjust dimension to fit multiply_scaling_rotation()
        batch, num_points, _ = scale.shape
        scale = scale.view(batch * num_points, 3)
        quaternion = quaternion.view(batch * num_points, 4)

        scaling_rotation_matrix = multiply_scaling_rotation(scale, quaternion)
        scaling_rotation_matrix = scaling_rotation_matrix.view(batch, num_points, 3, 3)

        covariances = torch.matmul(scaling_rotation_matrix, scaling_rotation_matrix.transpose(-1, -2))

        # Opacity
        opacities = torch.sigmoid(output2[:, :, 9:10]).squeeze(-1)

        return positions, covariances, colors, opacities
