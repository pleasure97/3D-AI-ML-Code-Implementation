from dataclasses import dataclass
import torch
from src.loss.base_loss import BaseLoss
from jaxtyping import Float
from torch import Tensor


@dataclass
class PointDistributionLossConfig:
    name: str
    sigma_0: float


class PointDistributionLoss(BaseLoss[PointDistributionLossConfig]):
    def __init__(self, config: PointDistributionLossConfig) -> None:
        super().__init__(config)

    def forward(self,
                positions: Float[Tensor, "batch num_points 3"],
                extrinsics: Float[Tensor, "batch num_views 4 4"],) -> Float:
        batch_size, num_points, _ = positions.shape
        device = positions.device
        dtype = positions.dtype

        _, num_views, _, _ = extrinsics.shape

        # Transform Positions from World Space to Camera Space
        ones = torch.ones((batch_size, num_points, 1), dtype=dtype, device=device)
        homogeneous_positions = torch.cat([positions, ones], dim=-1)  # [batch_size, num_points, 4]
        homogeneous_positions_expanded = homogeneous_positions.unsqueeze(1).expand(-1, num_views, -1, -1)
        homogeneous_positions_unsqueezed = homogeneous_positions_expanded.unsqueeze(-1)

        # Reshape for batched matrix multiplication
        camera_coordinates = torch.matmul(extrinsics.unsqueeze(2), homogeneous_positions_unsqueezed).squeeze(-1)

        # Camera Space Depth
        depths = camera_coordinates[..., 2]  # [batch_size, num_views, num_points]
        l_t = torch.abs(depths)

        # Calculate Mean, Variance, and Standard Deviation of l_t
        mean_l_t = l_t.mean(dim=-1, keepdim=True)   # [batch_size, num_views, 1]
        var_l_t = l_t.var(dim=-1, unbiased=False, keepdim=True) # [batch_size, num_views, 1]
        std_l_t = torch.sqrt(var_l_t + 1e-6)

        # Normalize and Mean Rays
        normalized_positions = positions.norm(dim=-1)
        mean_abs_o = normalized_positions.mean(dim=-1, keepdim=True) # [batch_size, 1]
        mean_abs_o = mean_abs_o.unsqueeze(-1).expand(-1, num_views, -1) # [batch_size, num_views, 1]

        # Calculate Point Distribution Loss
        point_distribution_loss_equation = l_t - ((mean_l_t / (std_l_t + 1e-6)) * self.config.sigma_0 + mean_abs_o)
        point_distribution_loss_per_view = point_distribution_loss_equation.mean(dim=-1)    # [batch_size, num_views]
        point_distribution_loss = point_distribution_loss_per_view.mean()

        return point_distribution_loss
