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
                u_near: float, u_far: float,
                rays_o: Float[Tensor, "batch height * width 3"],
                rays_d: Float[Tensor, "batch height * width 3"]) -> Float:
        # Calculate u_t related to u_near and u_far
        u_t = self.config.sigma_0 * u_near + (1 - self.config.sigma_0) * u_far
        # Multiply each ray direction vector
        l_t = u_t * rays_d
        # Calculate statistics per pixel
        mean_l_t = torch.mean(l_t)
        var_l_t = torch.var(l_t, unbiased=False)
        std_l_t = torch.sqrt(var_l_t)

        mean_abs_o = torch.mean(torch.abs(rays_o))

        point_distribution_loss = torch.mean(l_t -
                                             ((l_t - mean_l_t) / std_l_t * self.config.sigma_0 + mean_abs_o))

        return point_distribution_loss
