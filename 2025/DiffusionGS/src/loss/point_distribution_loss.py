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
                rays_d: Float[Tensor, "batch height * width 3"],
                timesteps: int) -> Float:
        l_ts = []
        for timestep in range(timesteps):
            u_t = self.config.sigma_0 * u_near + (1 - self.config.sigma_0) * u_far
            l_t = torch.append(u_t * rays_d[timestep])
            l_ts.append(l_t)

        l_t_tensor = torch.stack(l_ts)
        mean_l_t = torch.mean(l_t_tensor)
        var_l_t = torch.var(l_t_tensor, unbiased=False)
        std_l_t = torch.sqrt(var_l_t)

        abs_rays_o = [torch.abs(ray_o) for ray_o in rays_o]
        mean_abs_o = torch.mean(torch.stack(abs_rays_o))

        point_distribution_loss = torch.mean(l_t_tensor -
                                             ((l_t_tensor - mean_l_t) / std_l_t * self.config.sigma_0 + mean_abs_o))

        return point_distribution_loss
