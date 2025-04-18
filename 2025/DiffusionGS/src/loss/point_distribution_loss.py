from dataclasses import dataclass
import torch
from src.loss import Loss
from jaxtyping import Float
from torch import Tensor


@dataclass
class PointDistributionLossConfig:
    sigma_0: float


class PointDistributionLoss(Loss[PointDistributionLossConfig]):
    def forward(self,
                weight_u: float, u_near: float, u_far: float,
                rays_o: Float[Tensor], rays_d: Float[Tensor], timesteps: int) -> Float[Tensor]:
        l_ts = []
        for timestep in range(timesteps):
            u_t = weight_u * u_near + (1 - weight_u) * u_far
            l_t = torch.append(u_t * rays_d[timestep])
            l_ts.append(l_t)

        l_t_tensor = torch.stack(l_ts)
        mean_l_t = torch.mean(l_t_tensor)
        var_l_t = torch.var(l_t_tensor, unbiased=False)
        std_l_t = torch.sqrt(var_l_t)

        abs_rays_o = [torch.abs(ray_o) for ray_o in rays_o]
        mean_abs_o = torch.mean(torch.stack(abs_rays_o))

        point_distribution_loss = torch.mean(l_t_tensor -
                                             ((l_t_tensor - mean_l_t) / std_l_t * self.config.sigma_0 + abs_rays_o))

        return point_distribution_loss
