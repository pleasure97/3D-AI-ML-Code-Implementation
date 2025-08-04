from dataclasses import dataclass
import gsplat
from jaxtyping import Float
from torch import Tensor
import torch
from src.model import ModuleWithConfig

@dataclass
class RenderConfig:
    scale_modifier: float
    prefiltered: bool
    debug: bool


class GaussianRenderer(ModuleWithConfig[RenderConfig]):
    def __init__(self, config: RenderConfig):
        super().__init__(config)
        self.config = config

    def render(
            self,
            extrinsics: Float[Tensor, "batch 4 4"],
            intrinsics: Float[Tensor, "batch 3 3"],
            image_shape: tuple[int, int],
            colors: Float[Tensor, "batch 3"],
            gaussian_means: Float[Tensor, "batch gaussian 3"],
            gaussian_covariances: Float[Tensor, "batch gaussian 3 sh_degree"],
            gaussian_opacities: Float[Tensor, "batch gaussian"],
            background_white: True
    ) -> Float[Tensor, "batch 3 height width"]:

        batch_size, views = extrinsics.shape[:2]
        height, width = image_shape[-2], image_shape[-1]
        num_gaussians = gaussian_means.shape[1]
        device = gaussian_means.device

        if background_white:
            background_color = torch.zeros((batch_size, 3), device=device)
        else:
            background_color = torch.ones((batch_size, 3), device=device)

        images = torch.zeros(batch_size, views, 3, height, width, device=device, dtype=torch.float32)

        for batch in range(batch_size):
            for view in range(views):
                means = gaussian_means[batch]
                covariances = gaussian_covariances[batch]
                opacities = gaussian_opacities[batch]
                batch_colors = colors[batch]

                batch_extrinsics = extrinsics[batch, view : view + 1]
                batch_intrinsics = intrinsics[batch, view : view + 1]
                background = background_color[batch : batch + 1]

                quaternions = torch.zeros(num_gaussians, 4, device=device)
                scales = torch.zeros(num_gaussians, 3, device=device)

                # gsplat doesn't support mixed precision training
                output_colors, output_alphas, meta = gsplat.rasterization(
                    means=means.float(),
                    quats=quaternions,
                    scales=scales,
                    opacities=opacities.float(),
                    colors=batch_colors.float(),
                    viewmats=batch_extrinsics.float(),
                    Ks=batch_intrinsics.float(),
                    width=width,
                    height=height,
                    covars=covariances.float(),
                    backgrounds=background.float(),
                    sh_degree=None,
                    packed=False,
                    render_mode='RGB')

                image = output_colors[0].permute(2, 0, 1)  # [3, height, width]
                images[batch, view] = image

        return images
