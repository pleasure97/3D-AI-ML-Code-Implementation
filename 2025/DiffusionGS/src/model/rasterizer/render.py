from dataclasses import dataclass
from fast_gauss import GaussianRasterizationSettings, GaussianRasterizer
from jaxtyping import Float
from torch import Tensor
import torch
from src.model import ModuleWithConfig
from src.utils.geometry_util import get_fov, make_projection_matrix

@dataclass
class RenderConfig:
    sh_degree: int
    scale_modifier: float
    prefiltered: bool
    debug: bool

class GaussianRenderer(ModuleWithConfig[RenderConfig]):
    def __init__(self, config: RenderConfig):
        super().__init__(config)
        self.config = config
        self.rasterizer = None

    def render(
            self,
            extrinsics: Float[Tensor, "batch 4 4"],
            intrinsics: Float[Tensor, "batch 3 3"],
            near: Float[Tensor, "batch"],
            far: Float[Tensor, "batch"],
            image_shape: tuple[int, int],
            background_color: Float[Tensor, "batch 3"],
            colors: Float[Tensor, "batch 3"],
            gaussian_means: Float[Tensor, "batch gaussian 3"],
            gaussian_covariances: Float[Tensor, "batch gaussian 3 sh_degree"],
            gaussian_opacities: Float[Tensor, "batch gaussian"],
    ) -> Float[Tensor, "batch 3 height width"]:
        batch_size, gaussians, _, _ = extrinsics.shape
        height, width = image_shape

        # FOV & Projection
        fov_x, fov_y = get_fov(intrinsics).unbind(dim=-1)
        tan_fov_x = (0.5 * fov_x).tan()
        tan_fov_y = (0.5 * fov_y).tan()

        projection_matrix = make_projection_matrix(near, far, tan_fov_x, tan_fov_y)

        # Compute 2D means from 3D Gaussian Centers
        ones = torch.ones(batch_size, gaussians, 1, device=gaussian_means.device)
        homogeneous_points = torch.cat([gaussian_means, ones], dim=-1)
        camera_points = (extrinsics @ homogeneous_points.unsqueeze(-1)).squeeze(-1)

        # Camera to Clip
        clip = (projection_matrix @ camera_points.unsqueeze(-1)).squeeze(-1)
        normalized_coordinates = clip[..., :3] / clip[..., 3:].clamp(min=1e-6)
        pixel_x = (normalized_coordinates[..., 0] * 0.5 + 0.5) * width
        pixel_y = (normalized_coordinates[..., 1] * 0.5 + 0.5) * height
        means2D = torch.stack([pixel_x, pixel_y], dim=-1)

        background = background_color[:, :, None, None].expand(-1, -1, height, width)

        settings = GaussianRasterizationSettings(
            image_height=height,
            image_width=width,
            tanfovx=tan_fov_x,
            tanfovy=tan_fov_y   ,
            bg=background,
            scale_modifier=self.config.scale_modifier,
            viewmatrix=extrinsics,
            projmatrix=projection_matrix,
            sh_degree=self.config.sh_degree,
            campos=extrinsics[:, :3, 3],
            prefiltered=self.config.prefiltered,
            debug=self.config.debug
        )

        if self.rasterizer is None:
            self.rasterizer = GaussianRasterizer(settings)
        image, radii = self.rasterizer(
            means3D=gaussian_means,
            means2D=means2D,
            shs=None,
            colors_precomp=colors,
            opacities=gaussian_opacities,
            cov3D_precomp=gaussian_covariances)

        return image
