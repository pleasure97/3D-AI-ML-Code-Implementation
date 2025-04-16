from fast_gauss import GaussianRasterizationSettings, GaussianRasterizer
from jaxtyping import Float
from torch import Tensor
import torch
from src.utils.geometry_util import get_fov, make_projection_matrix


def render(
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
    batch_size, _, _ = extrinsics.shape
    height, width = image_shape

    fov_x, fov_y = get_fov(intrinsics).unbind(dim=-1)
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    projection_matrix = make_projection_matrix(near, far, tan_fov_x, tan_fov_y)

    camera_pose = extrinsics[:, :3, 3]

    # TODO - sh_degree, shs, scale_modifier, prefiltered, debug (Maybe Use Config dataclass)
    sh_degree = 3

    # +0 means to connect computational graph and prevent in-place operation
    means2D = torch.zeros_like(gaussian_means, dtype=gaussian_means.dtype, requires_grad=True, device="cuda") + 0
    try:
        means2D.retain_grad()
    except:
        pass

    settings = GaussianRasterizationSettings(
        image_height=height,
        image_width=width,
        tanfovx=tan_fov_x,
        tanfovy=tan_fov_y,
        bg=background_color,
        scale_modifier=1.,
        viewmatrix=extrinsics,
        projmatrix=projection_matrix,
        sh_degree=sh_degree,
        campos=camera_pose,
        prefiltered=False,
        debug=False
    )

    rasterizer = GaussianRasterizer(settings)
    image, radii = rasterizer(
        means3D=gaussian_means,
        means2D=means2D,
        shs=None,
        colors_precomp=colors,
        opacities=gaussian_opacities,
        cov3D_precomp=gaussian_covariances, )

    return image
