from fast_gauss import GaussianRasterizationSettings, GaussianRasterizer
from jaxtyping import Float
from torch import Tensor
from src.utils.geometry_util import get_fov


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


    settings = GaussianRasterizationSettings(
        image_height=,
        image_width=,
        tanfovx=,
        tanfovy=,
        bg=,
        scale_modifier=,
        viewmatrix=,
        projmatrix=,
        sh_degree=,
        campos=,
        prefiltered=,
        debug=
    )

    rasterizer = GaussianRasterizer(settings)
    image, radii = rasterizer(
        means3D=,
        means2D=,
        shs=None,
        colors_precomp=,
        opacities=,
        cov3D_precomp=,)