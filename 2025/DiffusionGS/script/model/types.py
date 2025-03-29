from dataclasses import dataclass
from jaxtyping import Float
from torch import Tensor


@dataclass
class Gaussians:
    positions: Float[Tensor, "batch gaussian dim"]
    covariances: Float[Tensor, "batch gaussian dim dim"]
    colors: Float[Tensor, "batch gaussian 3"]
    opacities: Float[Tensor, "batch gaussian"]
