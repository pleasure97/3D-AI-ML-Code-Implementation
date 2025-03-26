from typing import Literal, TypedDict
from jaxtyping import Float, Int64
from torch import Tensor

Stage = Literal["train", "val", "test"]

class BatchedViews(TypedDict):
    extrinsics: Float[Tensor, "batch _ 4 4"]
    intrinsics: Float[Tensor, "batch _ 3 3"]
    image: Float[Tensor, "batch _ _ _ _"]
    near: Float[Tensor, "batch _"]
    index: Int64[Tensor, "batch _"]