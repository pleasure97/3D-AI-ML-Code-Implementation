from typing import Literal, TypedDict
from jaxtyping import Float, Int64
from torch import Tensor

Stage = Literal["train", "val", "test"]


class BatchedViews(TypedDict):
    extrinsics: Float[Tensor, "batch _ 4 4"]  # [Batch Size * View * 4 * 4]
    intrinsics: Float[Tensor, "batch _ 3 3"]  # [Batch Size * View * 3 * 3]
    image: Float[Tensor, "batch _ _ _ _"]       # [Batch Size * View * Channel * Height * Width]
    near: Float[Tensor, "batch _"]        # [Batch Size * View]
    far: Float[Tensor, "batch _"]         # [Batch Size * View]
    index: Int64[Tensor, "batch _"]       # [Batch Size * View]


class BatchedExample(TypedDict):
    target: BatchedViews
    source: BatchedViews
    scene: list[str]