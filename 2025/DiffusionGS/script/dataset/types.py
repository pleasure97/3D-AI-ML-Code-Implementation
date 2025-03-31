from typing import Literal, TypedDict
from jaxtyping import Float, Int64
from torch import Tensor

Stage = Literal["train", "val", "test"]


class BatchedViews(TypedDict):
    extrinsics: Float[Tensor]  # [Batch Size * View * 4 * 4]
    intrinsics: Float[Tensor]  # [Batch Size * View * 3 * 3]
    image: Float[Tensor]       # [Batch Size * View * Channel * Height * Width]
    near: Float[Tensor]        # [Batch Size * View]
    far: Float[Tensor]         # [Batch Size * View]
    index: Int64[Tensor]       # [Batch Size * View]


class BatchedExample(TypedDict):
    target: BatchedViews
    source: BatchedViews
    scene: list[str]