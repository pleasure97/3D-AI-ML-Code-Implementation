from dataclasses import dataclass
from .loss import Loss
from ..model.types import RasterizedOutput
from ..dataset.types import BatchedExample
from jaxtyping import Float
from torch import Tensor


@dataclass
class NovelViewLossConfig:
    pass

@dataclass
class NovelViewLossConfigWrapper:
    A: NovelViewLossConfig


class NovelViewLoss(Loss[NovelViewLossConfig, NovelViewLossConfigWrapper]):
    def forward(self, prediction: RasterizedOutput, batch: BatchedExample) -> Float[Tensor]:
        pass

