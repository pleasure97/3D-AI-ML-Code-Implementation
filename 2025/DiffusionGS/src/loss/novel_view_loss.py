from dataclasses import dataclass
from typing import Literal
from src.loss import Loss
from src.model.types import Gaussians
from src.dataset.types import BatchedExample
from jaxtyping import Float
from torch import Tensor


@dataclass
class NovelViewLossConfig:
    name: Literal["novel_view_loss"]
    weight: float

class NovelViewLoss(Loss[NovelViewLossConfig]):
    def forward(self, prediction: Gaussians, batch: BatchedExample) -> Float[Tensor]:
        delta = prediction.colors - batch["target"]["image"]
        return self.config.weight * (delta ** 2).mean()

