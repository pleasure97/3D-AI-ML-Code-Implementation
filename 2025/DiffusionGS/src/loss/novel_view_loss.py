from dataclasses import dataclass
from src.loss.base_loss import BaseLoss
from src.model.types import Gaussians
from src.dataset.types import BatchedExample
from jaxtyping import Float


@dataclass
class NovelViewLossConfig:
    weight: float

class NovelViewLoss(BaseLoss[NovelViewLossConfig]):
    def forward(self, prediction: Gaussians, batch: BatchedExample) -> Float:
        delta = prediction.colors - batch["target"]["image"]
        return self.config.weight * (delta ** 2).mean()

