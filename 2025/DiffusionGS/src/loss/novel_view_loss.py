from dataclasses import dataclass
from src.loss.base_loss import BaseLoss, VGGLoss
from jaxtyping import Float
from torch import nn, Tensor


@dataclass
class NovelViewLossConfig:
    name: str
    weight: float


class NovelViewLoss(BaseLoss[NovelViewLossConfig]):
    def __init__(self, config: NovelViewLossConfig) -> None:
        super().__init__(config)

        self.mse = nn.MSELoss()
        self.vgg = VGGLoss()

    def forward(self,
                ground_truth_image: Tensor,
                predicted_image: Tensor) -> Float:
        mse_loss = self.mse(ground_truth_image, predicted_image)
        vgg_loss = self.vgg.forward(ground_truth_image, predicted_image)

        loss = mse_loss + self.config.weight * vgg_loss

        return loss


