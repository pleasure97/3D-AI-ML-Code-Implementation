from dataclasses import dataclass
from src.loss.base_loss import BaseLoss, VGGLoss
from jaxtyping import Float
from torch import nn, Tensor


@dataclass
class DenoisingLossConfig:
    name: str
    weight: float


# Denoised Multi-View Images and N views when timestep is 0
class DenoisingLoss(BaseLoss[DenoisingLossConfig]):
    def __init__(self, config: DenoisingLossConfig) -> None:
        super().__init__(config)

        self.vgg = VGGLoss()
        self.mse = nn.MSELoss()
        for vgg_loss_parameter in self.vgg.parameters():
            vgg_loss_parameter.requires_grad = False

    def forward(self,
                ground_truth_image: Tensor,
                predicted_image: Tensor) -> Float:
        mse_loss = self.mse(ground_truth_image, predicted_image)
        vgg_loss = self.vgg.forward(ground_truth_image, predicted_image)

        loss = mse_loss + self.config.weight * vgg_loss

        return loss
