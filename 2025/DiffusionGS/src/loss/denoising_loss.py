from dataclasses import dataclass
from src.loss.base_loss import BaseLoss, VGGLoss
from src.preprocess.types import BatchedExample
from jaxtyping import Float
from torch import nn


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

    def forward(self,
                batch: BatchedExample) -> Float:
        clean_view = batch["clean"]["views"]  # [batch_size, 1, channel, height, width]
        noisy_views = batch["noisy"]["views"]  # [batch_size, num_noisy_views, channel, height, width]

        batch_size, _, channel, height, width = clean_view.shape
        _, num_noisy_views, _, _, _ = noisy_views.shape

        # clean_repeated shape : [batch_size, num_noisy_views, channel, height, width]
        clean_repeated = clean_view.repeat(1, num_noisy_views, 1, 1, 1)
        clean_flattened = clean_repeated.view(batch_size * num_noisy_views, channel, height, width)
        noisy_flattened = noisy_views.view(batch_size * num_noisy_views, channel, height, width)

        mse_loss = self.mse(clean_flattened, noisy_flattened)
        vgg_loss = self.vgg.forward(clean_flattened, noisy_flattened)

        loss = mse_loss + self.config.weight * vgg_loss

        return loss
