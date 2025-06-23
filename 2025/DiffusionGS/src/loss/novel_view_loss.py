from dataclasses import dataclass
from src.loss.base_loss import BaseLoss, VGGLoss
from src.preprocess.types import BatchedExample
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

    def forward(self, batch: BatchedExample, rasterized_image: Tensor) -> Float:

        # TODO - Unlike denoising loss, Novel View Loss deals with M novel views.
        novel_views = batch["novel"]["views"]   # [batch_size, num_novel_views, channel, height, width]

        batch_size, num_novel_views, channel, height, width = novel_views.shape

        source_repeated = source.repeat(1, num_novel_views, 1, 1, 1)  # [batch_size, num_views, channel, height, width]
        source_flattened = source_repeated.view(batch_size * num_novel_views, channel, height, width)
        target_flattened = target.view(batch_size * num_novel_views, channel, height, width)

        mse_loss = self.mse(source_flattened, target_flattened)
        vgg_loss = self.vgg.forward(source_flattened, target_flattened)

        loss = mse_loss + self.config.weight * vgg_loss

        return loss

