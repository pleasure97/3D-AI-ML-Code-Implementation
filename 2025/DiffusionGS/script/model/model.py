from dataclasses import dataclass
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from typing import Optional
from pathlib import Path
from torch import nn
from denoiser.embedding.timestep_embedding import TimestepMLP
from denoiser.embedding.positional_embedding import PatchEmbedding
from denoiser.backbone.transformer_backbone import TransformerBackbone
from decoder.decoder import GaussianDecoder
from ..loss.loss import Loss
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR


@dataclass
class OptimizerConfig:
    learning_rate: float
    total_steps: int
    warmup_steps: int


@dataclass
class TrainConfig:
    pass


@dataclass
class TestConfig:
    output_path: Path


class DiffusionGS(LightningModule):
    logger: Optional[WandbLogger]
    timestep_mlp: TimestepEmbedding,
    patchify_mlp: PatchifyEmbedding,
    transformer_backbone: TransformerBackbone
    gaussian_decoder: GaussianDecoder
    losses: nn.ModuleList
    optimizer_config: OptimizerConfig
    train_config: TrainConfig
    test_config: TestConfig
    step_tracker: None  # TODO - Whether to use step tracker

    def __init__(self,
                 optimizer_config: OptimizerConfig,
                 train_config: TrainConfig,
                 test_config: TestConfig,
                 timestep_mlp: TimestepEmbedding,
                 patchify_mlp: PatchifyEmbedding,
                 transformer_backbone: TransformerBackbone,
                 gaussian_decoder: GaussianDecoder,
                 losses: list[Loss],
                 step_tracker: None) -> None:
        super().__init__()
        self.optimizer_config = optimizer_config
        self.train_config = train_config
        self.test_config = test_config
        self.step_tracker = step_tracker

    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        # TODO - Run the model
        TimestepMLP(...)

        # TODO - Compute the metrics

        # TODO - Compute the loss

        return total_loss

    @rank_zero_only
    def validation_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        pass

    def test_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        pass

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.optimizer_config.learning_rate)
        warmup_scheduler = LambdaLR(optimizer,
                                    lr_lambda=lambda epoch, warmup_iters: epoch / warmup_iters if epoch < warmup_iters else 1)
        cosine_annealing_scheduler = CosineAnnealingLR(optimizer,
                                                       T_max=self.optimizer_config.total_steps - self.optimizer_config.warmup_steps, eta_min=0)
        scheduler = SequentialLR(optimizer,
                                 schedulers=[warmup_scheduler, cosine_annealing_scheduler],
                                 milestones=[self.optimizer_config.warmup_steps])
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1}
        }
