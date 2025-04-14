from dataclasses import dataclass
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from typing import Optional
from pathlib import Path
from torch import nn
from denoiser.embedding.timestep_embedding import TimestepMLP
from denoiser.embedding.patch_embedding import PatchMLP
from denoiser.embedding.positional_embedding import PositionalEmbedding
from denoiser.backbone.transformer_backbone import TransformerBackbone
from decoder.decoder import GaussianDecoder
from src.utils.step_tracker import StepTracker
from ..loss.loss import Loss
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from fast_gauss import GaussianRasterizationSettings, GaussianRasterizer
from lightning.pytorch.utilities import rank_zero_only
from src.dataset.types import BatchedExample

@dataclass
class OptimizerConfig:
    learning_rate: float
    total_steps: int
    warmup_steps: int


@dataclass
class TrainConfig:
    timesteps: int


@dataclass
class TestConfig:
    output_path: Path


class DiffusionGS(LightningModule):
    logger: Optional[WandbLogger]
    timestep_mlp: TimestepMLP
    patch_mlp: PatchMLP
    positional_embedding: PositionalEmbedding
    transformer_backbone: TransformerBackbone
    gaussian_decoder: GaussianDecoder
    losses: nn.ModuleList
    optimizer_config: OptimizerConfig
    train_config: TrainConfig
    test_config: TestConfig
    step_tracker: StepTracker | None

    def __init__(self,
                 optimizer_config: OptimizerConfig,
                 train_config: TrainConfig,
                 test_config: TestConfig,
                 timestep_mlp: TimestepMLP,
                 patch_mlp: PatchMLP,
                 positional_embedding: PositionalEmbedding,
                 transformer_backbone: TransformerBackbone,
                 gaussian_decoder: GaussianDecoder,
                 losses: list[Loss],
                 step_tracker: StepTracker) -> None:
        super().__init__()
        self.optimizer_config = optimizer_config
        self.train_config = train_config
        self.test_config = test_config
        self.step_tracker = step_tracker

        self.timestep_mlp = timestep_mlp
        self.patch_mlp = patch_mlp
        self.positional_embedding = positional_embedding
        self.transformer_backbone = transformer_backbone
        self.gaussian_decoder = gaussian_decoder
        self.losses = nn.ModuleList(losses)

    def training_step(self, batch, batch_index):
        batch: BatchedExample = self.(batch)
        _, _, _, height, width = batch["target"]["image"].shape

        # TODO - Run the model
        for timestep in (self.train_config.timesteps):
            transformer_backbone_input = self.patch_mlp(batch["source"]) + self.positional_embedding
            self.transformer_backbone(transformer_backbone_input, timestep)

            GaussianRasterizer()

            # TODO - Compute the metrics

            # TODO - Compute the loss

            # TODO - Tell the dataloader process about the current step.

        return total_loss


    @rank_zero_only
    def validation_step(self, batch, batch_index):
        pass

    def test_step(self, batch, batch_index):
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
