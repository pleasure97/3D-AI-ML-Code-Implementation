from dataclasses import dataclass
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from typing import Optional
from pathlib import Path
from torch import nn, Tensor
import torch
from diffusion import DiffusionGenerator
from denoiser.embedding.timestep_embedding import TimestepMLP
from denoiser.embedding.patch_embedding import PatchMLP
from denoiser.embedding.positional_embedding import PositionalEmbedding
from denoiser.backbone.transformer_backbone import TransformerBackbone
from decoder.decoder import GaussianDecoder
from src.utils.step_tracker import StepTracker
from src.loss import LossesConfig
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from lightning.pytorch.utilities import rank_zero_only
from src.dataset.types import BatchedExample
from src.model.rasterizer.render import render
from src.evaluation.metrics import get_psnr

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
    diffusion_generator: DiffusionGenerator
    timestep_mlp: TimestepMLP
    patch_mlp: PatchMLP
    positional_embedding: PositionalEmbedding
    transformer_backbone: TransformerBackbone
    gaussian_decoder: GaussianDecoder
    losses: LossesConfig
    optimizer_config: OptimizerConfig
    train_config: TrainConfig
    test_config: TestConfig
    step_tracker: StepTracker | None

    def __init__(self,
                 optimizer_config: OptimizerConfig,
                 train_config: TrainConfig,
                 test_config: TestConfig,
                 diffusion_generator: DiffusionGenerator,
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

        self.diffusion_generator = diffusion_generator
        self.timestep_mlp = timestep_mlp
        self.patch_mlp = patch_mlp
        self.positional_embedding = positional_embedding
        self.sample = None
        self.transformer_backbone = transformer_backbone
        self.gaussian_decoder = gaussian_decoder
        self.losses = nn.ModuleList(losses)

    def training_step(self, batch, batch_index):
        current_step = self.global_step
        warmup_steps = self.optimizer_config.warmup_steps

        # TODO - Preprocess the BatchedExample
        samples: list[BatchedExample] = self.sample(batch)
        background_color = Tensor([0, 0, 0])  # TODO - if not dataset.white_background else [1, 1, 1]
        _, _, _, height, width = batch["target"]["image"].shape

        # TODO - Run the model
        noisy_images = []
        rasterized_images = []
        noisy_images = self.diffusion_generator.generate(batch)
        for timestep in reversed(range(self.diffusion_generator.total_timesteps, self.diffusion_generator.num_timesteps)):
            for sample in samples:
                noisy_image = noisy_images[timestep]

                timestep_mlp_output = self.timestep_mlp(timestep)
                transformer_backbone_input = self.patch_mlp(sample["target"]) + self.positional_embedding
                transformer_backbone_output = self.transformer_backbone(transformer_backbone_input, timestep)
                positions, covariances, colors, opacities = self.gaussian_decoder(timestep_mlp_output, transformer_backbone_output)

                rasterized_image = render(sample["target"]["extrinsics"],
                               sample["target"]["intrinsics"],
                               sample["target"]["near"],
                               sample["target"]["far"],
                               sample["target"]["image"].shape,
                               background_color,
                               colors,
                               positions,
                               covariances,
                               opacities)

                # TODO - Compute the metrics
                psnr = get_psnr(batch["target"]["image"], rasterized_image)
                self.log("train/psnr", psnr)

        # TODO - Compute the loss
        total_loss = 0

        for loss in self.losses:
            # TODO - Denoising Loss
            if loss.name == "DenoisingLoss":
                denoising_loss = DenoisingLoss()
                denoising_loss_value =
                self.log(f"loss/{denoising_loss}", denoising_loss_value)

            # TODO - Novel View Loss
            else if loss.name == "NovelViewLoss":
                novel_view_loss = NovelViewLoss()
                novel_view_loss_value =
                self.log(f"loss/{novel_view_loss}", novel_view_loss_value)

            # TODO - Point Distribution Loss
            else if loss.name == "PointDistributionLoss":
                point_distribution_loss = PointDistributionLoss()
                point_distribution_loss_value =
                self.log(f"loss/{point_distribution_loss}", point_distribution_loss_value)

        total_loss = torch.where(current_step > warmup_steps,
                                 denoising_loss_value + novel_view_loss_value,
                                 point_distribution_loss_value * torch.where(is_object, 1, 0))


        if self.global_rank == 0:
            print(f"train step {self.global_step}; "
                  f"scene = {batch['scene']}; "
                  f"source = {batch['source']['index'].tolist()}"
                  f"loss = {total_loss:.6f}")


        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        return total_loss


    @rank_zero_only
    def validation_step(self, batch, batch_index):
        batch: BatchedExample = self.sample(batch)

        if self.global_rank == 0:
            print(
                f"validation step {self.global_step}; "
                f"scene = {batch['scene']}; "
                f"source = {batch['source']['index'].tolist()}")

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
