from dataclasses import dataclass
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from typing import Optional
from pathlib import Path
import torch
from torch import nn, Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from src.utils.config_util import get_config
from src.utils.step_tracker import StepTracker
from src.utils.benchmarker import Benchmarker
from src.model.diffusion import DiffusionGenerator
from src.model.types import Gaussians
from src.model.rasterizer.render import render
from src.model.denoiser.embedding.timestep_embedding import TimestepMLP
from src.model.denoiser.embedding.patch_embedding import PatchMLP
from src.model.denoiser.embedding.positional_embedding import PositionalEmbedding
from src.model.denoiser.backbone.transformer_backbone import TransformerBackbone
from src.model.decoder.decoder import GaussianDecoder
from src.evaluation.metrics import get_psnr
from src.loss import LossesConfig
from src.loss.base_loss import BaseLoss
from src.loss.denoising_loss import DenoisingLoss
from src.loss.novel_view_loss import NovelViewLoss
from src.loss.point_distribution_loss import PointDistributionLoss


@dataclass
class OptimizerConfig:
    learning_rate: float
    total_steps: int
    warmup_steps: int


@dataclass
class TrainConfig:
    is_object_dataset: bool


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
    object_decoder: GaussianDecoder
    scene_decoder: GaussianDecoder
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
                 object_decoder: GaussianDecoder,
                 scene_decoder: GaussianDecoder,
                 losses: list[BaseLoss],
                 step_tracker: StepTracker) -> None:
        super().__init__()
        self.optimizer_config = optimizer_config
        self.train_config = train_config
        self.test_config = test_config
        self.step_tracker = step_tracker
        self.benchmarker = Benchmarker()

        self.diffusion_generator = diffusion_generator
        self.timestep_mlp = timestep_mlp
        self.patch_mlp = patch_mlp
        self.positional_embedding = positional_embedding

        self.transformer_backbone = transformer_backbone
        self.object_decoder = object_decoder
        self.scene_decoder = scene_decoder
        self.losses = nn.ModuleList(losses)

    def training_step(self, batch, batch_index):
        current_step = self.global_step
        # TODO - Preprocess the BatchedExample
        background_color = Tensor([0, 0, 0])  # TODO - if not preprocess.white_background else [1, 1, 1]
        _, _, _, height, width = batch["target"]["image"].shape

        # TODO - Run the model
        for timestep in reversed(
                range(self.diffusion_generator.total_timesteps, self.diffusion_generator.num_timesteps)):
            for sample in batch:
                noisy_image = self.diffusion_generator.generate(batch["source"]["image"])

                RPPC = batch["target"]["rppc"]
                timestep_mlp_output = self.timestep_mlp(timestep)
                transformer_backbone_input = self.patch_mlp(sample["target"]) + self.positional_embedding
                transformer_backbone_output = self.transformer_backbone(transformer_backbone_input, timestep, RPPC)
                positions, covariances, colors, opacities = self.gaussian_decoder(timestep_mlp_output,
                                                                                  transformer_backbone_output)

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

            del noisy_image, rasterized_image
            torch.cuda.empty_cache()

        # TODO - Compute the loss
        loss_dict = {}

        for loss in self.losses:
            if loss.name == "DenoisingLoss":
                loss_value = loss.forward(batch)

            # TODO - Novel View Loss
            # def forward(self, prediction: Gaussians, batch: BatchedExample) -> Float[Tensor]:
            elif loss.name == "NovelViewLoss":
                loss_value = loss.forward(Gaussians(positions, covariances, colors, opacities), batch)

            # TODO - Point Distribution Loss
            #   def forward(self,
            #                 weight_u: float, u_near: float, u_far: float,
            #                 rays_o: Float[Tensor], rays_d: Float[Tensor], timesteps: int)
            elif loss.name == "PointDistributionLoss":
                loss_value = loss.forward(weight_u, u_near, u_far, rays_o, ray_d, timestep)

            loss_dict[loss.name] = loss_value
            self.log(f"loss/{loss.name}", loss_value)

        total_loss = torch.where(current_step > self.optimizer_config.warmup_steps,
                                 loss_dict["DenoisingLoss"] + loss_dict["NovelViewLoss"],
                                 loss_dict["PointDistributionLoss"] * torch.where(self.training_config.is_object_dataset, 1, 0))

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
        if self.global_rank == 0:
            print("batch keys:", batch.keys())
            print("source type:", type(batch['source']))
            if isinstance(batch['source'], dict):
                print("source keys:", batch['source'].keys())
            else:
                print("source tensor shape:", batch['source'].shape)

    def test_step(self, batch, batch_index):
        batch_size, _, _, height, width = batch["target"]["image"].shape
        assert batch_size == 1
        if batch_index % 100 == 0:
            print(f"Test Step {batch_index}")

    def on_test_end(self) -> None:
        name = get_config()["wandb"]["name"]
        self.benchmarker.dump(self.test_config.output_path / name / "benchmark.json")
        self.benchmarker.dump_memory(self.test_config.output_path / name / "peak_memory.json")

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.optimizer_config.learning_rate)
        warmup_scheduler = LinearLR(optimizer,
                                    1 / self.optimizer_config.warmup_steps,
                                    1,
                                    self.optimizer_config.warmup_steps)
        cosine_annealing_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.optimizer_config.total_steps - self.optimizer_config.warmup_steps,
            eta_min=0)
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
