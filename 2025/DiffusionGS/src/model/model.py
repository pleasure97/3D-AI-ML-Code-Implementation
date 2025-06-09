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
from src.utils.geometry_util import make_c2w_from_extrinsics
from src.model.diffusion import DiffusionGenerator
from src.model.types import Gaussians
from src.model.rasterizer.render import GaussianRenderer
from src.model.denoiser.embedding.timestep_embedding import TimestepMLP
from src.model.denoiser.embedding.patch_embedding import PatchMLP
from src.model.denoiser.embedding.positional_embedding import PositionalEmbedding
from src.model.denoiser.backbone.transformer_backbone import TransformerBackbone
from src.model.decoder.decoder import GaussianDecoder
from src.model.denoiser.viewpoint.RPPC import get_rays
from src.evaluation.metrics import get_psnr
from src.loss import LossesConfig
from src.loss.base_loss import BaseLoss


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
    transformer_backbone: TransformerBackbone
    object_decoder: GaussianDecoder
    scene_decoder: GaussianDecoder
    gaussian_renderer: GaussianRenderer
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
                 transformer_backbone: TransformerBackbone,
                 object_decoder: GaussianDecoder,
                 scene_decoder: GaussianDecoder,
                 gaussian_renderer: GaussianRenderer,
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
        self.positional_embedding = None

        self.transformer_backbone = transformer_backbone
        self.object_decoder = object_decoder
        self.scene_decoder = scene_decoder
        self.gaussian_renderer = gaussian_renderer
        self.losses = nn.ModuleList(losses)

    def training_step(self, batch, batch_index):
        current_step = self.global_step

        background_color = Tensor([0, 0, 0])  # TODO - if not preprocess.white_background else [1, 1, 1]

        loss_dict = {}

        if self.positional_embedding is None:
            num_clean_views = batch["clean"]["views"].shape[1]
            batch_size, num_noisy_views, _, height, width = batch["noisy"]["views"].shape
            patch_size = self.patch_mlp.config.embedding.patch_size
            embedding_dim = self.patch_mlp.config.embedding.embedding_dim
            patches_per_view = (height // patch_size) * (width // patch_size)
            num_views = num_clean_views + num_noisy_views
            num_patches = num_views * patches_per_view
            self.positional_embedding = PositionalEmbedding(num_patches=num_patches, embedding_dim=embedding_dim)

        for timestep in reversed(
                range(self.diffusion_generator.num_timesteps, self.diffusion_generator.total_timesteps)):
            noisy_views = batch["noisy"]["views"].unbind(dim=1)
            transformer_input_tokens = self.tokenize_inputs(timestep,
                                                            batch["clean"]["views"],
                                                            noisy_views,
                                                            self.patch_mlp,
                                                            self.positional_embedding)

            RPPC = batch["noisy"]["RPPCs"]
            timestep_mlp_output = self.timestep_mlp(timestep)

            transformer_backbone_output = self.transformer_backbone(transformer_input_tokens, timestep, RPPC)
            positions, covariances, colors, opacities = self.gaussian_decoder(timestep_mlp_output,
                                                                              transformer_backbone_output)

            extrinsics = batch["noisy"]["extrinsics"]
            intrinsics = batch["noisy"]["intrinsics"]
            image_shape = batch["noisy"]["views"].shape
            rasterized_image = self.gaussian_renderer.render(
                extrinsics,
                intrinsics,
                batch["noisy"]["nears"],
                batch["noisy"]["fars"],
                image_shape,
                background_color,
                colors,
                positions,
                covariances,
                opacities)

            for loss in self.losses:
                if loss.name == "DenoisingLoss":
                    loss_value = loss.forward(batch)
                elif loss.name == "NovelViewLoss":
                    loss_value = loss.forward(Gaussians(positions, covariances, colors, opacities), batch)
                elif loss.name == "PointDistributionLoss":
                    height, width = image_shape
                    c2w = make_c2w_from_extrinsics(extrinsics)
                    rays_o, rays_d = get_rays(height, width, intrinsics, c2w)
                    loss_value = loss.forward(self.gaussian_decoder.u_near,
                                              self.gaussian_decoder.u_far,
                                              rays_o,
                                              rays_d,
                                              timestep)
                loss_dict[loss.name] = loss_value
                self.log(f"loss/{loss.name}", loss_value)

            # TODO - Compute the metrics
            psnr = get_psnr(batch["target"]["views"], rasterized_image)
            self.log("train/psnr", psnr)

            del rasterized_image  # noisy_image
            torch.cuda.empty_cache()

        total_loss = torch.where(current_step > self.optimizer_config.warmup_steps,
                                 loss_dict["DenoisingLoss"] + loss_dict["NovelViewLoss"],
                                 loss_dict["PointDistributionLoss"] * torch.where(
                                     self.training_config.is_object_dataset, 1, 0))

        if self.global_rank == 0:
            print(f"train step {self.global_step}; "
                  f"scene = {batch['scene']}; "
                  f"source = {batch['source']['index'].tolist()}"
                  f"loss = {total_loss:.6f}")

        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        return total_loss

    def tokenize_inputs(self,
                        timestep: int,
                        clean_image: torch.Tensor,
                        noisy_images: list[torch.Tensor],
                        patch_mlp: PatchMLP,
                        positional_embedding: PositionalEmbedding) -> torch.Tensor:
        if clean_image.dim() == 5:
            clean_image = clean_image.squeeze(1)
        clean_token = patch_mlp(clean_image)
        multiview_tokens = [clean_token]
        for selected_image in noisy_images:
            if selected_image.dim() == 5:
                selected_image = selected_image.squeeze(1)
            noised_image = self.diffusion_generator.generate(selected_image, timestep)
            noisy_token = patch_mlp(noised_image)
            multiview_tokens.append(noisy_token)
        tokens = torch.cat(multiview_tokens, dim=1)
        tokens = self.positional_embedding(tokens)

        return tokens

    @rank_zero_only
    def validation_step(self, batch, batch_index):
        if self.global_rank == 0:
            if isinstance(batch["clean"], dict):
                print("clean keys:", batch["clean"].keys())
            else:
                print("clean tensor shape:", batch["clean"].shape)

    def test_step(self, batch, batch_index):
        batch_size, _, _, height, width = batch["noisy"]["views"].shape
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
