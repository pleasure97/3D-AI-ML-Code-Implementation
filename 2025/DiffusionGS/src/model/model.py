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
from src.evaluation.metrics import get_psnr, get_ssim, get_fid, LPIPS
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
        self.lpips = LPIPS(device="cuda" if torch.cuda.is_available() else "cpu")
        self.freeze_loss_parameters()
        self.fid = None

    def freeze_loss_parameters(self):
        for loss in self.losses:
            for loss_parameter in loss.parameters():
                loss_parameter.requires_grad = False
        for parameter in self.lpips.parameters():
            parameter.requires_grad = False

    def training_step(self, batch, batch_index):
        current_step = self.global_step

        # Initialize Loss dict and Loss Values
        loss_dict = {}
        total_denoising_loss = 0.
        total_novel_view_loss = 0.
        total_point_distribution_loss = 0.

        # Do Lazy Initialization for Positional Embedding
        if self.positional_embedding is None:
            num_clean_views = batch["clean"]["views"].shape[1]  # [batch_size, num_clean_views, 3, height, width]
            batch_size, num_noisy_views, _, height, width = batch["noisy"]["views"].shape
            patch_size = self.patch_mlp.config.embedding.patch_size
            embedding_dim = self.patch_mlp.config.embedding.embedding_dim
            patches_per_view = (height // patch_size) * (width // patch_size)
            num_views = num_clean_views + num_noisy_views
            num_patches = num_views * patches_per_view
            self.positional_embedding = PositionalEmbedding(num_patches=num_patches, embedding_dim=embedding_dim)

        # Iterate Timesteps
        for timestep in reversed(
                range(self.diffusion_generator.num_timesteps, self.diffusion_generator.total_timesteps)):
            point_distribution_loss = 0.
            denoising_loss = 0.
            novel_view_loss = 0.

            # Tokenize inputs to enter Patchify MLP
            noisy_views = batch["noisy"]["views"].unbind(dim=1)  # [batch_size, num_noisy_views, 3, height, width]
            transformer_input_tokens = self.tokenize_inputs(timestep,
                                                            batch["clean"]["views"],
                                                            noisy_views,
                                                            self.patch_mlp)

            # Process timestep embedding and RPPC to skip connection
            clean_RPPC = batch["clean"]["RPPCs"]  # [batch_size, 1, 6, height, width]
            noisy_RPPCs = batch["noisy"]["RPPCs"]  # [batch_size, num_noisy_views, 6, height, width]
            RPPC = torch.cat([clean_RPPC, noisy_RPPCs], dim=1)  # [batch_size, num_noisy_views + 1, 6, height, width]

            transformer_backbone_output = self.transformer_backbone.forward(transformer_input_tokens, timestep, RPPC)

            # Separately Process timestep embedding which goes into decoders
            timestep_mlp_output = self.timestep_mlp(timestep)

            # Call forward propagation for each object decoder and scene decoder to train them
            # In finetuning phase, only one decoder will be used for each dataset type
            if self.train_config.is_object_dataset:
                positions, covariances, colors, opacities = self.object_decoder.forward(transformer_backbone_output,
                                                                                        timestep_mlp_output)
            else:
                positions, covariances, colors, opacities = self.scene_decoder.forward(transformer_backbone_output,
                                                                                       timestep_mlp_output)

            # Initialize variables that are repeatedly used in operations
            noisy_extrinsics = batch["noisy"]["extrinsics"]
            noisy_intrinsics = batch["noisy"]["intrinsics"]
            noisy_image_shape = batch["noisy"]["views"].shape
            num_noisy_views = noisy_extrinsics.shape[1]

            # Compute Point Distribution Loss
            PointDistributionLoss = next(loss for loss in self.losses if loss.name == "PointDistributionLoss")

            height, width = noisy_image_shape[-1], noisy_image_shape[-2]
            c2w = make_c2w_from_extrinsics(noisy_extrinsics)
            rays_o, rays_d = get_rays(height, width, noisy_intrinsics, c2w)

            # Point Distribution Loss is introduced to be object-centric generation more concentrated,
            # so we use u_near and u_far of object decoder
            point_distribution_loss_value = PointDistributionLoss.forward(
                self.object_decoder.u_near,
                self.object_decoder.u_far,
                rays_o,
                rays_d)
            point_distribution_loss += point_distribution_loss_value

            # Rasterize 3D Gaussians
            rasterized_images = self.gaussian_renderer.render(
                noisy_extrinsics,
                noisy_intrinsics,
                noisy_image_shape,
                colors,
                positions,
                covariances,
                opacities,
                background_white=True)

            # Iterate N Noisy Views
            for i in range(num_noisy_views):
                # Compute Denoising Loss between Multi-view Predicted Images and Ground-truth Image
                DenoisingLoss = next(loss for loss in self.losses if loss.name == "DenoisingLoss")
                # TODO - self.diffusion_generator.q_sample(batch["noisy"]["views"][:, i], timestep)
                denoising_loss_value = DenoisingLoss.forward(batch["noisy"]["views"][:, i], rasterized_images[:, i])
                denoising_loss += denoising_loss_value

                # Compute Novel View Loss between Multi-view Predicted Images and M Novel Views
                if timestep == 0:
                    NovelViewLoss = next(loss for loss in self.losses if loss.name == "NovelViewLoss")
                    num_novel_views = batch["novel"]["views"].shape[1]
                    for j in range(num_novel_views):
                        novel_view = batch["novel"]["views"][:, j]
                        novel_view_loss_value = NovelViewLoss.forward(novel_view, rasterized_images[:, i])
                        novel_view_loss += novel_view_loss_value
                    total_novel_view_loss /= num_novel_views

            denoising_loss /= num_noisy_views

            total_point_distribution_loss += point_distribution_loss
            total_denoising_loss += denoising_loss

            del rasterized_images
            torch.cuda.empty_cache()

        loss_dict["PointDistributionLoss"] = total_point_distribution_loss / self.diffusion_generator.total_timesteps
        self.log("loss/PointDistributionLoss", total_point_distribution_loss / self.diffusion_generator.total_timesteps)

        loss_dict["DenoisingLoss"] = total_denoising_loss / self.diffusion_generator.total_timesteps
        self.log("loss/DenoisingLoss", total_denoising_loss / self.diffusion_generator.total_timesteps)

        loss_dict["NovelViewLoss"] = total_novel_view_loss
        self.log("loss/NovelViewLoss", total_novel_view_loss)

        total_loss = \
            torch.where(current_step > self.optimizer_config.warmup_steps,
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

    def tokenize_inputs(self,
                        timestep: int,
                        clean_image: torch.Tensor,
                        noisy_images: list[torch.Tensor],
                        patch_mlp: PatchMLP) -> torch.Tensor:
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

        source_image, target_image = batch["clean"]["views"], batch["noisy"]["views"]
        prediction_image = self(source_image)

        psnr_value = get_psnr(target_image, prediction_image)
        self.log('val/psnr', psnr_value, on_step=False, on_epoch=True)

        ssim_value = get_ssim(target_image, prediction_image)
        self.log('val/ssim', ssim_value, on_step=False, on_epoch=True)

        lpips_value = self.lpips.forward(target_image, prediction_image)
        self.log('val/lpips', lpips_value, on_step=False, on_epoch=True)

        if self.fid is None:
            self.fid = get_fid(target_image, prediction_image)

    def on_test_end(self) -> None:
        name = get_config()["wandb"]["name"]
        self.benchmarker.dump(self.test_config.output_path / name / "benchmark.json")
        self.benchmarker.dump_memory(self.test_config.output_path / name / "peak_memory.json")

        fid_value = self.fid.compute()
        self.log('val/fid', fid_value, on_epoch=True)
        self.fid.reset()

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
