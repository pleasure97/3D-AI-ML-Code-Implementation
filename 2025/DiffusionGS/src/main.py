import torch
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from src.config import load_root_config
from utils.config_util import set_config
from utils.wandb_util import update_checkpoint_path
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from src.utils.step_tracker import StepTracker
from lightning.pytorch import Trainer
from loss.denoising_loss import DenoisingLoss
from loss.novel_view_loss import NovelViewLoss
from loss.point_distribution_loss import PointDistributionLoss
from model.model import DiffusionGS
from model.diffusion import DiffusionGenerator
from model.denoiser.embedding.timestep_embedding import TimestepMLP
from model.denoiser.embedding.patch_embedding import PatchMLP
from model.denoiser.embedding.positional_embedding import PositionalEmbedding
from model.denoiser.backbone.transformer_backbone import TransformerBackbone
from model.decoder.decoder import GaussianDecoder
from model.rasterizer.render import GaussianRenderer
from preprocess.dataloader import DataModule


@hydra.main(version_base=None, config_path="../config", config_name="main")
def train(config_dict: DictConfig):
    config = load_root_config(config_dict)
    set_config(config)

    # Output Directory
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"])
    print(f"Saving outputs to {output_dir}")

    # Callbacks
    callbacks = []

    # Logger
    logger = WandbLogger(
        project=config_dict.wandb.project,
        mode=config_dict.wandb.mode,
        name=f"{config_dict.wandb.name} ({output_dir.parent.name}/{output_dir.name})",
        tags=config_dict.wandb.get("tags", None),
        log_model="all",
        save_dir=output_dir,
        config=OmegaConf.to_container(config_dict)
    )

    # Callbacks - Learning Rate Monitor
    callbacks.append(
        LearningRateMonitor(logging_interval="step", log_momentum=True, log_weight_decay=True)
    )

    if wandb.run is not None:
        wandb.run.log_code("src")

    # Callbacks - Checkpoint
    callbacks.append(
        ModelCheckpoint(
            output_dir / "checkpoints",
            every_n_train_steps=config.checkpoint.every_num_timesteps,
            save_top_k=config.checkpoint.save_top_k
        )
    )

    # Load Checkpoint Path if exists
    checkpoint_path = update_checkpoint_path(config.checkpoint.load, config.wandb)

    # Step Tracker
    step_tracker = StepTracker()

    # Trainer
    trainer = Trainer(
        max_epochs=-1,
        accelerator="gpu",
        logger=logger,
        devices=1,
        callbacks=callbacks,
        precision=config.trainer.precision,
        val_check_interval=config.trainer.validation_check_interval,
        enable_progress_bar=True,
        gradient_clip_val=config.trainer.gradient_clip_validation,
        max_steps=config.trainer.max_steps)

    torch.manual_seed(config_dict.seed + trainer.global_rank)

    losses = [
        DenoisingLoss(config.losses.denoising),
        NovelViewLoss(config.losses.novel_view),
        PointDistributionLoss(config.losses.point_distribution),
    ]

    diffusion_gs = DiffusionGS(
        config.optimizer,
        config.train,
        config.test,
        DiffusionGenerator(config.model.diffusion),
        TimestepMLP(config.model.timestep),
        PatchMLP(config.model.patchify),
        PositionalEmbedding(config.model.positional),
        TransformerBackbone(config.model.backbone),
        GaussianDecoder(config.model.object_decoder),
        GaussianDecoder(config.model.scene_decoder),
        GaussianRenderer(config.model.render),
        losses,
        step_tracker)

    data_module = DataModule(config.dataset, config.dataloader, step_tracker, global_rank=trainer.global_rank)

    if config.mode == "train":
        trainer.fit(diffusion_gs, datamodule=data_module, ckpt_path=checkpoint_path)
    else:
        trainer.test(diffusion_gs, datamodule=data_module, ckpt_path=checkpoint_path)


if __name__ == "__main__":
    train()
