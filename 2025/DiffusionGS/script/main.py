import torch
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from .config import load_root_config
from utils.config_util import set_config
from utils.wandb_util import update_checkpoint_path
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch import Trainer
from dataset.dataloader import DataModule
from .model.decoder import decoder

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
        wandb.run.log_code("script")

    # Callbacks - Checkpoint
    callbacks.append(
        ModelCheckpoint(
            output_dir / "checkpoints",
            every_n_train_steps=config.checkpoint.every_num_time_steps,
            save_top_k=config.checkpoint.save_top_k
        )
    )

    # Load Checkpoint Path if exists
    checkpoint_path = update_checkpoint_path(config.checkpoint.load, config.wandb)

    # May need Step Tracker

    # Trainer
    trainer = Trainer(
        max_epochs=-1,
        accelerator="gpu",
        logger=logger,
        devices="auto",
        callbacks=callbacks,
        val_check_interval=config.trainer.validation_check_interval,
        enable_progress_bar=True,
        gradient_clip_val=config.trainer.gradient_clip_validation,
        max_steps=config.trainer.max_steps)

    torch.manual_seed(config_dict.seed + trainer.global_rank)

    diffusion_gs = DiffusionGS(
        config.optimizer,
        config.test,
        config.train,
        get_denoiser(config.model.denoiser),
        get_decoder(config.model.decoder, config.dataset),
        get_losses(config.loss),
        step_tracker)

    data_module = DataModule(config.dataset, config.dataloader, None, global_rank=trainer.global_rank)

    if config.mode == "train":
        trainer.fit(diffusion_gs, datamodule=data_module, ckpt_path=checkpoint_path)
    else:
        trainer.test(diffusion_gs, datamodule=data_module, ckpt_path=checkpoint_path)

if __name__ == "__main__":
    train()