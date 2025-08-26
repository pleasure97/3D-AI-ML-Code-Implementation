from dataclasses import dataclass
from typing import Literal, Optional, TypeVar, Type
from src.preprocess.dataloader import DatasetConfig, DataLoaderConfig
from src.model.diffusion import DiffusionGeneratorConfig
from src.model.denoiser.embedding.timestep_embedding import TimestepMLPConfig
from src.model.denoiser.embedding.patch_embedding import PatchMLPConfig
from src.model.denoiser.backbone.transformer_backbone import BackboneConfig
from src.model.decoder.decoder import GaussianDecoderConfig
from src.model.rasterizer.render import RenderConfig
from src.loss import LossesConfig
from src.model.model import OptimizerConfig, TrainConfig, TestConfig
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from dacite import from_dict, Config


@dataclass
class ModelConfig:
    diffusion: DiffusionGeneratorConfig
    timestep: TimestepMLPConfig
    patchify: PatchMLPConfig
    backbone: BackboneConfig
    object_decoder: GaussianDecoderConfig
    scene_decoder: GaussianDecoderConfig
    render: RenderConfig
@dataclass
class CheckpointConfig:
    load: Optional[str]  # wandb://
    every_num_timesteps: int
    save_top_k: int


@dataclass
class TrainerConfig:
    max_epochs: int
    max_steps: int
    precision: int
    validation_check_interval: int | float | None
    gradient_clip_validation: int | float | None
    accumulate_grad_batches: int


@dataclass
class RootConfig:
    wandb: dict
    mode: Literal["train", "test"]
    dataset: DatasetConfig
    dataloader: DataLoaderConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    checkpoint: CheckpointConfig
    trainer: TrainerConfig
    losses: LossesConfig
    train: TrainConfig
    test: TestConfig
    seed: int


TYPE_HOOKS = {Path: Path}

# template variable
T = TypeVar("T")


def load_config(
        config: DictConfig,
        data_class: Type[T],
        extra_type_hooks: dict = {}) -> T:
    """ Create a dataclass instance from the DictConfig of OmegaConf
    """
    return from_dict(
        data_class,
        OmegaConf.to_container(config),  # Convert to a Primitive Container
        # 'type_hooks' is a Lambda expression to convert key and value to pathlib.Path object type
        config=Config(type_hooks={**TYPE_HOOKS, **extra_type_hooks}))


def load_root_config(config: DictConfig) -> RootConfig:
    return load_config(config, RootConfig)
