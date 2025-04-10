from dataclasses import dataclass
from typing import Literal, Optional, TypeVar, Type
from .dataset.dataloader import DatasetConfig, DataLoaderConfig
from .model.diffusion import DiffusionConfig
from .model.denoiser import DenoiserConfig
from .model.decoder import DecoderConfig
from .model.model import OptimizerConfig, TrainConfig, TestConfig
from .loss import LossConfigWrapper
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from dacite import from_dict, Config


@dataclass
class ModelConfig:
    diffusion: DiffusionConfig
    denoiser: DenoiserConfig
    decoder: DecoderConfig


@dataclass
class CheckpointConfig:
    load: Optional[str]  # wandb://
    every_num_time_steps: int
    save_top_k: int


@dataclass
class TrainerConfig:
    max_steps: int
    validation_check_interval: int | float | None
    gradient_clip_validation: int | float | None


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
    loss: list[str]
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


def separate_loss_config_wrappers(joined: dict) -> list[LossConfigWrapper]:
    @dataclass
    class Dummy:
        dummy: LossConfigWrapper

    return [
        load_config(DictConfig({"dummy": {k: v}}), Dummy).dummy
        for k, v in joined.items()
    ]


def load_root_config(config: DictConfig) -> RootConfig:
    return load_config(config, RootConfig, {list[LossConfigWrapper]: separate_loss_config_wrappers})
