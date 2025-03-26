from dataclasses import dataclass
from typing import Literal, Optional
from .dataset.data_module import DatasetConfig
from .model.diffusion import DiffusionConfig
from .model.denoiser import DenoiserConfig
from .model.decoder import DecoderConfig

@dataclass
class ModelConfig:
    diffusion: DiffusionConfig
    denoiser: DenoiserConfig
    decoder: DecoderConfig

@dataclass
class CheckpointConfig:
    load: Optional[str] # wandb://
    every_num_time_steps: int
    save_top_k: int

@dataclass
class TrainerConfig:
    max_steps: int
    validation_check_interval: int | float | None
    gradient_check_validation: int | float | None

@dataclass
class RootConfig:
    wandb: dict
    mode: Literal["train", "test"]
    dataset: DatasetConfig
    model:
