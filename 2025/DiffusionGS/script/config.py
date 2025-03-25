from dataclasses import dataclass
from typing import Literal
from .dataset.dataset import DatasetConfig
from .model.encoder import EncoderConfig
from .model.decoder import DecoderConfig

@dataclass
class ModelConfig:
    encoder: EncoderConfig
    decoder: DecoderConfig

@dataclass
class RootConfig:
    wandb: dict
    mode: Literal["train", "test"]
    dataset: DatasetConfig
    model:
