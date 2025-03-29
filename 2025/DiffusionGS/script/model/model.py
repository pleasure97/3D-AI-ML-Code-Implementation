from dataclasses import dataclass
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from typing import Optional
from pathlib import Path
from torch import nn
from denoiser.backbone.transformer_backbone import TransformerBackbone
from decoder.decoder import GaussianDecoder
from denoiser.embedding.positional_embedding import Path


@dataclass
class OptimizerConfig:
    learning_rate: float
    warmup_steps: int

@dataclass
class TrainConfig:
    pass

@dataclass
class TestConfig:
    output_path: Path



class DiffusionGS(LightningModule):
    logger: Optional[WandbLogger]
    timestep_mlp: nn.Module
    patchify_mlp: nn.Module
    transformer_backbone: TransformerBackbone
    gaussian_decoder: GaussianDecoder
    losses: nn.ModuleList
    optimizer_config: OptimizerConfig
    train_config: TrainConfig
    test_config: TestConfig
    step_tracker: None

    def __init__(self,
                 timestep_mlp: TimestepMLP,
                 patchify_mlp: PatchifyMLP,
                 ):

