from dataclasses import dataclass
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from typing import Optional
from pathlib import Path


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
    encoder:
    decoder:
    losses:
    optimizer_config: OptimizerConfig
    train_config: TrainConfig
    test_config: TestConfig
    step_tracker: None

    def __init__(self):

