from dataclasses import dataclass
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from typing import Optional


@dataclass
class OptimizerConfig:
    learning_rate: float
    warmup_steps: int


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

