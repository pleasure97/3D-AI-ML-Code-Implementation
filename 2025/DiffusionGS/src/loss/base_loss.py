from typing import TypeVar, Generic
import torch.nn as nn

T = TypeVar("T")
class BaseLoss(nn.Module, Generic[T]):
    config: T

    def __init__(self, config) -> None:
        super().__init__()
        self.config: T = config
        self.name = self.config.name

