from typing import TypeVar, Generic
import torch.nn as nn

T = TypeVar("T")


class ModuleWithConfig(nn.Module, Generic[T]):
    def __init__(self, config: T):
        super().__init__()
        self.config: T = config
