from torch import nn, Tensor
from typing import TypeVar, Generic
from dataclasses import fields
from abc import ABC, abstractmethod
from ..model.types import RasterizedOutput
from ..dataset.types import BatchedExample
from jaxtyping import Float

T_Config = TypeVar("T_Config")
T_Wrapper = TypeVar("T_Wrapper")


class Loss(nn.Module, ABC, Generic[T_Config, T_Wrapper]):
    config: T_Config
    name: str

    def __init__(self, config: T_Wrapper) -> None:
        super().__init__()

        (field,) = fields(type(config))
        self.config = getattr(config, field.name)
        self.name = field.name

    @abstractmethod
    def forward(self, prediction: RasterizedOutput, batch: BatchedExample) -> Float[Tensor]:
        pass
