from torch import nn
from typing import TypeVar, Generic
from abc import ABC


T_Config = TypeVar("T_Config")


class Loss(nn.Module, ABC, Generic[T_Config]):
    config: T_Config
    name: str

    def __init__(self, config: T_Config) -> None:
        super().__init__()
        self.name = self.__class__.name
