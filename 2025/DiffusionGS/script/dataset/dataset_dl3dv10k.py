from dataclasses import dataclass
from .dataset_common import DatasetConfig
from typing import Literal
from pathlib import Path
from torch.utils.data import Dataset
from .types import Stage
import torchvision.transforms

@dataclass
class DatasetDL3DV10KConfig(DatasetConfig):
    name: Literal["DL3DV10K"]
    roots: list[Path]

class DatasetDL3DV10K(Dataset):
    config: DatasetDL3DV10KConfig
    stage: Stage
    to_tensor: torchvision.transforms.ToTensor
    u_near: float = 0.
    u_far: float = 500.

    def __init__(self, config: DatasetDL3DV10KConfig, stage: Stage) -> None:
        super().__init__()
        self.config = config
        self.stage = stage
        self.to_tensor = torchvision.transforms.ToTensor()

    def __iter__(self):
        pass

    def __len__(self) -> int:
        pass

