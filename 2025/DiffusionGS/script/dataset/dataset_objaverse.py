from dataclasses import dataclass
from .dataset_common import DatasetConfigCommon
from typing import Literal
from pathlib import Path
from torch.utils.data import Dataset
from .types import Stage
import torchvision.transforms

@dataclass
class DatasetObjaverseConfig(DatasetConfigCommon):
    name: Literal["Objaverse"]
    roots: list[Path]

class DatasetObjaverse(Dataset):
    config: DatasetObjaverseConfig
    stage: Stage
    to_tensor: torchvision.transforms.ToTensor
    u_near: float = 0.1
    u_far: float = 4.2

    def __init__(self, config: DatasetObjaverseConfig, stage: Stage) -> None:
        super().__init__()
        self.config = config
        self.stage = stage
        self.to_tensor = torchvision.transforms.ToTensor()

    def __iter__(self):
        pass

    def __len__(self) -> int:
        pass

