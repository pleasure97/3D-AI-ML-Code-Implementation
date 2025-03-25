from dataclasses import dataclass
from typing import Literal
from pathlib import Path
@dataclass
class DatasetConfig:
    name: Literal["Objaverse", "MVImgNet", "RealEstate10K", "DL3DV10K"]
    roots: list[Path]
