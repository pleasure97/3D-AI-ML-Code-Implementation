from dataclasses import dataclass
from typing import Literal
from pathlib import Path
from torch.utils.data import Dataset
from .dataset_common import DatasetConfig
from .dataset_objaverse import DatasetObjaverse
from .dataset_mvimgnet import DatasetMVImgNet
from .dataset_realestate10k import DatasetRealEstate10K
from .dataset_dl3dv10k import DatasetDL3DV10K
from .types import Stage


DATASETS: dict[str, Dataset] = {
    "Objaverse": DatasetObjaverse,
    "MVImgNet" : DatasetMVImgNet,
    "RealEstate10K": DatasetRealEstate10K,
    "DL3DV10K": DatasetDL3DV10K
}

def get_dataset(
    config: DatasetConfig,
    stage: Stage,
    step_tracker: None
) -> Dataset:
    return DATASETS[config.name](config, stage)
