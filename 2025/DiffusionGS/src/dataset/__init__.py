from torch.utils.data import Dataset
from src.dataset.dataset_common import DatasetConfig
from src.dataset.dataset_objaverse import DatasetObjaverse
from src.dataset.dataset_mvimgnet import DatasetMVImgNet
from src.dataset.dataset_realestate10k import DatasetRealEstate10K
from src.dataset.dataset_dl3dv10k import DatasetDL3DV10K
from src.dataset.types import Stage
from src.utils.step_tracker import StepTracker


DATASETS: dict[str, Dataset] = {
    "Objaverse": DatasetObjaverse,
    "MVImgNet" : DatasetMVImgNet,
    "RealEstate10K": DatasetRealEstate10K,
    "DL3DV10K": DatasetDL3DV10K
}

def get_dataset(
    config: DatasetConfig,
    stage: Stage,
    step_tracker: StepTracker | None
) -> Dataset:
    return DATASETS[config.name](config, stage)
