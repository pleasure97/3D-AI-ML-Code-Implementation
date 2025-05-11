from torch.utils.data import Dataset
from src.preprocess.dataset_common import DatasetConfig
from src.preprocess.dataset_objaverse import DatasetObjaverse
from src.preprocess.dataset_mvimgnet import DatasetMVImgNet
from src.preprocess.dataset_realestate10k import DatasetRealEstate10K
from src.preprocess.dataset_dl3dv10k import DatasetDL3DV10K
from src.preprocess.types import Stage
from src.utils.step_tracker import StepTracker
from src.model.denoiser.viewpoint.view_sampler import ViewSampler


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
    view_sampler = ViewSampler(config.view_sampler, stage, step_tracker)
    return DATASETS[config.name](config, stage, view_sampler)
