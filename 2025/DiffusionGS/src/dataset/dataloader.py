from dataclasses import dataclass
from lightning.pytorch import LightningDataModule
from . import DatasetConfig, get_dataset
from torch.utils.data import DataLoader
from src.utils.step_tracker import StepTracker

@dataclass
class DataLoaderStageConfig:
    batch_size: int
    num_workers: int
    seed: int | None

@dataclass
class DataLoaderConfig:
    train: DataLoaderStageConfig
    test: DataLoaderStageConfig
    validation: DataLoaderStageConfig

class DataModule(LightningDataModule):
    dataset_config: DatasetConfig
    dataloader_config: DataLoaderConfig
    step_tracker: StepTracker | None
    global_rank: int

    def __init__(self,
                 dataset_config: DatasetConfig,
                 dataloader_config: DataLoaderConfig,
                 step_tracker: StepTracker,
                 global_rank: int=0):
        super().__init__()
        self.dataset_config = dataset_config
        self.dataloader_config = dataloader_config
        self.step_tracker = step_tracker
        self.global_rank = global_rank

    def train_dataloader(self):
        dataset = get_dataset(self.dataset_config, "train", self.step_tracker)

        return DataLoader(
            dataset,
            self.dataloader_config.train.batch_size,
            shuffle=False,
            num_workers=self.dataloader_config.train.num_workers)

    def val_dataloader(self):
        dataset = get_dataset(self.dataset_config, "val", self.step_tracker)

        return DataLoader(
            dataset,
            self.datloader_config.validation.batch_size,
            num_workers=self.dataloader_config.validation.num_workers
        )

    def test_dataloader(self):
        dataset = get_dataset(self.dataset_config, "test", self.step_tracker)

        return DataLoader(
            dataset,
            self.dataset_config.test.batch_size,
            num_workers=self.dataloader_config.test.num_workers
        )