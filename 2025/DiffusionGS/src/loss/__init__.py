from dataclasses import dataclass
from denoising_loss import DenoisingLossConfig
from novel_view_loss import NovelViewLossConfig
from point_distribution_loss import PointDistributionLossConfig

@dataclass
class LossesConfig:
    denoising: DenoisingLossConfig
    novel_view: NovelViewLossConfig
    point_distribution: PointDistributionLossConfig
