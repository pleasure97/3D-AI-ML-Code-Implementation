from dataclasses import dataclass
from src.loss.denoising_loss import DenoisingLossConfig
from src.loss.novel_view_loss import NovelViewLossConfig
from src.loss.point_distribution_loss import PointDistributionLossConfig
@dataclass
class LossesConfig:
    denoising: DenoisingLossConfig
    novel_view: NovelViewLossConfig
    point_distribution: PointDistributionLossConfig

