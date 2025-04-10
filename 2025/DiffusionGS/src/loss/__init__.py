from .loss import Loss
from .denoising_loss import DenoisingLoss, DenoisingLossConfigWrapper
from .novel_view_loss import NovelViewLoss, NovelViewLossConfigWrapper
from .point_distribution_loss import PointDistributionLoss, PointDistributionLossConfigWrapper

LOSSES = {
    DenoisingLossConfigWrapper: DenoisingLoss,
    NovelViewLossConfigWrapper: NovelViewLoss,
    PointDistributionLossConfigWrapper: PointDistributionLoss
}

LossConfigWrapper = DenoisingLossConfigWrapper | NovelViewLossConfigWrapper | PointDistributionLossConfigWrapper


def get_losses(configs: list[LossConfigWrapper]) -> list[Loss]:
    return [LOSSES[type(config)](config) for config in configs]