from dataclasses import dataclass
from typing import Literal
from script.dataset.types import Stage
from jaxtyping import Float, Int64
import torch
from torch import Tensor


@dataclass
class ViewSamplerConfig:
    name: Literal["ViewSampler"]
    num_source_views: int
    num_target_views: int
    source_views: list[int] | None
    target_views: list[int] | None


class ViewSampler:
    config: ViewSamplerConfig
    stage: Stage
    cameras_are_circular: bool
    step_tracker: None

    def __init__(self, config: ViewSamplerConfig, stage: Stage, cameras_are_circular: bool, step_tracker: None) -> None:
        self.config = config
        self.stage = stage
        self.cameras_are_circular = cameras_are_circular
        self.step_tracker = None

    def sample(self,
               scene: str,
               extrinsics: Float[Tensor, "view 4 4"],
               intrinsics: Float[Tensor, "view 3 3"],
               device: torch.device = "cuda" if torch.cuda.is_available() else "cpu") \
            -> tuple[Int64[Tensor, " source_view"], Int64[Tensor, " target_view"]]:
        num_views, _, _ = extrinsics.shape

        source_index = torch.randint(0, num_views, size=(self.config.num_source_views,), device=device)
        if self.config.num_source_views is not None:
            source_index = torch.tensor(self.config.source_views, dtype=torch.int64, device=device)

        target_index = torch.randint(0, num_views, size=(self.config.num_target_views,), device=device)
        if self.config.num_target_views is not None:
            target_index = torch.tensor(self.config.target_views, dtype=torch.int64, device=device)

        return source_index, target_index

    @property
    def num_source_views(self) -> int:
        return self.config.num_source_views

    @property
    def num_target_views(self) -> int:
        return self.config.num_target_views
