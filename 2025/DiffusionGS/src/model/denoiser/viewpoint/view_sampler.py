from dataclasses import dataclass
from typing import Literal
from src.dataset.types import Stage
from src.utils.step_tracker import StepTracker
from jaxtyping import Float, Int64
import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass
class ViewSamplerConfig:
    name: Literal["ViewSampler"]
    num_source_views: int
    num_target_views: int
    source_views: list[int] | None
    target_views: list[int] | None
    theta1: float   # Maximum Angle between Noisy View and Condition View (45.)
    theta2: float   # Maximum Angle between Noisy View and Novel View (60.)
    phi1: float     # Minimum Cosine Angle between Noisy Direction and Condition Direction (60.)
    phi2: float     # Minimum Cosine Angle between Novel Direction and Condition Direction (75.)


class ViewSampler:
    config: ViewSamplerConfig
    stage: Stage
    cameras_are_circular: bool
    step_tracker: StepTracker | None

    def __init__(self, config: ViewSamplerConfig, stage: Stage, cameras_are_circular: bool, step_tracker: StepTracker) -> None:
        self.config = config
        self.stage = stage
        self.cameras_are_circular = cameras_are_circular
        self.step_tracker = step_tracker

    @staticmethod
    def compute_angle_between(self, vector1: Tensor, vector2: Tensor) -> Tensor:
        vector1 = F.normalize(vector1, dim=-1)
        vector2 = F.normalize(vector2, dim=-1)
        dot_product = torch.sum(vector1 * vector2, dim=-1).clamp(-1., 1.)
        return torch.rad2deg(torch.acos(dot_product))

    def sample(self,
               extrinsics: Float[Tensor, "view 4 4"],
               device: torch.device = "cuda" if torch.cuda.is_available() else "cpu") \
            -> tuple[Int64[Tensor, " source_view"], Int64[Tensor, " target_view"]]:
        num_views = extrinsics.shape[0]

        # Extract Camera Location and Direction Vector
        camera_position = extrinsics[:, :3, 3]
        camera_forward = -extrinsics[:, :3, 2]

        # Select Condition View
        condition_index = torch.randint(0, num_views, (1,), device=device).item()
        condition_position = camera_position[condition_index]
        condition_direction = camera_forward[condition_index]

        # Filter based on Position (theta1, theta2)
        theta1 = torch.deg2rad(torch.tensor(self.config.theta1, device=device))
        theta2 = torch.deg2rad(torch.tensor(self.config.theta2, device=device))

        positions_angles = self.compute_angle_between(camera_position - condition_position,
                                                      torch.zeros_like(camera_position, device=device))

        # Filter based on Direction (phi1, phi2)
        phi1 = torch.deg2rad(torch.tensor(self.config.phi1, device=device))
        phi2 = torch.deg2rad(torch.tensor(self.config.phi2, device=device))

        forward_angles = self.compute_angle_between(camera_forward, condition_direction.expand_as(camera_forward))

        # Noisy Views
        valid_source_mask = (positions_angles <= theta1) & (forward_angles <= phi1)
        valid_source_indices = torch.where(valid_source_mask)[0]

        # Novel Views
        valid_target_mask = (positions_angles <= theta2) & (forward_angles <= phi2)
        valid_target_indices = torch.where(valid_target_mask)[0]

        source_index = valid_source_indices[torch.randperm(len(valid_source_indices))][:self.config.num_source_views]
        target_index = valid_target_indices[torch.randperm(len(valid_target_indices))][:self.config.num_target_views]

        return source_index, target_index

    @property
    def num_source_views(self) -> int:
        return self.config.num_source_views

    @property
    def num_target_views(self) -> int:
        return self.config.num_target_views
