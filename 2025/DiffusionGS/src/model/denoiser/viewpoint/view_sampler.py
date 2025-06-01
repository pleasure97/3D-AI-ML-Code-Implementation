from dataclasses import dataclass
from typing import Literal
from src.preprocess.types import Stage
from src.utils.step_tracker import StepTracker
from jaxtyping import Float, Int64
import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass
class ViewSamplerConfig:
    name: Literal["view_sampler"]
    num_clean_views: int
    num_noisy_views: int
    theta1: float   # Maximum Angle between Noisy View and Condition View (45.)
    theta2: float   # Maximum Angle between Noisy View and Novel View (60.)
    phi1: float     # Minimum Cosine Angle between Noisy Direction and Condition Direction (60.)
    phi2: float     # Minimum Cosine Angle between Novel Direction and Condition Direction (75.)


class ViewSampler:
    config: ViewSamplerConfig
    stage: Stage
    step_tracker: StepTracker | None

    def __init__(self, config: ViewSamplerConfig, stage: Stage, step_tracker: StepTracker) -> None:
        self.config = config
        self.stage = stage
        self.step_tracker = step_tracker

    @staticmethod
    def compute_angle_between(vector1: Tensor, vector2: Tensor) -> Tensor:
        vector1 = F.normalize(vector1, dim=-1)
        vector2 = F.normalize(vector2, dim=-1)
        dot_product = torch.sum(vector1 * vector2, dim=-1).clamp(-1., 1.)
        return torch.acos(dot_product)

    def sample(self,
               extrinsics: Float[Tensor, "view 4 4"],
               device: torch.device = "cpu") \
            -> tuple[Int64[Tensor, " source_view"], Int64[Tensor, " target_view"]]:
        num_views = extrinsics.shape[0]

        # Extract Camera Location and Direction Vector
        camera_position = extrinsics[:, :3, 3]
        camera_forward = -extrinsics[:, :3, 2]

        # Select Condition View
        condition_index = torch.randint(0, num_views, (1,), device=device).item()
        condition_position = camera_position[condition_index]
        condition_direction = camera_forward[condition_index]
        expanded_condition_position = condition_position.unsqueeze(0).expand_as(camera_position)
        expanded_condition_direction = condition_direction.unsqueeze(0).expand_as(camera_forward)

        # Filter based on Position (theta1, theta2)
        theta1 = torch.deg2rad(torch.tensor(self.config.theta1, device=device))
        theta2 = torch.deg2rad(torch.tensor(self.config.theta2, device=device))

        # Filter based on Direction (phi1, phi2)
        phi1 = torch.deg2rad(torch.tensor(self.config.phi1, device=device))
        phi2 = torch.deg2rad(torch.tensor(self.config.phi2, device=device))

        # Compute Position Angles
        positions_angles = self.compute_angle_between(camera_position - condition_position,
                                                      expanded_condition_position)

        # Compute Forward Direction Angles
        forward_angles = self.compute_angle_between(camera_forward, condition_direction.expand_as(camera_forward))

        # Noisy Views
        valid_clean_mask = (positions_angles <= theta1) & (forward_angles <= phi1)
        valid_clean_indices = torch.where(valid_clean_mask)[0]

        # Novel Views
        if valid_clean_indices.numel() > 0:
            source_position_normalized = F.normalize(camera_position[valid_clean_indices], dim=-1)
            all_position_normalized = F.normalize(camera_position, dim=-1)
            dot_matrix = torch.matmul(all_position_normalized, source_position_normalized.t()).clamp(-1., 1.)
            angle_matrix = torch.acos(dot_matrix)
            novel_position_mask = torch.any(angle_matrix <= theta2, dim=1)
        else:
            novel_position_mask = torch.zeros(num_views, dtype=torch.bool, device=device)
        orientation_angle_to_condition = self.compute_angle_between(camera_forward, expanded_condition_direction)
        novel_orientation_mask = orientation_angle_to_condition <= phi2

        valid_noisy_mask = novel_position_mask & novel_orientation_mask
        valid_noisy_indices = torch.where(valid_noisy_mask)[0]

        if valid_clean_indices.numel() >= self.config.num_clean_views:
            clean_index = valid_clean_indices[torch.randperm(valid_clean_indices.numel(), device=device)][:self.config.num_clean_views]
        else:
            choices = torch.randint(0, valid_clean_indices.numel(), (self.config.num_clean_views,), device=device)
            clean_index = valid_clean_indices[choices]

        if valid_noisy_indices.numel() >= self.config.num_noisy_views:
            noisy_indices = valid_noisy_indices[torch.randperm(valid_noisy_indices.numel(), device=device)][:self.config.num_noisy_views]
        else:
            choices = torch.randint(0, valid_noisy_indices.numel(), (self.config.num_noisy_views,), device=device)
            noisy_indices = valid_noisy_indices[choices]

        return clean_index, noisy_indices

    @property
    def num_clean_views(self) -> int:
        return self.config.num_clean_views

    @property
    def num_noisy_views(self) -> int:
        return self.config.num_noisy_views
