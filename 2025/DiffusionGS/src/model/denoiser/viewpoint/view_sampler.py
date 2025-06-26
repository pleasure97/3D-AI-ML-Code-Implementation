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
    num_novel_views: int
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
    def compute_angle_between_vectors(vector1: Tensor, vector2: Tensor) -> Tensor:
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
        camera_position = extrinsics[:, :3, 3]  # [num_views, 3]
        camera_forward = -extrinsics[:, :3, 2]  # [num_views, 3]

        # Select Condition View
        clean_index = torch.randint(0, num_views, (1,), device=device).item()
        clean_position = camera_position[clean_index]
        clean_forward = camera_forward[clean_index]

        # Filter based on Position (theta1)
        theta1 = torch.deg2rad(torch.tensor(self.config.theta1, device=device))
        # Filter based on Forward Direction (phi1)
        phi1 = torch.deg2rad(torch.tensor(self.config.phi1, device=device))

        # the angle between the condition view position and i-th noisy view position
        theta_cd = self.compute_angle_between_vectors(camera_position, clean_position.expand_as(camera_position))
        # the angle between the condition view forward direction and i-th noisy view forward direction
        phi_cd = self.compute_angle_between_vectors(camera_forward, clean_forward.expand_as(camera_forward))

        # first and second constraints for noisy views
        noisy_view_mask = (theta_cd <= theta1) & (phi_cd >= torch.cos(phi1))
        noisy_indices_candidates = torch.where(noisy_view_mask)[0]
        N = self.config.num_noisy_views
        num_noisy_indices_candidates = noisy_indices_candidates.numel()
        if num_noisy_indices_candidates >= N:
            noisy_indices = noisy_indices_candidates[torch.randperm(num_noisy_indices_candidates, device=device)[:N]]
        else:
            noisy_extras = torch.randint(0, num_noisy_indices_candidates, (N - num_noisy_indices_candidates,), device=device)
            noisy_extra_indices = noisy_indices_candidates[noisy_extras]
            noisy_indices = torch.cat([noisy_indices_candidates, noisy_extra_indices], dim=0)

        # Exclude 1 clean view and N noisy views from candidate views.
        excluded_indices = torch.cat([clean_index, noisy_indices])
        all_indices = torch.arange(num_views, device=device)
        remaining_indices = all_indices[~torch.isin(all_indices, excluded_indices)]

        # Check if there are no remaining views left.
        if remaining_indices.numel() == 0:
            remaining_indices = noisy_indices_candidates
        # Extract remaining position and forward vectors
        remaining_position = camera_position[remaining_indices]
        remaining_forward = camera_forward[remaining_indices]

        # Filter based on Position (theta2)
        theta2 = torch.deg2rad(torch.tensor(self.config.theta2, device=device))
        # Filter based on Forward Direction (phi2)
        phi2 = torch.deg2rad(torch.tensor(self.config.phi2, device=device))

        # TODO - Refactor into a function like compute_angle_between_vectors()
        normalized_noisy_position = F.normalize(camera_position[noisy_indices], dim=-1) # [num_noisy_indices, 3]
        normalized_remaining_position = F.normalize(remaining_position, dim=-1)         # [num_remaining_indices, 3]
        normalized_cosine_matrix = normalized_noisy_position @ normalized_remaining_position.t() # [num_noisy_indices, num_remaining_indices]
        theta_dn_matrix = torch.acos(normalized_cosine_matrix.clamp(-1., 1.))   # [num_noisy_indices, num_remaining_indices]

        phi_nv = self.compute_angle_between_vectors(camera_forward, remaining_forward.expand_as(camera_forward))

        # first and second constraints for novel views
        novel_view_mask = ((theta_dn_matrix <= theta2).any(dim=0)) & (phi_nv >= torch.cos(phi2))
        novel_indices_candidates = remaining_indices[novel_view_mask]

        M = self.config.num_novel_views
        num_novel_indices_candidates = novel_indices_candidates.numel()
        if novel_indices_candidates.numel() >= M:
            novel_indices = novel_indices_candidates[torch.randperm(num_novel_indices_candidates, device=device)[:M]]
        else:
            novel_extras = torch.randint(0, num_novel_indices_candidates, (M - num_novel_indices_candidates,), device=device)
            novel_extra_indices = novel_indices_candidates[novel_extras]
            novel_indices = torch.cat([novel_indices_candidates, novel_extra_indices], dim=0)

        return clean_index, noisy_indices, novel_indices

    @property
    def num_clean_views(self) -> int:
        return self.config.num_clean_views

    @property
    def num_noisy_views(self) -> int:
        return self.config.num_noisy_views

    @property
    def num_novel_veiws(self) -> int:
        return self.config.num_novel_views
