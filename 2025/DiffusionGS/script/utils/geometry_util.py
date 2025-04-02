import torch
from jaxtyping import Float
from torch import Tensor
import torch.nn.functional as F


def get_fov(intrinsics: Float[Tensor, "batch 3 3"]) -> Float[Tensor, "batch 2"]:
    inverse_intrinsics = intrinsics.inverse()

    def convert_to_camera_directional_vector(vector):
        vector = torch.tensor(vector, dtype=torch.float32, device=intrinsics.device).unsqueeze(-1)
        vector = torch.matmul(inverse_intrinsics, vector).squeeze(-1)
        return F.normalize(vector, dim=-1)

    left = convert_to_camera_directional_vector([0, 0.5, 1])
    right = convert_to_camera_directional_vector([1, 0.5, 1])
    top = convert_to_camera_directional_vector([0.5, 0, 1])
    bottom = convert_to_camera_directional_vector([0.5, 1, 1])

    fov_x = torch.acos((left * right).sum(dim=-1))
    fov_y = torch.acos((top * bottom).sum(dim=-1))

    return torch.stack((fov_x, fov_y), dim=-1)
