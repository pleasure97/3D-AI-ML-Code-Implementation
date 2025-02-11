from scipy.spatial import KDTree
import torch
import numpy as np

def distCUDA2(points: np.ndarray, device: torch.device=device):
  """
    Calculates the average squared distance to the 3 nearest neighbors
    for each point in a point cloud using a KDTree.

    Args:
        points (np.ndarray): Input point cloud with shape (N, 3),
            where N is the number of points, and each point has 3 coordinates (x, y, z).
        device (torch.device, optional): The target device (e.g., 'cuda' or 'cpu')
            where the resulting tensor will be stored. Defaults to `device`.

    Returns:
        torch.Tensor: A 1D tensor of shape (N,) containing the average squared distances
            to the 3 nearest neighbors for each point.

  Notes:
    In source code, they use `from simple_knn._C import distCUDA2`.
    This version re-implements the functionality purely using PyTorch and SciPy,
          based on a solution by @rfeinman (see https://github.com/graphdeco-inria/gaussian-splatting/issues/292).
  """
  dists, indices = KDTree(points).query(points, k=4)
  meanDists = (dists[:, 1:] ** 2).mean(1)

  return torch.tensor(meanDists, device=device)
import torch
import numpy as np

def make_rotation_matrix(q: torch.Tensor, device: torch.device=device) -> torch.Tensor:
  """
  Args:
    q - quaternion, a tensor with shape (1, 4)
    device - torch.device, "cuda" or "cpu"
  Returns:
    R - Rotation matrix, a tensor with shape (3, 3)
  """
  q_r, q_i, q_j, q_k = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

  if device == "cuda":
    R = torch.zeros((3, 3), device="cuda")
  elif device == "cpu":
    R = torch.zeros((3, 3), device="cpu")

  R[0, 0] = 1 - 2 * (q_j ** 2 + q_k ** 2)
  R[1, 1] = 1 - 2 * (q_i ** 2 + q_k ** 2)
  R[2, 2] = 1 - 2 * (q_i ** 2 + q_j ** 2)

  R[0, 1] = 2 * (q_i * q_j - q_r * q_k)
  R[1, 0] = 2 * (q_i * q_j + q_r * q_k)

  R[0, 2] = 2 * (q_i * q_k + q_r * q_j)
  R[2, 0] = 2 * (q_i * q_k - q_r * q_j)

  R[1, 2] = 2 * (q_j * q_k - q_r * q_i)
  R[2, 1] = 2 * (q_j * q_k + q_r * q_i)

  return R

def make_scaling_matrix(points: np.array, device: torch.device=device) -> torch.Tensor:
  """
  Args:
    points - derived from SfM point cloud, should be in CUDA
    device - torch.device, "cuda" or "cpu"
  Returns:
    S - Scaling matrix, a tensor with shape (3, 3)

  """

  points_distances = torch.clamp(distCUDA2(points).float().to(device), min=1e-7)
  print(f"points_distances shape : {points_distances.shape}")

  scale_factors = torch.log(torch.sqrt(points_distances)).mean(dim=0)

  S = torch.full((3, 3), scale_factors, device=device)
  print(f"S shape : {S.shape}")
  return S

def calc_covariance_matrix (R : torch.Tensor, S: torch.Tensor) -> torch.Tensor:
  """
  Args:
    R - Rotation matrix, a tensor with shape (3, 3)
    S - Scaling matrix, a tensor with shape (3, 3)
  Returns:
    SIGMA - Covariance matrix, a tensor with shape (3, 3)
  """

  RS = (R @ S)
  SIGMA = RS @ RS.T

  return SIGMA
import torch

def inverse_sigmoid(x: torch.Tensor):
  return torch.log(x / (1-x))
### Modified and Restructured from PleNoxels Soruce Code ###
def exponential_learning_rate_function(lr_init, lr_final, lr_delay_steps=0, lr_delay_multiplier=1., max_steps=1_000_000):
  def internal_process_learning_rate(step):
    if step < 0 or (lr_init == 0. and lr_final == 0.):
      return 0.
    if lr_delay_steps > 0:
      delay_rate = lr_delay_multiplier + (1 - lr_delay_multiplier) * np.sin(0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1))
    else:
      delay_rate = 1.
    timestep = np.clip(step / max_steps, 0, 1)
    log_interpolation = np.exp(np.log(lr_init) * (1 - timestep) + np.log(lr_final) * timestep)
    return delay_rate * log_interpolation
  return internal_process_learning_rate
