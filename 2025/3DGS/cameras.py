CAMERA_MODEL_IDS = {
    0: {"model_id": 0, "model_name": "SIMPLE_PINHOLE", "num_params": 3},
    1: {"model_id": 1, "model_name": "PINHOLE", "num_params": 4},
    2: {"model_id": 2, "model_name": "SIMPLE_RADIAL", "num_params": 4},
    3: {"model_id": 3, "model_name": "RADIAL", "num_params": 5},
    4: {"model_id": 4, "model_name": "OPENCV", "num_params": 8},
    5: {"model_id": 5, "model_name": "OPENCV_FISHEYE", "num_params": 8},
    6: {"model_id": 6, "model_name": "FULL_OPENCV", "num_params": 12},
    7: {"model_id": 7, "model_name": "FOV", "num_params": 5},
    8: {"model_id": 8, "model_name": "SIMPLE_RADIAL_FISHEYE", "num_params": 4},
    9: {"model_id": 9, "model_name": "RADIAL_FISHEYE", "num_params": 5},
    10: {"model_id": 10, "model_name": "THIN_PRISM_FISHEYE", "num_params": 12}
}

CAMERA_MODEL_NAMES = {
    "SIMPLE_PINHOLE": {"model_id": 0, "model_name": "SIMPLE_PINHOLE", "num_params": 3},
    "PINHOLE": {"model_id": 1, "model_name": "PINHOLE", "num_params": 4},
    "SIMPLE_RADIAL": {"model_id": 2, "model_name": "SIMPLE_RADIAL", "num_params": 4},
    "RADIAL": {"model_id": 3, "model_name": "RADIAL", "num_params": 5},
    "OPENCV": {"model_id": 4, "model_name": "OPENCV", "num_params": 8},
    "OPENCV_FISHEYE": {"model_id": 5, "model_name": "OPENCV_FISHEYE", "num_params": 8},
    "FULL_OPENCV": {"model_id": 6, "model_name": "FULL_OPENCV", "num_params": 12},
    "FOV": {"model_id": 7, "model_name": "FOV", "num_params": 5},
    "SIMPLE_RADIAL_FISHEYE": {"model_id": 8, "model_name": "SIMPLE_RADIAL_FISHEYE", "num_params": 4},
    "RADIAL_FISHEYE": {"model_id": 9, "model_name": "RADIAL_FISHEYE", "num_params": 5},
    "THIN_PRISM_FISHEYE": {"model_id": 10, "model_name": "THIN_PRISM_FISHEYE", "num_params": 12}
}
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
import math
from diff_gaussian_rasterization.diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

class Camera(nn.Module):
  def __init__(self, resolution, uid, colmap_id, R, T, FovX, FovY, depth_params,
               image, inv_depth_map, image_name, trans=np.array([0., 0., 0.]),
               scale=1., device=device):
    super().__init__()

    self.uid = uid
    self.colmap_id = colmap_id
    self.R = R
    self.T = T
    self.FovX = FovX
    self.FovY = FovY
    self.image_name = image_name

    resized_image = image.resize(resolution)
    resized_image_rgb = torch.from_numpy(np.array(resized_image)) / 255.
    if len(resized_image_rgb.shape) == 3:
      resized_image_rgb.permute(2, 0, 1)
    else:
      resized_image_rgb.unsqueeze(dim=-1).permute(2, 0, 1)
    ground_truth_image = resized_image_rgb[:3, ...]
    self.alpha_mask = None
    if (resized_image_rgb.shape[0] == 4):
      self.alpha_mask = resized_image_rgb[3:4, ...].to(device)
    else:
      self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(device))

    self.original_image = ground_truth_image.clamp(0., 1.).to(device)
    self.image_width = self.original_image.shape[2]
    self.image_height = self.original_image.shape[1]

    self.inv_depth_map = None
    self.depth_reliable = False

    self.zfar = 100.
    self.znear = 0.01

    self.trans = trans
    self.scale = scale

    self.world_view_transform = torch.tensor(get_world_to_view_to(R, T, trans, scale)).transpose(0, 1).to(device)
    self.projection_matrix = get_projection_matrix(self.z_near, self.z_far, self.FovX, self.FoxY).transpose(0, 1)
    self.full_projective_transform = self.world_view_transform @ self.projection_matrix
    self.camera_center = self.world_view_transform.inverse()[3, :3]

def render(viewpoint_camera,
           model: torch.nn.Module,
           background_color: torch.Tensor,
           scaling_modifier=1.0,
           compute_3d_covariances=False,
           separate_sh=True,
           override_color=None):

  screen_space_points = torch.zeros_like(model.xyz, dtype=model.xyz.dtype, requires_grad=True, device=device)
  screen_space_points.retain_grad()

  tan_FovX = math.tan(viewpoint_camera.FovX * 0.5)
  tan_FovY = math.tan(viewpoint_camera.FovY * 0.5)

  rasterization_settings = GaussianRasterizationSettings(
      image_height=int(viewpoint_camera["height"]),
      image_width=int(viewpoint_camera["width"]),
      tanfovx=tan_FovX,
      tanfovy=tan_FovY,
      bg=background_color,
      scale_modifier=scaling_modifier,

  )

  rasterizer = GaussianRasterizer(raster_settings=rasterization_settings)

  means3D = model.xyz
  means2D = screen_space_points
  opacities = model.opacities

  scales = model.scaling_vectors
  rotations = model.quaternions
  precomputed_3d_covariances = None

  shs = None
  precomputed_colors = None

  if separate_sh:
    dc, shs = model.features_diffuse_color, model.features_rest
    rendered_image, radii, depth_image = rasterizer(
        means3D=means3D,
        means2D=means2D,
        dc=dc,
        shs=shs,
        colors_precomp=precomputed_colors,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=precomputed_3d_covariances
    )

  rendered_image = rendered_image.clamp(0, 1)
  return {"render": rendered_image,
          "viewspace_points": screen_space_points,
          "visibility_filter": (radii > 0).nonzero(),
          "radii": radii,
          "depth": depth_image}
