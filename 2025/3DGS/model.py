from torch import nn
import numpy as np

class GaussianSplattingModelV1(nn.Module):
  def __init__(self, points, colors):

    super().__init__()

    points_to_tensor, features_diffuse_color, features_rest, quaternions, scaling_vectors, opacities = self.initialize_from_gaussians(points, colors)

    self.xyz = nn.Parameter(points_to_tensor, requires_grad=True)
    self.features_diffuse_color = nn.Parameter(features_diffuse_color, requires_grad=True)
    self.features_rest = nn.Parameter(features_rest, requires_grad=True)
    self.quaternions = nn.Parameter(quaternions, requires_grad=True)
    self.scaling_vectors = nn.Parameter(scaling_vectors, requires_grad=True)
    self.opacities = nn.Parameter(opacities, requires_grad=True)

    # self.update_covariances()

    self.set_activation_functions()

  def update_covariances(self):
    self.covariances = calc_covariance_matrix(make_rotation_matrix(self.quaternions), make_scaling_matrix(self.scaling_vectors))

  def initialize_from_gaussians(self, points, colors, max_sh_degree=3, C0=0.28209479177387814):
    # Conver numpy array of points to torch.Tensor
    points_to_tensor = torch.tensor(np.asarray(points), device=device)

    # Conver numpy array of colors to torch.Tensor and simultaneously convert RGB to Spherical Harmonics coefficients.
    colors_to_SH = (torch.tensor(np.asarray(colors), device=device) - 0.5) / C0
    features = torch.zeros((colors_to_SH.shape[0],  3, (max_sh_degree + 1) ** 2), device=device)

    features[:, :3, 0] = colors_to_SH
    features[:, 3:, 1:] = 0.

    features_diffuse_color = features[:, :, 0:1].transpose(1, 2).contiguous()
    features_rest = features[:, :, 1:].transpose(1, 2).contiguous()

    quaternions = torch.zeros((points_to_tensor.shape[0], 4), device=device)
    quaternions[:, 0] = 1

    distances = torch.clamp_min(distCUDA2(points), 1e-7)
    scaling_vectors = torch.log(torch.sqrt(distances)).unsqueeze(-1).expand(-1, 3)

    opacities = inverse_sigmoid(0.1 * torch.ones((points_to_tensor.shape[0], 1), device=device))

    return points_to_tensor, features_diffuse_color, features_rest, quaternions, scaling_vectors, opacities

  def set_activation_functions(self, opacity_activation=torch.sigmoid, covariance_scale_activation=torch.exp):
    self.opacity_activation = opacity_activation
    self.covariance_scale_activation = covariance_scale_activation


  def print_model_parameters(self):
    print(f"{'Parameter Name':<30} {'Shape':<30} {'Trainable':<10}")
    print("-" * 70)
    for name, param in self.named_parameters():
        print(f"{name:<30} {str(list(param.shape)):<30} {param.requires_grad}")

  def forward(self, x:torch.Tensor):
    return self.xyz, self.features_diffuse_color, self.features_rest, self.quaternions, self.scaling_vectors, self.opacities
from torch import nn
import numpy as np
import os
from plyfile import PlyData, PlyElement

### Modified and Restructured from Gaussian Splatting Source Code ###
class GaussianSplattingModelV2(nn.Module):
  def __init__(self,
               points: torch.Tensor,
               colors: torch.Tensor,
               camera_infos: dict,
               optimizer: torch.optim=torch.optim.Adam,
               iterations: int=7_000):

    super().__init__()

    points_to_tensor, features_diffuse_color, features_rest, quaternions, scaling_vectors, opacities, max_radii = self.initialize_from_gaussians(points, colors)

    self.xyz = nn.Parameter(points_to_tensor, requires_grad=True)
    self.features_diffuse_color = nn.Parameter(features_diffuse_color, requires_grad=True)
    self.features_rest = nn.Parameter(features_rest, requires_grad=True)
    self.quaternions = nn.Parameter(quaternions, requires_grad=True)
    self.scaling_vectors = nn.Parameter(scaling_vectors, requires_grad=True)
    self.opacities = nn.Parameter(opacities, requires_grad=True)
    self.max_radii = nn.Parameter(max_radii, requires_grad=True)

    self.camera_infos = camera_infos

    # self.update_covariances()

    self.set_activation_functions()

    self.initialize_optimization_params(optimizer)

    self.iterations = iterations

    self.set_learning_rate_schedulers(iterations)

  def update_covariances(self):
    self.covariances = calc_covariance_matrix(make_rotation_matrix(self.quaternions), make_scaling_matrix(self.scaling_vectors))

  def initialize_from_gaussians(self, points, colors, max_sh_degree=3, C0=0.28209479177387814):
    # Conver numpy array of points to torch.Tensor
    points_to_tensor = torch.tensor(np.asarray(points), device=device)

    # Conver numpy array of colors to torch.Tensor and simultaneously convert RGB to Spherical Harmonics coefficients.
    colors_to_SH = (torch.tensor(np.asarray(colors), device=device) - 0.5) / C0
    features = torch.zeros((colors_to_SH.shape[0],  3, (max_sh_degree + 1) ** 2), device=device)

    features[:, :3, 0] = colors_to_SH
    features[:, 3:, 1:] = 0.

    features_diffuse_color = features[:, :, 0:1].transpose(1, 2).contiguous()
    features_rest = features[:, :, 1:].transpose(1, 2).contiguous()

    quaternions = torch.zeros((points_to_tensor.shape[0], 4), device=device)
    quaternions[:, 0] = 1

    distances = torch.clamp_min(distCUDA2(points), 1e-7)
    scaling_vectors = torch.log(torch.sqrt(distances)).unsqueeze(-1).expand(-1, 3)

    opacities = inverse_sigmoid(0.1 * torch.ones((points_to_tensor.shape[0], 1), device=device))

    max_radii = torch.zeros((points_to_tensor.shape[0]), device=device)

    return points_to_tensor, features_diffuse_color, features_rest, quaternions, scaling_vectors, opacities, max_radii

  def set_activation_functions(self,
                               opacity_activation=torch.sigmoid,
                               covariance_scale_activation=torch.exp,
                               scaling_split_activation=torch.log):
    self.opacity_activation = opacity_activation
    self.covariance_scale_activation = covariance_scale_activation
    self.scaling_split_activation = scaling_split_activation

  def initialize_optimization_params(self, optimizer: torch.optim):

    self.xyz_graidents = torch.zeros((self.xyz.shape[0], 1), device=device)
    self.denominator = torch.zeros((self.xyz.shape[0], 1), device=device)

    param_groups = [
        {'params': [self.xyz], 'lr': 1.6e-4, 'name': 'positions'},
        {'params': [self.features_diffuse_color], 'lr': 2.5e-3, 'name': 'features_diffuse_color'},
        {'params': [self.features_rest], 'lr': 2.5e-3, 'name': 'features_dc'},
        {'params': [self.quaternions], 'lr': 1e-3, 'name': 'rotations'},
        {'params': [self.opacities], 'lr': 2.5e-2, 'name': 'opacities'},
        {'params': [self.scaling_vectors], 'lr': 5e-3, 'name': 'scales'},
    ]

    self.optimizer = optimizer(params=param_groups, lr=0.)

    exposure = torch.eye(3, 4, device=device).unsqueeze(0).repeat(len(self.camera_infos), 1, 1)
    self.exposure = nn.Parameter(exposure, requires_grad=True)
    self.exposure_optimizer = torch.optim.Adam([self.exposure])

  def set_learning_rate_schedulers(self,
                              position_lr_init: float=1.6e-4,
                              position_lr_final: float=1.6e-6,
                              position_lr_delay_multiplier: float=1e-2,
                              exposure_lr_init: float=1e-2,
                              exposure_lr_final: float=1e-3,
                              exposure_lr_delay_steps: int=0,
                              exposure_lr_delay_multiplier: float=0.,
                              max_steps: int=7_000):
    self.xyz_scheduler = exponential_learning_rate_function(
        lr_init=position_lr_init,
        lr_final=position_lr_final,
        lr_delay_multiplier=position_lr_delay_multiplier,
        max_steps=max_steps)

    self.exposure_scheduler = exponential_learning_rate_function(
        lr_init=exposure_lr_init,
        lr_final=exposure_lr_final,
        lr_delay_steps=exposure_lr_delay_steps,
        lr_delay_multiplier=exposure_lr_delay_multiplier,
        max_steps=iterations)

  def update_learning_rate(self):
    for param_group in self.exposure_optimizer.param_groups:
      param_group["lr"] = self.exposure_scheduler(self.iterations)

    for param_group in self.optimizer.param_groups:
      if param_group["name"] == "positions":
        lr = self.xyz_scheduler(self.iterations)
        param_group["lr"] = lr
        return lr

  ### Updated from V1 to V2 - 1. remove_gaussian() ###
  def remove_gaussian(self, epsilon_alpha=1/255, radius_threshold=20):
    """ Removes any gaussians that are essentially transparent with alpha less than a threshold."""
    is_transparent = (self.opacities < epsilon_alpha).squeeze()
    is_too_large_radius = self.max_radii > radius_threshold
    remove_mask = torch.logical_and(~is_transparent, ~is_too_large_radius)

    optimizable_tensors = {}
    for group in self.optimizer.param_groups:
      stored_state = self.optimizer.state.get(group['params'][0], None)
      if stored_state is None:
        group["params"][0] = nn.Parameter(group["params"][0][remove_mask], requires_grad=True)
        optimizable_tensors[group["name"]] = group["params"][0]
      else:
        stored_state["1st_momentum"] = stored_state["1st_momentum"][remove_mask]
        stored_state["2nd_momentum"] = stored_state["2nd_momentum"][remove_mask]

        del self.optimizer.state[group["params"][0]]
        group["params"][0] = nn.Parameter(group["params"][0][remove_mask], requires_grad=True)
        self.optimizer.state[group["params"][0]] = stored_state

        optimizable_tensors[group["name"]] = group["params"][0]

    self.xyz = optimizable_tensors["xyz"]
    self.features_diffuse_color = optimizable_tensors["features_diffuse_color"]
    self.features_rest = optimizable_tensors["features_rest"]
    self.opacities = optimizable_tensors["opacities"]
    self.scaling_vectors = optimizable_tensors["scaling_vectors"]
    self.quaternions = optimizable_tensors["quaternions"]
    self.xyz_gradients = self.xyz_gradients[remove_mask]
    self.denominator = self.denominator[remove_mask]
    self.max_radii = self.max_radii[remove_mask]

  ### Updated from V1 to V2 - 3. clone_gaussian() ###
  def clone_gaussian(self, tau_pos=2e-4, percent_dense=1e-2, scene_extent=0.):
    # Calculate Normalized Gradient Vectors
    grads = self.xyz_gradients / self.denominator
    grads[grads.isnan()] = 0.
    # Distinguish Gaussians to be cloned with a threshold, tau_pos
    clone_mask = torch.where(torch.norm(grads, dim=-1) >= tau_pos, True, False)

    ## Why use scene_extent?
    # 1. Avoid issues with outliers in the initial point cloud
    # 2. Adjust the learning rate relative to the scene size
    # 3. Balance between coordinate range and physical scene size
    if scene_extent:
        clone_mask = torch.logical_and(clone_mask,
                                       torch.max(self.scaling_vectors, dim=1).values <= percent_dense * (scene_extent if scene_extent else 1))

    masked_tensors = {
      "xyz": self.xyz[clone_mask],
      "features_diffuse_color": self.features_diffuse_color[clone_mask],
      "features_rest": self.features_rest[clone_mask],
      "opacities": self.opacities[clone_mask],
      "scaling_vectors": self.scaling_vectors[clone_mask],
      "quaternions": self.quaternions[clone_mask]
    }

    clone_tensors = {}
    for group in self.optimizer.param_groups:
      assert len(group["params"]) == 1
      clone_tensor = masked_tensors[group["name"]]
      stored_state = self.optimizer.state.get(group["params"][0], None)
      if stored_state is not None:
        stored_state["1st_momentum"] = torch.cat((stored_state["1st_momentum"], torch.zeros_like(clone_tensor)), dim=0)
        stored_state["2nd_momentum"] = torch.cat((stored_state["2nd_momentum"], torch.zeros_like(clone_tensor)), dim=0)

        del self.optimizer.state[group["params"][0]]
        self.optimizer.state[group["params"][0]] = stored_state

        group["params"][0] = nn.Parameter(torch.cat((group["params"][0], clone_tensor), dim=0), requires_grad=True)
        clone_tensors[group["name"]] = group["params"][0]
      else:
        group["params"][0] = nn.Parameter(torch.cat((group["params"][0], clone_tensor), dim=0), requires_grad=True)

        clone_tensors[group["name"]] = group["params"][0]

    self.xyz = clone_tensors["xyz"]
    self.features_diffuse_color = clone_tensors["features_diffuse_color"]
    self.features_rest = clone_tensors["features_rest"]
    self.opacities = clone_tensors["opacities"]
    self.scaling_vectors = clone_tensors["scaling_vectors"]
    self.quaternions = clone_tensors["quaternions"]

    self.xyz_gradients = torch.zeros((self.xyz.shape[0], 1), device=device)
    self.denominator = torch.zeros((self.xyz.shape[0], 1), device=device)
    self.max_radii = torch.zeros((self.xyz.shape[0]), device=device)


  ### Updated from V1 to V2 - 2. split_gaussian() ###
  def split_gaussian(self, tau_pos=2e-4, percent_dense=1e-2, scene_extent=0., num_splits=2):
    # Calculate Normalized Gradient Vectors
    grads = self.xyz_gradients / self.denominator
    grads[grads.isnan()] = 0.
    padded_grad = torch.zeros((self.xyz.shape[0]), device=device)
    padded_grad[:grads.shape[0]] = grads.squeeze()

    split_mask = torch.where(padded_grad >= tau_pos, True, False)
    split_mask = torch.logical_and(split_mask,
                                   torch.max(self.scaling_vectors, dim=1).values > percent_dense * (scene_extent if scene_extent else 1))

    stds = self.scaling_vectors[split_mask].repeat(num_splits, 1)
    means = torch.zeros((stds.size[0], 3), device=device)
    samples = torch.normal(mean=means, std=stds)
    rotations = make_rotation_matrix(self.quaternions[split_mask]).repeat(num_splits, 1, 1)

    masked_tensors = {
        "xyz": self.xyz[split_mask].repeat(num_splits, 1) + torch.bmm(rotations, samples.unsqueeze(-1)).squeeze(-1),
        "scaling_vectors": self.scaling_split_activation(self.scaling_vectors[split_mask].repeat(num_splits, 1)), # In source code, scaling_vectors are divided by 0.8 * num_splits
        "quaternions": self.quaternions[split_mask].repeat(num_splits, 1),
        "features_diffuse_color": self.features_diffuse_color[split_mask].repeat(num_splits, 1, 1),
        "features_rest": self.features_rest[split_mask].repeat(num_splits, 1, 1),
        "opacities": self.opacities[split_mask].repeat(num_splits, 1)
    }

    split_tensors = {}
    for group in self.optimizer.param_groups:
      assert (len(group["params"]) == 1)
      split_tensor = masked_tensors[group["name"]]
      stored_state = self.optimizer.state.get(group["params"][0], None)
      if stored_state is not None:
        stored_state["1st_momentum"] = torch.cat((stored_state["1st_momentum"], torch.zeros_like(split_tensor)), dim=0)
        stored_state["2nd_momentum"] = torch.cat((stored_state["2nd_momentum"], torch.zeros_like(split_tensor)), dim=0)

        del self.optimizer.state[group["params"][0]]
        self.optimizer.state[group["params"][0]] = stored_state

        group["params"][0] = nn.Parameter(torch.cat((group["params"][0], split_tensor), dim=0), requires_grad=True)
        split_tensors[group["name"]] = group["params"][0]
      else:
        group["params"][0] = nn.Parameter(torch.cat((group["params"][0], split_tensor), dim=0), requires_grad=True)
        split_tensors[group["name"]] = group["params"][0]

    self.xyz = split_tensors["xyz"]
    self.features_diffuse_color = split_tensors["features_diffuse_color"]
    self.features_rest = split_tensors["features_rest"]
    self.opacities = split_tensors["opacities"]
    self.scaling_vectors = split_tensors["scaling_vectors"]
    self.quaternions = split_tensors["quaternions"]

    self.xyz_gradients = torch.zeros((self.xyz.shape[0], 1), device=device)
    self.denominator = torch.zeros((self.xyz.shape[0], 1), device=device)
    self.max_radii = torch.zeros((self.xyz.shape[0]), device=device)

    prune_mask = torch.cat((split_mask, torch.zeros(num_splits * split_mask.sum(), device=device, dtype=bool)))
    prune_tensors = {}

    for group in self.optimizer.param_groups:
      stored_state = self.optimizer.state.get(group["params"][0], None)
      if stored_state is not None:
        stored_state["1st_momentum"] = stored_state["1st_momentum"][~prune_mask]
        stored_state["2nd_momentum"] = stored_state["2nd_momentum"][~prune_mask]

        del self.optimizer.state[group["params"][0]]
        self.optimizer.state[group["params"][0]] = stored_state

        group["params"][0] = nn.Parameter(group["params"][0][~prune_mask], requires_grad=True)
        prune_tensors[group["name"]] = group["params"][0]
      else:
        group["params"][0] = nn.Parameter(group["params"][0][~prune_mask], requires_grad=True)
        prune_tensors[group["name"]] = group["params"][0]

    self.xyz = prune_tensors["xyz"]
    self.features_diffuse_color = prune_tensors["features_diffuse_color"]
    self.opacities = prune_tensors["opacities"]
    self.scaling_vectors = prune_tensors["scaling_vectors"]
    self.quaternions = prune_tensors["quaternions"]

    self.xyz_gradients = self.xyz_gradients[~prune_mask]
    self.denominator = self.denominator[~prune_mask]
    self.max_radii = self.max_radii[~prune_mask]

  def print_model_parameters(self):
    print(f"{'Parameter Name':<30} {'Shape':<30} {'Trainable':<10}")
    print("-" * 70)
    for name, param in self.named_parameters():
        print(f"{name:<30} {str(list(param.shape)):<30} {param.requires_grad}")

  def save_ply(self, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    xyz = self.xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    features_diffuse_color = self.features_diffuse_color.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    features_rest = self.features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = self.opacities.detach().cpu().numpy()
    scaling_vectors = self.scaling_vectors.detach().cpu().numpy()
    quaternions = self.quaternions.detach().cpu().numpy()

    attribute_list = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    for i in range(self.features_diffuse_color[1] * self.features_diffuse_color.shape[2]):
      attribute_list.append('feature_diffuse_color_{}'.format(i))
    for i in range(self.features_rest.shape[1] * self.features_rest.shape[2]):
      attribute_list.append('feature_rest_{}'.format(i))
    attribute_list.append('opacity')
    for i in range(self.scaling_vectors.shape[1]):
      attribute_list.append('scale_{}'.format(i))
    for i in range(self.quaternions.shape[1]):
      attribute_list.append('rotation_{}'.format(i))

    dtype_full = [(attribute, 'f4') for attribute in attribute_list]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, features_diffuse_color, features_rest, opacities, scaling_vectors, quaternions), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

  def add_densification_stats(self, viewspace_point_tensor, update_filter):
    self.xyz_gradients[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
    self.denominator[update_filter] += 1

  def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
    grads = self.xyz_gradients / self.denominator
    grads[grads.isnan()] = 0.

    self.temp_radii = radii
    self.clone_gaussians(grads, max_grad, extent)
    self.split_gaussians(grads, max_grad, extent)

    prune_mask = (self.opacities < min_opacity).squeeze()
    if max_screen_size:
      big_points_vs = self.max_radii > max_screen_size
      big_points_ws = self.scaling_vectors.max(dim=1).values > 0.1 * extent
      prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
    self.prune_points(prune_mask)
    temp_radii = self.temp_radii
    self.temp_radii = None

    torch.cuda.empty_cache()

  def reset_opacity(self):
    new_opacities = inverse_sigmoid(torch.min(self.opacities, torch.ones_like(self.opacities) * 0.01))

    optimizable_tensors = {}
    for group in self.optimizer.param_groups:
      stored_state = self.optimizer.state.get(group['params'][0], None)
      if stored_state is None:
        group["params"][0] = nn.Parameter(group["params"][0][remove_mask], requires_grad=True)
        optimizable_tensors[group["name"]] = group["params"][0]
      else:
        stored_state["1st_momentum"] = stored_state["1st_momentum"][remove_mask]
        stored_state["2nd_momentum"] = stored_state["2nd_momentum"][remove_mask]

        del self.optimizer.state[group["params"][0]]
        group["params"][0] = nn.Parameter(group["params"][0][remove_mask], requires_grad=True)
        self.optimizer.state[group["params"][0]] = stored_state

        optimizable_tensors[group["name"]] = group["params"][0]

    self.opacities = optimizable_tensors["opacity"]

  def capture(self):
    return (self.xyz, self.features_diffuse_color, self.scaling_vectors,
            self.quaternions, self.opacities, self.max_radii, self.xyz_gradients,
            self.denominator, self.optimizer.state_dict())

  def forward(self, x:torch.Tensor):
    return self.xyz, self.features_diffuse_color, self.features_rest, self.quaternions, self.scaling_vectors, self.opacities
