import torch
import torch.nn.functional as F


def get_rays(height, width, intrinsics, c2w, jitter=False):
    """
      height : image height
      width : image width
      intrinsics : 4 by 4 intrinsic matrix
      c2w : 4 by 4 camera to world extrinsic matrix
    """
    u, v = torch.meshgrid(torch.arange(width, device=c2w.device), torch.arange(height, device=c2w.device), indexing="ij")
    B = c2w.shape[0]
    u, v = u.reshape(-1), v.reshape(-1)
    u_noise = v_noise = 0.5
    if jitter:
        u_noise = torch.rand(u.shape, device=c2w.device)
        v_noise = torch.rand(v.shape, device=c2w.device)
    u, v = u + u_noise, v + v_noise  # add half pixel
    pixels = torch.stack((u, v, torch.ones_like(u)), dim=0)  # (3, H * W)
    pixels = pixels.unsqueeze(0).repeat(B, 1, 1)  # (B, 3 , H * W)
    if intrinsics.sum() == 0:
        inv_intrinsics = torch.eye(3, device=c2w.device).tile(B, 1, 1)
    else:
        inv_intrinsics = torch.linalg.inv(intrinsics)
    rays_d = inv_intrinsics @ pixels  # (B, 3, H * W)
    rays_d = c2w[:, :3, :3] @ rays_d
    rays_d = rays_d.transpose(-1, -2)  # (B, H * W, 3)
    rays_d = F.normalize(rays_d, dim=-1)

    rays_o = c2w[:, :3, 3].reshape((-1, 3))  # (B, 3)
    rays_o = rays_o.unsqueeze(1).repeat(1, height * width, 1)  # (B, H * W, 3)

    return rays_o, rays_d


def plucker_embedding(height, width, intrinsics, c2w, jitter=False):
    """Computes the plucker coordinates from batched cam2world & intrinsics matrices, as well as pixel coordinates
  C2W : (Batch Size, 4, 4)
  intrinsics : (Batch Size, 3, 3)
  """
    rays_o, rays_d = get_rays(height, width, intrinsics, c2w, jitter=jitter)  # (B, H * W, 3), (B, H * W, 3)
    cross = torch.cross(rays_o, rays_d, dim=-1)
    plucker = torch.cat((rays_d, cross), dim=1)

    plucker = plucker.view(-1, height, width, 6).permute(0, 3, 1, 2)
    return plucker  # (B, 6, H, W, )


def reference_point_plucker_embedding(height, width, intrinsics, c2w, jitter=False):
    """Computes the reference point plucker coordinates from batched cam2world & intrinsics matrices, as well as pixel coordinates
  H : image height
  W : image width
  C2W : (Batch Size, 4, 4)
  intrinsics : (Batch Size, 3, 3)
  """
    rays_o, rays_d = get_rays(height, width, intrinsics, c2w, jitter=jitter)  # (B, H * W, 3), (B, H * W, 3)
    o_dot_d = (rays_o * rays_d).sum(dim=-1, keepdim=True)  # (B, H * W , 1)
    reference_point = rays_o - o_dot_d * rays_d  # (B, H * W, 3)
    reference_point_plucker = torch.cat((rays_d, reference_point), dim=1)

    reference_point_plucker = reference_point_plucker.view(-1, height, width, 6).permute(0, 3, 1, 2)
    return reference_point_plucker  # (B, 6, H, W)
