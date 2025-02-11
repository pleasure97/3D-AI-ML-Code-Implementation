import sys
import math
import numpy as np
from typing import NamedTuple
import json
from tqdm import tqdm
from random import randint
import os

class SceneInfo(NamedTuple):
  points: np.array,
  colors: np.array,
  normals: np.array,
  cameras: list,
  nerf_normalization: dict,
  ply_path: str,
  is_nerf_synthetic: bool

def prepare_tensorboard_summary(output_path="output"):
    # Set up output folder
    print("Output folder: {}".format(output_path))
    os.makedirs(output_path, exist_ok = True)

    try:
      from torch.utils.tensorboard import SummaryWriter
    except:
      print("[INFO] Couldn't find tensorboard... installing it.")
      !pip install -q tensorboard
      from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(output_path)

    return writer

def train_report(tensorboard_writer,
                 iteration, L1Loss, loss, l1_loss, elapsed,
                 train_cameras, renderFunc, model, renderArgs, device=device):

  tensorboard_writer.add_scalar('train_loss/l1_loss', L1Loss.item(), iteration)
  tensorboard_writer.add_scalar('train_loss/total_loss', loss.item(), iteration)
  tensorboard_writer.add_scalar('iteration_time', elapsed, iteration)

  torch.cuda.empty_cache()
  validation_config = ({'name': 'train', 'cameras': [train_cameras[idx % len(train_cameras)]] for idx in range(5, 30, 5)})

  if validation_config['cameras'] and len(config['cameras']) > 0:
    l1_test = 0.
    psnr_test = 0.
    for idx, viewpoint in enumerate(validation_config['cameras']):
      image = torch.clamp(renderFunc(viewpoint, model, *renderArgs)["render"], 0., 1.)
      ground_truth_image = torch.clamp(viewpoint.original_image.to(device))
      if (tensorboard_writer and (idx < 5)):
        tensorboard_writer.add_images(validation_config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
      l1_test += l1_loss(image, ground_truth_image).mean().double()
      psnr_test += psnr(image, ground_truth_image).mean().double()
    psnr_test /= len(validation_config['cameras'])
    l1_test /= len(validation_config['cameras'])
    print("\n[ITER {}] Evaluating {} : L1 {} PSNR {}".format(iteration, validation_config['name'], l1_test, psnr_test))

    tensorboard_writer.add_scalar(validation_config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
    tensorboard_writer.add_scalar(validation_config['name'] + '/loss_viewpoint - psnr',psnr_test, iteration)

  tensorboard_writer.add_histogram("opacity_histogram", model.opacities, iteration)
  tensorboard_writer.add_scalar("total_points", model.xyz.shape[0], iteration)

  torch.cuda.empty_cache()

def train_step(colmap_path: str="COLMAP",
               model_path: str="models",
               checkpoint: str=None,
               resolution_scales=[1.0],
               white_background=False):
    first_iter = 0
    tensorboard_writer = prepare_tensoboard_summary()

    # model
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir()]

    train_cameras = {}
    test_cameras = {}

    cameras_extrinsic_file = os.path.join(colmap_path, "sparse/0", "images.bin")
    cameras_intrinsic_file = os.path.join(colmap_path, "sparse/0", "cameras.bin")
    cameras_extrinsics = read_extrinsics_bin(cameras_extrinsic_file)
    cameras_intrinsics = read_intrinsics_bin(cameras_intrinsic_file)

    depth_params_file = os.path.join(colmap_path, "sparse/0", "depth_params.json")
    depth_params = None

    test_cam_names_list = []
    # reading_dir = "images"
    camera_infos_unsorted = read_colmap_cameras(camera_extrinsics=cameras_extrinsics,
                                                camera_intrinsics=cameras_intrinsics,
                                                depths_params=depth_params,
                                                images_folder=os.path.join(path, "images"),
                                                depths_folder=os.path.join(path, ""),
                                                test_camera_names_list=[])
    camera_infos = sorted(camera_infos_unsorted.copy(), key = lambda x : x["image_name"])

    nerf_normalization = get_nerf_ppNorm(camera_infos)

    ply_path = os.path.join(colmap_path, "sparse/0/points3D.ply")
    bin_path = os.path.join(colmap_path, "sparse/0/points3D.bin")
    text_path = os.path.join(colmap_path, "sparse/0/points3D.txt")

    if not os.path.exists(ply_path):
      print("Converting points3d.bin to .ply...")
      xyz, rgb, _ = read_points3D_bin(bin_path)
      store_ply(ply_path, xyz, rgb)
    try:
      points, colors, normals = fetch_ply(ply_path)
    except:
      points, colors, normals = None, None, None

    scene_info = SceneInfo(points=points,
                           colors=colors,
                           normals=normals,
                           cameras=camera_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=True)

    if not checkpoint:
      with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(model_path, "input.ply"), 'wb') as dest_file:
        dest_file.write(src_file.read())
      json_cameras = []
      camera_list = []
      if scene_info.cameras:
        camera_list.extend(scene_info.cameras)
      for id, camera in enumerate(camera_list):
        json_cameras.append(camera_to_json(id, camera))
      with open(os.path.join(model_path, "cameras.json"), "w") as f:
        json.dump(json_cameras, f)

    cameras_extent = scene_info.nerf_normalization["radius"]

    for resolution_scale in resolution_scales:
      print("Loading Cameras...")
      train_cameras[resolution_scale] = camera_list_from_infos

    model = GaussianSplattingModelV2(points, colors)

    model.initialize_from_gaussians(points, colors)

    background_color = [1, 1, 1] if white_background else [0, 0, 0]
    background = torch.tensor(background_color, device=device)

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    ema_loss_for_log = 0.
    # train_cameras[1] means get a train camera which scale is equal to 1.
    viewpoint_stack = train_cameras[1].copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))

    iterations = 7_000
    progress_bar = tqdm(range(first_iter, iterations), desc="Training Progress")
    first_iter += 1
    for iteration in range(first_iter, iterations + 1):
      iter_start.record()
      model.update_learning_rate(iteration)

      random_idx = randint(0, len(viewpoint_indices))
      viewpoint_camera = viewpoint_stack.pop(random_idx)
      viewpoint_idx = viewpoint_indices.pop(random_idx)

      render_output = render(viewpoint_camera, model, background)
      image, viewpoint_tensor, visibility_filter, radii = \
        render_output["render"], render_output["viewspace_points"], render_output["visibility_filter"], render_output["radii"]
      ground_truth_image = viewpoint_camera.original_image.to(device)

      LAMBDA = 0.2
      L1Loss = L1_Loss(image, ground_truth_image)
      loss = (1. - LAMBDA) * L1Loss + LAMBDA * (1. - ssim(image, ground_truth_image))
      loss.backward()

      iter_end.record()

      with torch.no_grad():
        ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
        if iteration % 10 == 0:
          progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
          progress_bar.update(10)
        if iteration == iterations:
          progress_bar.close()

        train_report(tensorboard_writer, iteration, L1Loss, loss, L1_Loss, iter_start.elapsed_time(), train_cameras, render, model, (background))
        if (iteration == 7_000):
          print("\n[ITER {}] Saving Gaussians.".format(iteration))
          point_cloud_path = os.path.join(model_path, "point_cloud/iterations_{}".format(iteration))
          model.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

        # Dennsification
        # 7000 / 2 = densify until iteration
        if iteration < 7_000 / 2:
          model.max_radii[visibility_filter] = torch.max(model.max_radii[visibility_filter], radii[visibility_filter])
          model.add_densification_stats(viewpoint_tensor, visibility_filter)
          # 500 = densify from iteration, 100 = densification interval
          if iteration > 500 and iteration % 100 == 0:
            # 3000 = opacity reset interval
            size_threshold = 20 if iteration > 3_000 else None
            model.densify_and_prune(2e-4, 5e-3, scene_info.nerf_normalization["radius"], size_threshold)

          if iteration % 3_000 = 0 or (white_background and iteration == 500):
            model.reset_opacity()

        if iteration < 7_000:
          model.optimizer.step()
          model.optimizer.zero_grad(set_to_none=True)

        if iteration in [1000 * i for i in range(1, 6)]:
          print("\n[ITER {}] Saving Checkpoint".format(iteration))
          torch.save((model.capture(), iteration), model_path + "/checkpoint" + str(iteration) + ".pth")
