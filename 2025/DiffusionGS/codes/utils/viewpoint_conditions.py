from pathlib import Path
import pycolmap
import os 
import shutil
import numpy as np
from scipy.spatial.transform import Rotation as Rot

def make_SfM_points(image_path: Path):
  """
  Args:
    image_path : pathlib.Path - Path containing Images to be converted using SfM
  """
  top_path = Path("COLMAP/")
  relative_path = image_path.relative_to(image_path.parents[0])
  output_path = top_path / relative_path

  output_path.mkdir(parents=True, exist_ok=True)

  database_path = output_path / "database.db"
  sfm_path = output_path / "sfm"

  if database_path.exists():
    database_path.unlink()

  pycolmap.extract_features(database_path, image_path)
  pycolmap.match_exhaustive(database_path)

  num_images = pycolmap.Database(database_path).num_images

  if sfm_path.exists():
    shutil.rmtree(sfm_path)
  sfm_path.mkdir(exist_ok=True)

  records = pycolmap.incremental_mapping(database_path, image_path, sfm_path)

  return sfm_path
def convert_cameras_bin(cameras_path: Path):
    if not os.path.exists(cameras_path):
        raise Exception(f"No such file : {cameras_path}")

    with open(cameras_path, 'r') as f:
        lines = f.readlines()

    if len(lines) < 3:
        raise Exception(f"Invalid cameras.txt file : {cameras_path}")

    comments = lines[:3]
    contents = lines[3:]

    ids = []
    Ks = []


    for cam_idx, content in enumerate(contents):
        content_items = content.split(' ')
        cam_id = content_items[0]
        cam_type = content_items[1]
        img_w, img_h = int(content_items[2]), int(content_items[3])

        if cam_type == "OPENCV":
            fx, fy = content_items[4], content_items[5]
            cx, cy = content_items[6], content_items[7]
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            dist = content_items[8:] + [0] # k1 k2 p1 p2 + k3(0)
            dist = np.asarray(dist)
            ids.append(cam_id)
            Ks.append(K)
        elif cam_type== "PINHOLE":
            fx, fy = content_items[4], content_items[5]
            cx, cy = content_items[6], content_items[7]
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            dist = np.zeros([5], dtype=np.float32)
            ids.append(cam_id)
            Ks.append(K)
        else:
            raise NotImplementedError(f"Only opencv/pinhole camera will be supported.")

    return ids, Ks

def convert_images_bin(images_path: Path):
    if not os.path.exists(images_path):
        raise Exception(f"No such file : {images_path}")

    with open(images_path, 'r') as f:
        lines = f.readlines()

    if len(lines) < 2:
        raise Exception(f"Invalid cameras.txt file : {images_path}")

    comments = lines[:4]
    contents = lines[4:]

    image_ids = []
    camera_ids = []
    image_names = []
    poses = []
    for img_idx, content in enumerate(contents[::2]):
        content_items = content.split(' ')
        img_id = content_items[0]
        q_xyzw = np.array(content_items[2:5] + content_items[1:2], dtype=np.float32) # colmap uses wxyz
        t_xyz = np.array(content_items[5:8], dtype=np.float32)
        cam_id = content_items[8]
        img_name = content_items[9]

        R = Rot.from_quat(q_xyzw).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, -1] = t_xyz

        image_ids.append(img_id)
        camera_ids.append(cam_id)
        image_names.append(img_name)
        poses.append(T)

    return image_ids, camera_ids, image_names, poses
def pixel_to_ray_direction(u, v, camera_intrinsic, rotation_matrix):
  pixel_homogeneous = np.array([u, v, 1.]) # homogenous coordinate
  camera_ray = np.linalg.inv(camera_intrinsic) @ pixel_homogeneous # convert to camera coordinate system
  world_ray = rotation_matrix.T @ camera_ray # convert to world coordinate system

  return world_ray / np.linalg.norm(world_ray) # normalized ray direction

def get_viewpoint_conditions(sfm_path: str, height: int=800, width: int=400) -> dict:
  """
  Extract viewpoint conditions from SfM Results.

  Args:
    sfm_path (str): saved path of COLMAP's incremental_mapping result
  """
  sfm_path = Path(sfm_path)
  images_path = sfm_path / "images.bin"
  cameras_path = sfm_path / "cameras.bin"

  viewpoint_conditions = {}

  image_ids, camera_ids, image_names, rotations, translations = convert_images_bin(images_path)
  camera_ids, camera_intrinsics, _, _ = convert_cameras_bin(cameras_path)

  for image_id, rotation_matrix, translation_vector in zip(image_ids, rotations, translations):
    # Get camera position
    camera_position = -rotation_matrix.T @ translation_vector
    print(f"Camera Position of Image {image_id} : {camera_position}")

    # Load camera intrinsics
    camera_index = camera_ids[image_id]
    K = camera_intrinsics[camera_index]

    viewpoint_condition = np.zeros((height, width, 6))

    for i in range(height):
      for j in range(width):
        ray_dir = pixel_to_ray_direction(j, i, K, rotation_matrix)
        viewpoint_condition[i, j, :3] = camera_position
        viewpoint_condition[i, j, 3:] = ray_dir

    print("Viewpoint Condition's shape : ", viewpoint_condition.shape)

    viewpoint_conditions[image_id] = viewpoint_condition

  return viewpoint_conditions
