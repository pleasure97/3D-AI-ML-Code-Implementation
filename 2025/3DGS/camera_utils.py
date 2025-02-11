import struct
import numpy as np

def read_next_bytes(file, num_bytes, format_char_sequence, endian_character="<"):
    data = file.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_extrinsic_bin(bin_path: str):
    images = {}
    with open(bin_path, "rb") as f:
      num_images = read_next_bytes(f, 8, "Q")[0]
      for _ in range(num_images):
        image_properties = read_next_bytes(f, 64, "idddddddi")
        image_id = binary_image_properties[0]
        quarternion_vector = np.array(image_properties[1:5])
        translation_vector = np.array(image_properties[5:8])
        camera_id = image_properties[8]
        image_name = ""
        current_char = read_next_bytes(f, 1, "c")[0]
        while current_char != b"\x00":
          image_name += current_char.decode("utf-8")
          current_char = read_next_bytes(f, 1, "c")[0]
        num_points2D = read_next_bytes(f, 8, "Q")[0]
        xy_ids = read_next_bytes(f, 24 * num_points2D, "ddq" * num_points2D)
        xys = np.array(xy_ids).reshape(-1, 3)[:, :2]
        point3D_IDs = np.array(xy_ids[2::3], dtype=int)

        images[image_id] = {"id": image_id,
                            "qvec": quarternion_vector,
                            "tvec": translation_vector,
                            "camera_id": camera_id,
                            "name": image_name,
                            "xys": xys,
                            "point3D_IDs": point3D_IDs}
    return images

def read_intrinsic_bin(bin_path: str):
    cameras = {}
    with open(bin_path, "rb") as f:
      num_cameras = read_next_bytes(f, 8, "Q")[0]
      for _ in range(num_cameras):
        camera_properties = read_next_bytes(f, 24, "iiQQ")
        camera_id = camera_properties[0]
        model_id = camera_properties[1]
        model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
        width = camera_properties[2]
        height = camera_properties[3]
        num_params = CAMERA_MODEL_IDS[model_id].num_params
        params = read_next_bytes(f, 8 * num_params, "d" * num_params)
        cameras[camera_id] = {"id": camera_id,
                              "model": model_name,
                              "width": width,
                              "height": height,
                              "params": np.array(params)}
      assert len(cameras) == num_cameras
    return cameras

def read_points3D_bin(bin_path: str):
  with open(bin_path, "rb") as f:
    num_points = read_next_bytes(f, 8, "Q")[0]

    xyzs = np.empty((num_points, 3))
    rgbs = np.empty((num_points, 3))
    errors = np.empty((num_points, 3))

    for point_id in num_points:
      binary_point_line_properties = read_next_bytes(f, num_bytes=43, format_char_sequence="QdddBBBd")
      xyz = np.array(binary_point_line_properties[1:4])
      rgb = np.array(binary_point_line_properties[4:7])
      error = np.array(binary_point_line_properties[7])
      track_length = read_next_bytes(f, num_bytes=8, format_char_sequence="Q")[0]
      track_elements = read_next_bytes(f, num_bytes=8 * track_length, format_char_sequence="ii" * track_length)
      xyzs[point_id] = xyz
      rgbs[point_id] = rgb
      errors[point_id] = error

  return xyzs, rgbs, errors
from plyfile import PlyData, PlyElement
from PIL import Image
import cv2
import math

def quaternions_to_rotations(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def focal_to_fov(focal, pixels):
  return 2 * math.atan(pixels / (2 * focal))

def fov_to_focal(fov, pixels):
  return pixels / (2 * math.tan(fov / 2))

def read_colmap_cameras(camera_extrinsics, camera_intrinsics, depths_params, images_folder, depths_folder, test_camera_names_list):
    camera_infos = []
    for index, key in enumerate(camera_extrinsics):
      sys.stdout.write('\r')
      sys.stdout.write("Reading Camera {}/{}".format(index + 1, len(camera_extrinsics)))
      sys.stdout.flush()

      extrinsic = camera_extrinsics[key]
      intrinsic = cmaera_intrinsics[extrinsic.camera_id]
      height = intrinsic.height
      width = intrinsic.width
      uid = intrinsic.id

      R = np.transpose(quaternions_to_rotations(extrinsic.qvec))
      T = np.array(extrinsic.tvec)

      if intrinsic.model == "SIMPLE_PINHOLE":
        focal_length_x = intrinsic.params[0]
        FovX = focal_to_fov(focal_length_x, width)
        FovY = focal_to_fov(focal_length_x, height)
      elif intrinsic.model == "PINHOLE":
        focal_length_x = intrinsic.params[0]
        focal_length_y = intrinsic.params[0]
        FovX = focal_to_fov(focal_length_x, width)
        FovY = focal_to_fov(focal_length_y, height)
      num_remove = len(extrinsic.name.split('.')[-1]) + 1
      depth_params = None

      image_path = os.path.join(images_folder,extrinsic.name)
      image_name = extrinsic.name
      depth_path = ""

      camera_info = {
          "uid": uid, "R": R, "T": T, "FovX": FovX, "FovY": FovY,
          "depth_path": depth_path, "depth_params": depth_params,
          "image_path": image_path, "image_name": image_name,
           "width": width, "height": height,
          "is_test": image_name in test_camera_names_list
          }
      camera_infos.append(camera_info)

    sys.stdout.write('\n')

    return camera_infos

def get_center_and_diag(camera_centers):
  camera_centers = np.hstack(camera_centers)
  average_camera_center = np.mean(camera_centers, axis=1, keepdims=True)
  distance = np.linalg_norm(camera_centers - average_camera_center, axis=0, keepdims=True)
  diagonal = np.max(distance)
  return average_camera_center.flatten(), diagonal

def get_world_to_view_to(rotation_matrix, translation_vector, translate=np.array([0., 0., 0.]), scale=1.0):
  rotation_with_translation = np.zeros((4, 4))
  rotation_with_translation[:3, :3] = rotation_matrix.transpose()
  rotation_with_translation[:3, 3] = translation_vector
  rotation_with_translation[3, 3] = 1.0

  camera_to_world = np.linalg.inv(rotation_with_translation)
  camera_center = camera_to_world[:3, 3]
  camera_center =  (camera_center + translate) * scale
  camera_to_world[:3, 3] = camera_center
  rotation_with_translation = np.linalg.inv(camera_to_world)

  return np.float32(rotation_with_translation)

def get_projection_matrix(z_near, z_far, FovX, FovY, device=device):
  tan_half_FovX = math.tan((FovX / 2))
  tan_half_FovY = math.tan((FovY / 2))

  top = tan_half_FovY * z_near
  bottom = -top
  right = tan_half_FovX * z_near
  left = -right

  P = torch.zeros(4, 4, device=device)

  z_sign = 1.

  P[0, 0] = 2. * z_near / (right - left)
  P[1, 1] = 2. * z_near / (top - bottom)
  P[0, 2] = (right + left) / (right - left)
  P[1, 2] = (top + bottom ) / (top - bottom)
  P[2, 2] =  z_sign * z_far / (z_far - z_near)
  P[2, 3] = - (z_far * z_near) / (z_far - z_near)
  P[3, 2] = z_sign

  return P

def get_NeRF_ppNorm(camera_info):

  camera_centers = []

  for camera in camera_info:
    WorldToCamera = get_world_to_view_to(camera["R"], camera["T"])
    CameraToWorld = np.linalg.inv(WorldToCamera)
    camera_centers.append(CameraToWorld[:3, 3:4])

  center, diagonal = get_center_and_diag(camera_centers)
  radius = diagonal * 1.1

  translate = -center

  return {"translate": translate, "radius": radius}

def store_ply(path, xyz, rgb):
  # Define the data type for the structured array
  data_type = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
               ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
               ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
  normals = np.zeros_like(xyz)
  elements = np.empty(xyz.shape[0], dtype=data_type)
  attributes = np.concatenate((xyz, normals, rgb), axis=1)
  elements[:] = list(map(tuple, attributes))

  vertex_element = PlyElement.describe(elements, 'vertex')
  ply_data = PlyData([vertex_element])
  ply_data.write(path)

def fetch_ply(path):
  plyData = PlyData.read(path)
  vertices = plyData['vertex']
  positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
  colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
  normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T

  return points, colors, normals

def camera_to_json(id, camera):
  rotation_with_translation = np.zeros((4, 4))
  rotation_with_translation[:3, :3] = camera["R"].transpose()
  rotation_with_translation[:3, 3] = camera["T"]
  rotation_with_translation[3, 3] =1.0

  world_to_camera = np.linalg.inv(rotation_with_translation)
  position = world_to_camera[:3, 3]
  rotation = world_to_camera[:3, :3]
  serializable_array_2d = [x.tolist() for x in rotation]
  camera_entry = {
      "id": id,
      "image_name": camera["image_name"],
      "width": camera["width"],
      "height": camera["height"],
      "position": position.tolist(),
      "rotation": serializable_array_2d,
      "fx": fov_to_focal(camera["FovX"], camera.width),
      "fy": fov_to_focal(camera["FovY"], camera.height)
  }

  return camera_entry

def load_camera(resolution, id, camera_info, resolution_scale):
  image = Image.open(camera_info.image_path)

  if camera_info.depth_path != "":
    inv_depth_map = cv2.imread(camera_info.depth_path, -1).astype(np.float32) / 512
  else:
    inv_depth_map = None

  original_width, original_height = image.size
  if resolution in [1, 2, 4, 8]:
    resolution = round(original_width / (resolution_scale * resolution)), round(original_height / (resolution_scale * resolution))
  else:
    if resolution == -1:
      if original_width > 1600:
        global_down = original_width / 1600
      else:
        global_down = 1
    else:
      global_down = original_width / resolution

    scale = float(global_down) * float(resolution_scale)
    resolution = (int(original_width / scale), int(original_height / scale))

  return {"resolution": resolution, "colmap_id": camera_info.uid, "R": camera_info.R, "T": camera_info.T,
          "FovX": camera_info.FovX, "FovY": camera_info.FovY, "depth_params": camera_info.depth_params,
          "image": image, "inv_depth_map": inv_depth_map, "image_name": camera_info.image_name, "uid": id}

def camera_list_from_infos(camera_infos, resolution_scale, resolution = 1):
  camera_list = []

  for id, camera_info in camera_infos:
    camera_list.append(load_camera(resolution, id, camera_info, resolution_scale))

  return camera_list
