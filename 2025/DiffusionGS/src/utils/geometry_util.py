import os

import torch
from jaxtyping import Float
from torch import Tensor
import torch.nn.functional as F
from pathlib import Path
import pycolmap
import shutil
import struct
import numpy as np


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


def quaternion_to_rotation_matrix(quaternion):
    """Convert quaternion to a 3x3 rotation matrix"""
    w, x, y, z = quaternion
    return np.array([
        [1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x ** 2 - 2 * y ** 2]
    ])


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

def inspect_cameras_bin(path, n_bytes=64):
    path = Path(path)
    data = path.read_bytes()[:n_bytes]
    print(f"File: {path}")
    print(f"Read {len(data)} bytes:")
    print(data.hex(" "))
    if len(data) >= 8:
        print("First Q (num_cameras):", struct.unpack("<Q", data[:8])[0])

def read_next_bytes(fid,
                    format_char_sequence: str,
                    endian_character: str = "<"):
    """ Read and unpack the next bytes from a binary file. """
    byte_format = endian_character + format_char_sequence
    byte_size = struct.calcsize(byte_format)
    data = fid.read(byte_size)
    if len(data) != byte_size:
        raise IOError(f"Expected {byte_size} bytes, got {len(data)} bytes")
    return struct.unpack(byte_format, data)


def convert_cameras_bin(cameras_path: str | Path):
    """ Read cameras.bin and return camera intrinsics """
    cameras = {}

    with open(cameras_path, "rb") as fid:
        num_cameras = read_next_bytes(fid, "Q")[0]
        for _ in range(num_cameras):
            camera_id, model_id = read_next_bytes(fid, "ii")
            width, height = read_next_bytes(fid, "QQ")

            if model_id == 1:  # Simple Pinhole
                focal_length, center_x, center_y = read_next_bytes(fid, "ddd")
                intrinsics = np.array([
                    [focal_length, 0, center_x],
                    [0, focal_length, center_y],
                    [0, 0, 1]])
            elif model_id == 2:  # Pinhole
                focal_length_x, focal_length_y, center_x, center_y = read_next_bytes(fid, "dddd")
                intrinsics = np.array([
                    [focal_length_x, 0, center_x],
                    [0, focal_length_y, center_y],
                    [0, 0, 1]])
            else:
                raise ValueError(f"Unsupported Camera Model {model_id}")
            cameras[camera_id] = (intrinsics, width, height)
    return cameras


def convert_images_bin(images_path: str | Path):
    """ Read images.bin and return image IDs, camera IDs, extrinsics,and image names """
    images = {}

    with open(images_path, "rb") as fid:
        num_images = read_next_bytes(fid, "Q")[0]
        for _ in range(num_images):
            # q - quaternion / t - translation
            image_id, q_w, q_x, q_y, q_z, t_x, t_y, t_z, camera_id, name_len = read_next_bytes(fid, "i ddddddd i i")
            name_bytes = bytearray()
            while True:
                b = fid.read(1)
                if not b or b == b"\x00":
                    break
                name_bytes += b
            try:
                name = name_bytes.decode("utf-8")
            except UnicodeDecodeError:
                name = name_bytes.decode("latin-1")
            # Each point consists of x (double), y (double), point3d_id (uint64)
            num_points2D = read_next_bytes(fid, "Q")[0]
            fid.seek(num_points2D * (8 + 8 + 8), os.SEEK_CUR)

            # Convert Quaternion to Rotation Matrix
            q = np.array([q_w, q_x, q_y, q_z])
            R = quaternion_to_rotation_matrix(q)
            t = np.array([t_x, t_y, t_z])

            images[image_id] = (camera_id, R, t, name)

    return images


def make_rotation_matrix(quaternion):
    norm = torch.sqrt(quaternion[:, 0] ** 2 + quaternion[:, 1] ** 2 + quaternion[:, 2] ** 2)

    quaternion = quaternion / norm[:, None]

    rotation_matrix = torch.zeros((quaternion.size(0), 3, 3))

    r = quaternion[:, 0]
    x = quaternion[:, 1]
    y = quaternion[:, 2]
    z = quaternion[:, 3]

    rotation_matrix[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rotation_matrix[:, 0, 1] = 2 * (x * y - r * z)
    rotation_matrix[:, 0, 2] = 2 * (x * z + r * y)
    rotation_matrix[:, 1, 0] = 2 * (x * y + r * z)
    rotation_matrix[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rotation_matrix[:, 1, 2] = 2 * (y * z - r * x)
    rotation_matrix[:, 2, 0] = 2 * (x * z - r * y)
    rotation_matrix[:, 2, 1] = 2 * (y * z + r * x)
    rotation_matrix[:, 2, 2] = 1 - 2 * (x * x + y * y)

    return rotation_matrix


def multiply_scaling_rotation(scale, quaternion):
    scaling_matrix = torch.zeros((scale.shape[0], 3, 3), dtype=torch.float, device="cuda")
    rotation_matrix = make_rotation_matrix(quaternion)

    scaling_matrix[:, 0, 0] = scale[:, 0]
    scaling_matrix[:, 1, 1] = scale[:, 1]
    scaling_matrix[:, 2, 2] = scale[:, 2]

    multiplied_matrix = rotation_matrix @ scaling_matrix

    return multiplied_matrix

def make_projection_matrix(near, far, tan_fov_x, tan_fov_y):
    top = tan_fov_x * near
    bottom = -top
    right = tan_fov_y * near
    left = -right

    projection_matrix = torch.zeros(4, 4)

    sign = 1.

    projection_matrix[0, 0] = 2. * near / (right - left)
    projection_matrix[1, 1] = 2. * near / (top - bottom)
    projection_matrix[0, 2] = (right + left) / (right - left)
    projection_matrix[1, 2] = (top + bottom) / (top - bottom)
    projection_matrix[3, 2] = sign
    projection_matrix[2, 2] = sign * far / (far - near)
    projection_matrix[2, 3] = -(far * near) / (far - near)

    return projection_matrix
