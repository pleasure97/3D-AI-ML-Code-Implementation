from dataclasses import dataclass
from src.dataset.dataset_common import DatasetConfig
from typing import Literal
from torch.utils.data import IterableDataset
import torch
from torchvision.transforms import ToTensor
import os
import glob
from PIL import Image
from src.dataset.types import Stage
from src.model.denoiser.viewpoint.view_sampler import ViewSampler
from src.utils.geometry_util import convert_cameras_bin, convert_images_bin, make_rotation_matrix
from src.dataset.preprocessing.preprocess_utils import crop_example

@dataclass
class DatasetMVImgNetConfig(DatasetConfig):
    name: Literal["MVImgNet"]
    root: str
    max_fov: float
    image_shape: tuple

class DatasetMVImgNet(IterableDataset):
    config: DatasetMVImgNetConfig
    stage: Stage
    view_sampler: ViewSampler
    u_near: float = 0.1
    u_far: float = 4.2

    def __init__(self, config: DatasetMVImgNetConfig, stage: Stage) -> None:
        super().__init__()
        self.config = config
        self.stage = stage
        self.view_sample = ViewSampler

        scenes = os.path.join(self.config.root, "*", "*")
        self.scenes = sorted(glob.glob(scenes))

        if self.stage in ("train", "val"):
            self.scenes = [self.scenes[i] for i in torch.randperm(len(self.scenes))]

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:
            num_workers, worker_id = worker_info.num_workers, worker_info.id
            self.scenes = (scene for i, scene in enumerate(self.scenes) if i % num_workers == worker_id)

        for scene in self.scenes:
            # Read COLMAP Binaries
            sparse_directory = os.path.join(scene, "sparse", "0")
            cameras = convert_cameras_bin(os.path.join(sparse_directory, "cameras.bin"))
            images = convert_images_bin(os.path.join(sparse_directory, "images.bin"))

            view_ids = sorted(images.keys())
            num_views = len(view_ids)

            images = []
            extrinsics = []
            intrinsics = []

            for view_id in view_ids:
                # Extrinsic
                quaternion, translation, camera_id, name, _ = images[view_id]
                rotation_matrix = make_rotation_matrix(quaternion)
                # TODO - Check Device
                translation_vector = torch.tensor(translation, dtype=torch.float32)
                extrinsic = torch.eye(4, dtype=torch.float32)
                extrinsic[:3, :3], extrinsic[:3, 3] = rotation_matrix, translation_vector
                extrinsics.append(extrinsic)

                # Intrinsic
                model, width, height, parameters = cameras[camera_id]
                focal_x, focal_y, center_x, center_y = parameters[0], parameters[0], parameters[1], parameters[2]
                # TODO - Check Device
                intrinsic = torch.eye(3, dtype=torch.float32)
                intrinsic[0, 0], intrinsic[0, 1] = focal_x, focal_y
                intrinsic[0, 2], intrinsic[1, 2] = center_x, center_y
                intrinsics.append(intrinsic)

                image_path = os.path.join(scene, "images", name)
                pil = Image.open(image_path).convert("RGB")
                images.append(ToTensor()(pil))

            extrinsics = torch.stack(extrinsics, dim=0)
            intrinsics = torch.stack(intrinsics, dim=0)
            images = torch.stack(images, dim=0)

            source_indices, target_indices = self.view_sampler.sample(extrinsics)

            example = {
                "source": {
                    "extrinsics": extrinsics[source_indices],
                    "intrinsics": intrinsics[source_indices],
                    "image": images[source_indices],
                    "near": self.u_near,
                    "far": self.u_far,
                    "indices": source_indices
                },
                "target": {
                    "extrinsics": extrinsics[target_indices],
                    "intrinsics": intrinsics[target_indices],
                    "image": images[target_indices],
                    "near": self.u_near,
                    "far": self.u_far,
                    "indices": target_indices
                },
                "scene": scene
            }

            yield crop_example(example, tuple(self.config.image_shape))

    def __len__(self) -> int:
        return len(self.scenes)
