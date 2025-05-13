from dataclasses import dataclass
from src.preprocess.dataset_common import DatasetConfig
from typing import Literal
from torch.utils.data import IterableDataset
import torch
from torchvision.transforms import ToTensor
import os
from pathlib import Path
from PIL import Image
from src.preprocess.types import Stage
from src.model.denoiser.viewpoint.view_sampler import ViewSampler
from src.utils.geometry_util import convert_cameras_bin, convert_images_bin, make_rotation_matrix
from src.preprocess.preprocess_utils import crop_example

@dataclass
class DatasetMVImgNetConfig(DatasetConfig):
    name: Literal["MVImgNet"]
    root: str
    max_fov: float

class DatasetMVImgNet(IterableDataset):
    config: DatasetMVImgNetConfig
    stage: Stage
    view_sampler: ViewSampler

    def __init__(self,
                 config: DatasetMVImgNetConfig,
                 stage: Stage,
                 view_sampler: ViewSampler,
                 device: torch.device = "cuda" if torch.cuda.is_available() else "cpu") -> None:
        super().__init__()
        self.config = config
        self.stage = stage
        self.view_sampler = view_sampler
        self.device = device

        root = Path(self.config.root)
        scenes = list(root.glob("*/*"))
        self.scenes = [scene for scene in scenes if (scene / "sparse" / "0").is_dir()]

        if self.stage in ("train", "val"):
            self.scenes = [self.scenes[i] for i in torch.randperm(len(self.scenes))]

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:
            num_workers, worker_id = worker_info.num_workers, worker_info.id
            stream = (scene for i, scene in enumerate(self.scenes) if i % num_workers == worker_id)
        else:
            stream = iter(self.scenes)

        for scene in stream:
            # Read COLMAP Binaries
            sparse_directory = os.path.join(scene, "sparse", "0")
            cameras_dict = convert_cameras_bin(os.path.join(sparse_directory, "cameras.bin"))
            images_dict = convert_images_bin(os.path.join(sparse_directory, "images.bin"))

            view_ids = sorted(images_dict.keys())

            images = []
            extrinsics = []
            intrinsics = []

            for view_id in view_ids:
                # Extrinsic
                camera_id, rotation_matrix, translation_vector, name = images_dict[view_id]
                extrinsic = torch.eye(4, device=self.device)
                extrinsic[:3, :3] = torch.from_numpy(rotation_matrix)
                extrinsic[:3, 3] = torch.from_numpy(translation_vector)
                extrinsics.append(extrinsic)

                # Intrinsic
                intrinsic, _, _ = cameras_dict[camera_id]
                intrinsics.append(torch.from_numpy(intrinsic).to(self.device))

                # File Name
                file_name = f"{view_id:03d}.jpg"
                image_path = os.path.join(scene, "images", file_name)
                pil = Image.open(image_path).convert("RGB")
                images.append(ToTensor()(pil).to(self.device))

            extrinsics = torch.stack(extrinsics, dim=0)
            intrinsics = torch.stack(intrinsics, dim=0)
            images = torch.stack(images, dim=0)

            source_indices, target_indices = self.view_sampler.sample(extrinsics)

            example = {
                "source": {
                    "extrinsics": extrinsics[source_indices],
                    "intrinsics": intrinsics[source_indices],
                    "image": images[source_indices],
                    "near": self.config.u_near,
                    "far": self.config.u_far,
                    "indices": source_indices
                },
                "target": {
                    "extrinsics": extrinsics[target_indices],
                    "intrinsics": intrinsics[target_indices],
                    "image": images[target_indices],
                    "near": self.config.u_near,
                    "far": self.config.u_far,
                    "indices": target_indices
                },
                "scene": scene
            }

            yield crop_example(example, tuple(self.config.image_shape))

    def __len__(self):
        return len(self.scenes)
