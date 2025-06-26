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
from src.utils.geometry_util import convert_cameras_bin, convert_images_bin
from src.preprocess.preprocess_utils import crop_example
from src.model.denoiser.viewpoint.RPPC import reference_point_plucker_embedding


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
                 device: torch.device = "cpu") -> None:
        super().__init__()
        self.config = config
        self.stage = stage
        self.dataset_iterator = None
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

            # View Sampler
            extrinsics = torch.stack(extrinsics, dim=0)
            intrinsics = torch.stack(intrinsics, dim=0)
            images = torch.stack(images, dim=0)

            clean_index, noisy_indices, novel_indices = self.view_sampler.sample(extrinsics)
            clean_index = int(clean_index.item())

            # Sample Clean View
            clean_extrinsic = extrinsics[clean_index].unsqueeze(0)  # [num_clean_view, 4, 4]
            clean_intrinsic = intrinsics[clean_index].unsqueeze(0)  # [num_clean_view, 3, 3]
            clean_view = images[clean_index].unsqueeze(0)           # [num_clean_view, 3, Height, Width]
            clean_u_near = torch.tensor([self.config.u_near], device=self.device)     # [1]
            clean_u_far = torch.tensor([self.config.u_far], device=self.device)       # [1]
            clean_C2W = torch.inverse(clean_extrinsic)
            clean_RPPC = reference_point_plucker_embedding(
                self.config.image_shape[0],
                self.config.image_shape[1],
                clean_intrinsic,
                clean_C2W,
                jitter=False)

            # Sample Noisy Views
            num_noisy_views = noisy_indices.shape[0]
            noisy_extrinsics = extrinsics[noisy_indices]    # [num_noisy_views, 4, 4]
            noisy_intrinsics = intrinsics[noisy_indices]    # [num_noisy_views, 3, 3]
            noisy_views = images[noisy_indices]            # [num_noisy_views, 3, Height, Width]
            noisy_u_nears = torch.full((num_noisy_views,), self.config.u_near, device=self.device)
            noisy_u_fars = torch.full((num_noisy_views,), self.config.u_far, device=self.device)
            noisy_C2Ws = torch.inverse(noisy_extrinsics)
            noisy_RPPCs = reference_point_plucker_embedding(
                self.config.image_shape[0],
                self.config.image_shape[1],
                noisy_intrinsics,
                noisy_C2Ws,
                jitter=False)                               # [num_noisy_views, 6, Height, Width]

            # Construct "example" dictionary.
            example = {
                "clean": {
                    "extrinsics": clean_extrinsic,
                    "intrinsics": clean_intrinsic,
                    "views": clean_view,
                    "nears": clean_u_near,
                    "fars": clean_u_far,
                    "indices": clean_index,
                    "RPPCs": clean_RPPC
                },
                "noisy": {
                    "extrinsics": noisy_extrinsics,
                    "intrinsics": noisy_intrinsics,
                    "views": noisy_views,
                    "nears": noisy_u_nears,
                    "fars": noisy_u_fars,
                    "indices": noisy_indices,
                    "RPPCs": noisy_RPPCs
                },
                "scene": str(scene)
            }

            yield crop_example(example, tuple(self.config.image_shape))

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, index: int):
        # Assume that the dataset is type of IterableDataset
        if self.dataset_iterator is None:
            self.dataset_iterator = iter(self.dataset_iterator)
        return next(self.dataset_iterator)
