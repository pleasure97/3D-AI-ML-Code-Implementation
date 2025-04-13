from dataclasses import dataclass
import objaverse
from .dataset_common import DatasetConfig
from typing import Literal, List
from pathlib import Path
from torch.utils.data import IterableDataset
from .types import Stage
from ..model.denoiser.viewpoint.view_sampler import ViewSampler
import torch
from jaxtyping import Float, UInt8
from torch import Tensor
from PIL import Image
from io import BytesIO
import torchvision.transforms as transforms
import json
from .preprocessing.preprocess_utils import crop_example

@dataclass
class DatasetObjaverseConfig(DatasetConfig):
    name: Literal["Objaverse"]
    uids: List[str]
    max_fov: float
    num_images_to_render: int
    download_processes: int
    image_shape: tuple


class DatasetObjaverse(IterableDataset):
    config: DatasetObjaverseConfig
    stage: Stage
    view_sampler: ViewSampler
    chunks: list
    u_near: float = 0.1
    u_far: float = 4.2

    def __init__(self, config: DatasetObjaverseConfig, stage: Stage, view_sampler) -> None:
        super().__init__()
        self.config = config
        self.stage = stage
        self.view_sampler = view_sampler

        self.chunks = objaverse.load_objects(uids=self.config.uids, download_processes=self.config.download_processes)

        if self.stage in ("train", "val"):
            self.chunks = self.shuffle(self.chunks)

    @staticmethod
    def shuffle(self, chunks: list) -> list:
        indices = torch.randperm(len(chunks))
        return [chunks[index] for index in indices]

    @staticmethod
    def convert_poses(self, poses: Float[Tensor, "batch 18"]
                      ) -> tuple[Float[Tensor, "batch 4 4"], Float[Tensor, "batch 3 3"]]:
        batch_size = poses.shape[0]

        # Convert the intrinsics to a 3 by 3 matrix.
        intrinsics = torch.eye(3, dtype=torch.float32).unsqueeze(0).expand(batch_size, -1, -1).clone()
        focal_x, focal_y, center_x, center_y = poses[:, 0], poses[:, 1], poses[:, 2], poses[:, 3]
        intrinsics[:, 0, 0] = focal_x
        intrinsics[:, 1, 1] = focal_y
        intrinsics[:, 0, 2] = center_x
        intrinsics[:, 1, 2] = center_y

        # Convert the extrinsics to a 4 by 4 matrix.
        world_to_camera = torch.eye(4, dtype=torch.float32).unsqueeze(0).expand(batch_size, -1, -1).clone()
        return world_to_camera.inverse(), intrinsics

    @staticmethod
    def convert_images(self, images: list[UInt8[Tensor, "..."]]) -> Float[Tensor, "batch 3 height width"]:
        outputs = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            outputs.append(transforms.ToTensor()(image))
        return torch.stack(outputs)

    def __iter__(self):
        if self.stage in ("train", "val"):
            self.chunks = self.shuffle(self.chunks)

        worker_info = torch.utils.data.get_worker_info()

        if self.stage == "test" and worker_info is not None:
            self.chunks = [chunk for chunk_index, chunk in enumerate(self.chunks)
                           if chunk_index % worker_info.num_workers == worker_info.id]

        for chunk_path in self.chunks:
            chunk = torch.load(chunk_path)
            if self.stage in ("train", "val"):
                chunk = self.shuffle(chunk)
            for example in chunk:
                extrinsics, intrinsics = self.convert_poses(example["cameras"])
                scene = example["key"]

                source_indices, target_indices = self.view_sampler.sample(scene, extrinsics, intrinsics)

                source_images = [example["images"][source_index.item()] for source_index in source_indices]
                target_images = [example["images"][target_index.item()] for target_index in target_indices]

                example = {
                    "source": {
                        "extrinsics": extrinsics[source_indices],
                        "intrinsics": intrinsics[source_indices],
                        "image": source_images,
                        "near": self.u_near,
                        "far": self.u_far,
                        "indices": source_indices
                    },
                    "target": {
                        "extrinsics": extrinsics[target_indices],
                        "intrinsics": intrinsics[target_indices],
                        "image": target_images,
                        "near": self.u_near,
                        "far": self.u_far,
                        "indices": target_indices
                    },
                    "scene": scene
                }

                yield crop_example(example, tuple(self.config.image_shape))

    def __len__(self) -> int:
        return len(self.chunks)

