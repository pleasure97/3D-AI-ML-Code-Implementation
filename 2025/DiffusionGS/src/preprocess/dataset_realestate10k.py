from dataclasses import dataclass
from typing import Literal
from jaxtyping import Float, UInt8
from pathlib import Path
from PIL import Image
from io import BytesIO
import json
import torch
from torch import Tensor
from torch.utils.data import IterableDataset
from src.preprocess.dataset_common import DatasetConfig
from src.preprocess.types import Stage
from src.preprocess.preprocess_utils import crop_example
from src.model.denoiser.viewpoint.view_sampler import ViewSampler
from src.model.denoiser.viewpoint.RPPC import reference_point_plucker_embedding
import torchvision.transforms as transforms


@dataclass
class DatasetRealEstate10KConfig(DatasetConfig):
    name: Literal["RealEstate10K"]
    roots: list[Path]
    image_shape: tuple
    background_color: tuple
    cameras_are_circular: bool
    max_fov: float

class DatasetRealEstate10K(IterableDataset):
    config: DatasetRealEstate10KConfig
    stage: Stage
    view_sampler: ViewSampler

    def __init__(self,
                 config: DatasetRealEstate10KConfig,
                 stage: Stage,
                 view_sampler: ViewSampler) -> None:
        super().__init__()
        self.config = config
        self.stage = stage
        self.view_sampler = view_sampler

        self.chunks = []

        for root in config.roots:
            root = root / self.stage
            root_chunks = sorted([path for path in root.iterdir() if path.suffix == ".torch"])
            self.chunks.extend(root_chunks)

    @staticmethod
    def shuffle(self, chunks: list) -> list:
        indices = torch.randperm(len(chunks))
        return [chunks[index] for index in indices]

    @staticmethod
    def convert_poses(self, poses: Float[Tensor, "batch 18"]
                      ) -> tuple[Float[Tensor, "batch 4 4"], Float[Tensor, "batch 3 3"]]:
        batch_size = poses.shape[0]

        # Convert the intrinsics to a 3 by 3 matrix.
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = intrinsics.unsqueeze(0).expand(batch_size, -1, -1).clone()
        focal_x, focal_y, center_x, center_y = poses[:, 0], poses[:, 1], poses[:, 2], poses[:, 3]
        intrinsics[:, 0, 0] = focal_x
        intrinsics[:, 1, 1] = focal_y
        intrinsics[:, 0, 2] = center_x
        intrinsics[:, 1, 2] = center_y

        # Convert the extrinsics to a 4 by 4 matrix.
        world_to_camera = torch.eye(4, dtype=torch.float32)
        world_to_camera = world_to_camera.unsqueeze(0).expand(batch_size, -1, -1).clone()

        return world_to_camera.inverse(), intrinsics

    @staticmethod
    def convert_images(self, images: list[UInt8[Tensor, "..."]]) -> Float[Tensor, "batch 3 height width"]:
        outputs = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            outputs.append(transforms.ToTensor(image))
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

                clean_index, noisy_indices, novel_indices = self.view_sampler.sample(scene, extrinsics)
                clean_info = self._prepare_views(clean_index, extrinsics, intrinsics, example["images"])
                noisy_info = self._prepare_views(noisy_indices, extrinsics, intrinsics, example["images"])
                novel_info = self._prepare_views(clean_index, extrinsics, intrinsics, example["images"])

                example = {
                    "clean": clean_info,
                    "noisy": noisy_info,
                    "novel": novel_info,
                    "scene": scene
                }

                yield crop_example(example, tuple(self.config.image_shape))

    def _prepare_views(self, indices, extrinsics, intrinsics, images, jitter=False):
        num_sampled_views = indices.shape[0]

        if isinstance(indices, int):
            indices = torch.tensor([indices], device=self.device)
        elif isinstance(indices, list):
            indices = torch.tensor(indices, device=self.device)

        sampled_extrinsics = extrinsics[indices]  # [num_sampled_views, 4, 4]
        sampled_intrinsics = intrinsics[indices]  # [num_sampled_views, 3, 3]
        sampled_views = images[indices]

        u_nears = torch.full((num_sampled_views,), self.config.u_near, device=self.device)
        u_fars = torch.full((num_sampled_views,), self.config.u_far, device=self.device)

        C2Ws = torch.inverse(sampled_extrinsics)
        RPPCs = reference_point_plucker_embedding(
            self.config.image_shape[0],
            self.config.image_shape[1],
            sampled_intrinsics,
            C2Ws,
            jitter=jitter)  # [num_noisy_views, 6, Height, Width]

        return {
            "extrinsics": sampled_extrinsics,
            "intrinsics": sampled_intrinsics,
            "views": sampled_views,
            "nears": u_nears,
            "fars": u_fars,
            "indices": indices,
            "RPPCs": RPPCs
        }

    def index(self) -> dict[str, Path]:
        merged_index = {}
        data_stages = [self.stage]
        for data_stage in data_stages:
            for root in self.config.roots:
                with (root / data_stage / "index.json").open("r") as f:
                    index = json.load(f)
                index = {key: Path(root / data_stage / value) for key, value in index.items()}
                assert not (set(merged_index.keys()) & set(index.keys()))
                merged_index = {**merged_index, **index}
        return merged_index

    def __len__(self) -> int:
        return len(self.index.keys())
