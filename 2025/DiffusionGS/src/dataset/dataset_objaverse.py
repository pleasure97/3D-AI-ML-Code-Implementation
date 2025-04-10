from dataclasses import dataclass
from .dataset_common import DatasetConfig
from typing import Literal
from pathlib import Path
from torch.utils.data import IterableDataset
from .types import Stage
from ..model.denoiser.viewpoint.view_sampler import ViewSampler
import torch
from jaxtyping import Float
from torch import Tensor
from einops import repeat

@dataclass
class DatasetObjaverseConfig(DatasetConfig):
    name: Literal["Objaverse"]
    roots: list[Path]
    max_fov: float
    num_images_to_render: int


class DatasetObjaverse(IterableDataset):
    config: DatasetObjaverseConfig
    stage: Stage
    view_sampler: ViewSampler
    chunks: list[Path]
    u_near: float = 0.1
    u_far: float = 4.2

    def __init__(self, config: DatasetObjaverseConfig, stage: Stage, view_sampler) -> None:
        super().__init__()
        self.config = config
        self.stage = stage
        self.view_sampler = view_sampler

        self.chunks = []
        for root in self.config.roots:
            root = root / self.stage
            root_chunks = sorted([path for path in root.iterdir() if path.suffix == ".torch"])
            self.chunks.extend(root_chunks)

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
                extrinsics, intrinsics = self.

    @staticmethod
    def shuffle(self, chunks: list) -> list:
        indices = torch.randperm(len(chunks))
        return [chunks[index] for index in indices]

    @staticmethod
    def convert_poses(self,
                      poses: Float["batch 18"], # 0 ~ 3 : Intrinsics / 4 ~ 5 : Empty / 6~17: Extrinsics
                     ) -> tuple[Float[Tensor, "batch 4 4"], # Intrinsics
                                Float[Tensor, "batch 3 3"]]: # Extrinsics
        batch, _ = poses.shape

        # Intrinsics
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=batch).clone()
        fx, fy, cx, cy = poses[:, 4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy

        # Extrinsics (world-to-camera matrix)
        w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=batch).clone()


    def __len__(self) -> int:
        pass

