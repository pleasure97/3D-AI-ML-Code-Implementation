import objaverse
from dataclasses import dataclass
from typing import Literal, List, Set
from jaxtyping import Float, UInt8
from PIL import Image
from io import BytesIO
import os
import json
from src.preprocess.dataset_common import DatasetConfig
from src.preprocess.types import Stage
from src.preprocess.preprocess_utils import crop_example
from src.model.denoiser.viewpoint.view_sampler import ViewSampler
from src.model.denoiser.viewpoint.RPPC import reference_point_plucker_embedding
import torch
from torch import Tensor
from torch.utils.data import IterableDataset
import torchvision.transforms as transforms


@dataclass
class DatasetObjaverseConfig(DatasetConfig):
    name: Literal["Objaverse"]
    root: str
    uids: List[str]
    max_fov: float
    download_processes: int
    image_shape: tuple
    u_near: float
    u_far: float


class DatasetObjaverse(IterableDataset):
    config: DatasetObjaverseConfig
    stage: Stage
    view_sampler: ViewSampler
    chunks: list

    def __init__(self, config: DatasetObjaverseConfig, stage: Stage, view_sampler) -> None:
        super().__init__()
        print(f"[DEBUG] pending_uids ({len(self.pending_uids)}): {self.pending_uids[:5]} …")
        self.config = config
        self.stage = stage
        self.view_sampler = view_sampler

        # Load downloaded uids
        self.downloaded_uids: Set[str] = self._load_downloaded_uids()
        # Prepare pending uids
        self.pending_uids = [uid for uid in self.config.uids if uid not in self.downloaded_uids]

        self.streaming_downloader = self._streaming_download()

    def _load_downloaded_uids(self) -> Set[str]:
        if os.path.exists(self.config.root):
            with open(self.config.root, "r") as f:
                return set(json.load(f))
        return set()

    def _save_downloaded_uids(self, uid: str):
        self.downloaded_uids.add(uid)
        with open(self.config.root, "w") as f:
            json.dump(list(self.downloaded_uids), f)

    def _streaming_download(self):
        for objaverse_object in objaverse.load_objects(uids=self.pending_uids,
                                                       download_processes=self.config.download_processes):
            uid = objaverse_object["uid"]
            path = objaverse_object["path"]
            self._save_downloaded_uids(uid)
            yield path

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
        print("[DEBUG] __iter__ called")
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:
            num_workers, worker_id = worker_info.num_workers, worker_info.id
            self._streaming_download = (functor for i, functor in enumerate(self._streaming_download)
                                        if i % num_workers == worker_id)

        for chunk_path in self._streaming_download:
            try:
                chunk = torch.load(chunk_path)
            except Exception as e:
                print(f"Failed to load {chunk_path}: {e}")
                continue

            if self.stage in ("train", "val"):
                chunk = [chunk[i] for i in torch.randperm(len(chunk))]

            for example in chunk:
                extrinsics, intrinsics = self.convert_poses(example["cameras"])
                scene = example["key"]
                clean_index, noisy_indices, novel_indices = self.view_sampler.sample(extrinsics)

                clean_info = self._prepare_views(clean_index, extrinsics, intrinsics, example["images"])
                noisy_info = self._prepare_views(noisy_indices, extrinsics, intrinsics, example["images"])
                novel_info = self._prepare_views(novel_indices, extrinsics, intrinsics, example["images"])

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

    def __len__(self):
        return len(self.chunks)
