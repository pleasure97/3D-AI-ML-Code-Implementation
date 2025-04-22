from dataclasses import dataclass
import objaverse
from src.dataset.dataset_common import DatasetConfig
from typing import Literal, List, Set
from torch.utils.data import IterableDataset
from src.dataset.types import Stage
from src.model.denoiser.viewpoint.view_sampler import ViewSampler
import torch
from jaxtyping import Float, UInt8
from torch import Tensor
from PIL import Image
from io import BytesIO
import torchvision.transforms as transforms
from src.dataset.preprocessing.preprocess_utils import crop_example
import os
import json

@dataclass
class DatasetObjaverseConfig(DatasetConfig):
    name: Literal["Objaverse"]
    root: str
    uids: List[str]
    max_fov: float
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

            for ch in chunk:
                extrinsics, intrinsics = self.convert_poses(ch["cameras"])
                scene = ch["key"]
                source_indices, target_indices = self.view_sampler.sample(extrinsics)

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
        return len(self.pending_uids)

