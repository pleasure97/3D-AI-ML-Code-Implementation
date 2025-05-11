from dataclasses import dataclass
from src.model.denoiser.viewpoint.view_sampler import ViewSamplerConfig

@dataclass
class DatasetConfig:
    name: str
    path: str
    image_shape: list[int]
    remove_background: bool
    center_and_scale: bool
    root: str
    view_sampler: ViewSamplerConfig

