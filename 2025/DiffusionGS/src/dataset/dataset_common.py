from dataclasses import dataclass

@dataclass
class DatasetConfig:
    image_shape: list[int]
    crop_image: list[int]
    remove_background: bool
    center_and_scale: bool

