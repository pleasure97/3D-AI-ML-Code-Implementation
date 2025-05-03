from dataclasses import dataclass

@dataclass
class DatasetConfig:
    name: str
    path: str
    image_shape: list[int]
    remove_background: bool
    center_and_scale: bool
    root: str

