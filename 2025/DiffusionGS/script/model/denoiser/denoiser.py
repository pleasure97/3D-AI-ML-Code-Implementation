from dataclasses import dataclass
from typing import Literal

@dataclass
class DenoiserConfig:
    name: Literal["Denoiser"]