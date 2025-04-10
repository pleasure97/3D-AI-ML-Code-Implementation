from .transformer_backbone import TransformerBackbone
from typing import Any
from .transformer_backbone import BackboneConfig

BACKBONES: dict[str, TransformerBackbone[Any]] = {}

def get_backbone(config: BackboneConfig, d_in: int) -> TransformerBackbone[Any]:
    return BACKBONES[config.name](config, d_in)
