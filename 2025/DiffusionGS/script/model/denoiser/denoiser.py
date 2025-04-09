from dataclasses import dataclass
from typing import Literal
from torch import nn, Tensor
from script.dataset.types import BatchedViews
from jaxtyping import Float
from backbone.transformer_backbone import BackboneConfig, TransformerBackbone
from embedding.patch_embedding import PatchEmbeddingConfig, PatchEmbedding
from embedding.positional_embedding import PositionalEmbeddingConfig, PositionalEmbedding
from viewpoint.RPPC import reference_point_plucker_embedding
from viewpoint.view_sampler import ViewSamplerConfig, ViewSampler

@dataclass
class DenoiserConfig:
    name: Literal["Denoiser"]
    transformer_backbone: BackboneConfig
    patch_embedding: PatchEmbeddingConfig
    positional_embedding: PositionalEmbeddingConfig
    view_sampler: ViewSamplerConfig


class Denoiser(nn.Module, DenoiserConfig):
    def __init__(self, config: DenoiserConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, context: BatchedViews) -> Float[Tensor]: