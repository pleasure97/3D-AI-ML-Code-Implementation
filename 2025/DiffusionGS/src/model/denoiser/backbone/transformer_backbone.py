from dataclasses import dataclass
from typing import Literal
from src.model import ModuleWithConfig
from src.model.denoiser.embedding.timestep_embedding import TimestepMLP, TimestepMLPConfig
import torch.nn as nn

@dataclass
class BackboneLayerConfig:
    name: Literal["transformer_backbone_layer"]
    timestep_mlp: TimestepMLPConfig
    attention_dim: int
    num_heads: int
    dropout: float
@dataclass
class BackboneConfig:
    name: Literal["transformer_backbone"]
    layer: BackboneLayerConfig
    num_layers: int

class TransformerBackboneLayer(ModuleWithConfig[BackboneLayerConfig]):
    def __init__(self, config: BackboneLayerConfig):
        super().__init__(config)

        self.config = config

        self.timestep_mlp = TimestepMLP(self.config.timestep_mlp)

        self.self_attn = nn.MultiheadAttention(self.config.attention_dim, num_heads=self.config.num_heads)

        self.layer_norm = nn.LayerNorm(normalized_shape=self.config.attention_dim)

        self.mlp = nn.Sequential(
            nn.Linear(self.config.attention_dim, self.config.attention_dim * 4),
            nn.GELU(),
            nn.Dropout(p=self.config.dropout),
            nn.Linear(self.config.attention_dim * 4, self.config.attention_dim),
            nn.Dropout(p=self.config.dropout)
        )

    def forward(self, x, timestep, rppc):
        # timestep_embedding : [batch_size, 1, embedding_dim]
        timestep_embedding = self.timestep_mlp(timestep)
        # TODO - Change Variable Name
        B, D, H, W = rppc.shape
        N = H * W
        rppc_view = rppc.view(B, D, N).permute(0, 2, 1)
        x = x + timestep_embedding + rppc_view  # [batch_size, num_patches, embedding_dim]
        x = x.transpose(0, 1)  # [num_patches, batch_size, embedding_dim]
        attn_output, _ = self.self_attn(x, x, x)
        x = x + attn_output
        x = x.transpose(0, 1)  # [batch_size, num_patches, embedding_dim]
        x = self.layer_norm(x)

        x = x + timestep_embedding + rppc_view
        mlp_output = self.mlp(x)
        x = self.layer_norm(mlp_output)

        return x


class TransformerBackbone(ModuleWithConfig[BackboneConfig]):
    def __init__(self, config: BackboneConfig):
        super().__init__(config)

        self.config = config
        self.layers = nn.ModuleList([
            TransformerBackboneLayer(self.config.layer)
            for _ in range(self.config.num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
