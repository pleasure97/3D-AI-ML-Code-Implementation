from dataclasses import dataclass
from typing import Literal
from src.model import ModuleWithConfig
from src.model.denoiser.embedding.timestep_embedding import TimestepMLP, TimestepMLPConfig
import torch.nn as nn
import torch


@dataclass
class BackboneLayerConfig:
    name: Literal["transformer_backbone_layer"]
    timestep_mlp: TimestepMLPConfig
    patch_size: int
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

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # rppc_patcher - viewpoint conditions?
        self.rppc_patcher = nn.Conv2d(
            in_channels=6,
            out_channels=self.config.attention_dim,
            kernel_size=self.config.patch_size,
            stride=self.config.patch_size,
            padding=0,
            device=self.device)

    def forward(self, x, timestep: torch.Tensor, rppc: torch.Tensor):
        batch_size, num_views, channels, height, width = rppc.shape

        rppc_tokens = []
        patches_per_view = None
        for view in range(num_views):
            rppc_view = rppc[:, view]  # [batch_size, 6, height, width]
            rppc_patched = self.rppc_patcher(rppc_view)  # [batch_size, attention_dim, patch_height, patch_width]
            if patches_per_view is None:
                _, _, patch_height, patch_width = rppc_patched.shape
                patches_per_view = patch_height * patch_width
            rppc_patched = rppc_patched.flatten(2).permute(0, 2, 1)  # [batch_size, patch_size, attention_dim]
            rppc_tokens.append(rppc_patched)

        rppc_embedding = torch.cat(rppc_tokens, dim=1)

        # timestep_embedding : [batch_size, 1, embedding_dim]
        timestep_embedding = self.timestep_mlp(timestep)  # [batch_size, embedding_dim]
        timestep_embedding = timestep_embedding.expand(-1, patches_per_view * num_views, -1)

        combined_embedding = timestep_embedding + rppc_embedding

        x = x + combined_embedding  # [batch_size, num_patches, embedding_dim]
        x = x.transpose(0, 1)  # [num_patches, batch_size, embedding_dim]
        attn_output, _ = self.self_attn(x, x, x)
        x = x + attn_output
        x = x.transpose(0, 1)  # [batch_size, num_patches, embedding_dim]
        x = self.layer_norm(x)

        x = x + combined_embedding
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

    def forward(self, x, timestep: torch.Tensor, RPPC: torch.Tensor):
        for layer in self.layers:
            x = layer(x, timestep, RPPC)
        return x
