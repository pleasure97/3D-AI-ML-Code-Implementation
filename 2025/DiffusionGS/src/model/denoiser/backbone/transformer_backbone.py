from dataclasses import dataclass
from typing import Literal
from ..embedding.timestep_embedding import TimestepEmbeddingConfig, TimestepEmbedding
import torch.nn as nn

@dataclass
class BackboneLayerConfig:
    name: Literal["TransformerBackboneLayer"]
    timestep_embedding: TimestepEmbedding[TimestepEmbeddingConfig]
    attention_dim: int
    num_heads: int
    dropout: float
@dataclass
class BackboneConfig:
    name: Literal["TransformerBackbone"]
    layer: BackboneLayerConfig
    num_layers: int

class TransformerBackboneLayer(nn.Module, BackboneLayerConfig):
    def __init__(self, config: BackboneLayerConfig):
        super().__init__()

        self.config = config

        self.timestep_embedding = self.config.timestep_embedding

        self.self_attn = nn.MultiheadAttention(self.config.attention_dim, num_heads=12)

        self.layer_norm = nn.LayerNorm(normalized_shape=self.config.attention_dim)

        self.mlp = nn.Sequential(
            nn.Linear(self.config.attention_dim, self.config.attention_dim * 4),
            nn.GELU(),
            nn.Dropout(p=self.config.dropout),
            nn.Linear(self.attention_dim * 4, self.attention_dim),
            nn.Dropout(p=self.config.dropout)
        )

    def forward(self, x):
        # timestep_embedding : [batch_size, 1, embedding_dim]
        x = x + self.timestep_embedding  # [batch_size, num_patches, embedding_dim]
        x = x.transpose(0, 1)  # [num_patches, batch_size, embedding_dim]
        attn_output, _ = self.self_attn(x, x, x)
        x = x + attn_output
        x = x.transpose(0, 1)  # [batch_size, num_patches, embedding_dim]
        x = self.layer_norm(x)

        x = x + self.timestep_embedding
        mlp_output = self.mlp(x)
        x = self.layer_norm(x)

        return x


class TransformerBackbone(nn.Module):
    def __init__(self, config: BackboneConfig):
        super().__init__()

        self.config = config
        self.layers = nn.ModuleList([
            TransformerBackboneLayer(timestep_embedding=self.config.layer.timestep_embedding,
                                     embedding_dim=self.config.layer.attention_dim,
                                     num_heads=self.config.layer.num_heads,
                                     dropout=self.config.layer.dropout)
            for _ in range(self.config.num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
