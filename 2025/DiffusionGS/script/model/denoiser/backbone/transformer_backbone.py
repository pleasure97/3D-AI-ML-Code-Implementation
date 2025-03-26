import torch
import torch.nn as nn

class TransformerBackboneLayer(nn.Module):
  def __init__(self, timestep_embedding: torch.Tensor, embedding_dim: int=768, num_heads: int=12, dropout: float=0.1):
    super().__init__()

    self.timestep_embedding = timestep_embedding

    self.self_attn = nn.MultiheadAttention(embedding_dim, num_heads=12)

    self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

    self.mlp = nn.Sequential(
        nn.Linear(embedding_dim, embedding_dim * 4),
        nn.GELU(),
        nn.Dropout(p=dropout),
        nn.Linear(embedding_dim * 4, embedding_dim),
        nn.Dropout(p=dropout)
    )

  def forward(self, x):
    # timestep_embedding : [batch_size, 1, embedding_dim]
    x = x + self.timestep_embedding # [batch_size, num_patches, embedding_dim]
    x = x.transpose(0, 1) # [num_patches, batch_size, embedding_dim]
    attn_output, _ = self.self_attn(x, x, x)
    x = x + attn_output
    x = x.transpose(0, 1) # [batch_size, num_patches, embedding_dim]
    x = self.layer_norm(x)

    x = x + self.timestep_embedding
    mlp_output = self.mlp(x)
    x = self.layer_norm(x)

    return x

class TransformerBackbone(nn.Module):
    def __init__(self, num_layers: int, timestep_embedding:torch.Tensor, embedding_dim: int = 768, num_heads: int = 12, dropout: float = 0.1):
        super().__init__()

        self.timestep_embedding = timestep_embedding

        self.layers = nn.ModuleList([
            TransformerBackboneLayer(timestep_embedding=self.timestep_embedding, embedding_dim=embedding_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x