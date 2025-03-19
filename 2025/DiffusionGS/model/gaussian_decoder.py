import torch
import torch.nn as nn

class GaussianDecoder(nn.Module):
    def __init__(self, transformer_output_tensor: torch.Tensor, u_near: float, u_far: float, 
                 input_dim: int=768, hidden_dim: int=768, output_dim: int=14, weight: float=0.5):
        super().__init__()

        self.transformer_output_tensor = transformer_output_tensor

        self.u_near = u_near
        self.u_far = u_far

        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU())

        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim))
        
        # weight to control center
        self.weight = weight

    def forward(self, x):
        output1 = self.mlp1(x)

        output1 = output1 + self.transformer_output_tensor.mean(dim=1, keepdim=True)

        output2 = self.mlp2(output1)

        # Positions are clipped to [-1, 1]^3
        position = torch.tanh(output2[:, :, :3])

        # Depth Transition
        depth = self.weight * self.u_near + (1 - self.weight) * self.u_far
        position[:, :, 2] = depth

        # Color is in [0, 1]
        color = torch.sigmoid(output2[:, :, 3:7])

        # Scale > 0
        scale = torch.exp(output2[:, :, 7:10])

        # Rotation
        rotation = output2[:, :, 10:13]

        # Opacity
        opacity = torch.sigmoid(output2[:, :, 13:14])

        return position, color, scale, rotation, opacity
