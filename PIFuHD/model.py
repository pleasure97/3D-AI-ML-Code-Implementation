from geometry import index, orthogonal, perspective
import torch.nn as nn

class BaseNetwork(nn.Module):
    def __init__(self, projection_mode='orthogonal', criteria={'occ':nn.MSELoss}):
        super().__init__()

        self.name = 'base_network'
        self.criteria = criteria

        self.projection = None
        self.index = None

        self.preds = None
        self.labels = None
        self.nmls = None
        self.labels_nml = None
        self.preds_surface = None

    def forward(self, points, images, calibrations, transforms=None):
        
