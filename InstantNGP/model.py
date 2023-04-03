import torch
import torch.nn as nn
import torch.nn.functional as F


class InstantNeRF(nn.Module):
    def __init__(self,
                 num_layers = 3,
                 hidden_dim = 64,
                 geo_feat_dim = 15,
                 num_layers_color = 4,
                 hidden_dim_color = 64,
                 input_channel = 3,
                 input_channel_views = 3
                 ):

        self.input_channel = input_channel
        self.input_channel_views = input_channel_views

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim =hidden_dim
        self.geo_feat_dim = geo_feat_dim

        sigma_net = []
        for l in range(self.num_layers):
            if l == 0:
                in_dim = self.input_channel
            else:
                in_dim = hidden_dim

            if l == self.num_layers - 1 :
                out_dim = 1 + self.geo_feat_dim
            else :
                out_dim = hidden_dim


            sigma_net.append(nn.Linear(in_dim, out_dim, bias = False))

        self.sigma_net = nn.ModuleList(sigma_net)

        # color network
        self.num_layers_color = num_layers_color
        self.hidden_dim_color = hidden_dim_color

        color_net = []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.input_channel_views + self.geo_feat_dim
            else:
                in_dim = hidden_dim_color

            if l == num_layers_color - 1:
                out_dim = 3
            else :
                out_dim = hidden_dim_color

            color_net.append(nn.Linear(in_dim, out_dim, bias = False))

        self.color_net = nn.ModuleList(color_net)

    def forward(self, x):
        input_points, input_views = torch.split(x, [self.input_channel, self.input_channel_views], dim = -1)

        # sigma
        h = input_points
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace = True)

        sigma, geo_feat = h[..., 0], h[..., 1]

        # color
        h = torch.cat([input_views, geo_feat], dim = -1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace = True)

        color = h
        outputs = torch.cat([color, sigma.unsqueeze(dim = -1)], -1)

        return outputs