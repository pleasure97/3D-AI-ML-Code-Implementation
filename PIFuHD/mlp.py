import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self,
                 filter_channels,
                 merge_layer=0,
                 res_layers=[],
                 use_group_norm=True,
                 last_op=None
                 ):
        super().__init__()

        self.filters = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.merge_layer = merge_layer if merge_layer > 0 else len(filter_channels) // 2
        self.res_layers = res_layers
        self.use_group_norm = use_group_norm
        self.last_op = last_op

        for filter_channel in range(len(filter_channels) - 1):
            if filter_channel in self.res_layers:
                self.filters.append(
                    nn.Conv1d(
                        filter_channels[filter_channel] + filter_channels[0],
                        filter_channels[filter_channel + 1],
                        1))
            else:
                self.filters.append(
                    nn.Conv1d(
                        filter_channels[filter_channel],
                        filter_channels[filter_channel + 1],
                        1))
            if filter_channel != len(filter_channels) - 2:
                if use_group_norm:
                    self.norm_layers.append(nn.GroupNorm(32, filter_channels[filter_channel + 1]))
                else:
                    self.norm_layers.append(nn.BatchNorm1d(filter_channels[filter_channel + 1]))

    def forward(self, feature):
        '''
        :param feature: [B, C_in, N]
        :return: prediction : [B, C_out, N]
        '''
        y = feature
        tmp_y = feature
        phi = None

        for i, filter in enumerate(self.filters):
            y = filter(
                y if i not in self.res_layers
                else torch.cat([y, tmp_y], 1)
            )
            if i != len(self.filters) - 1:
                y = F.leaky_relu(self.norm_layers[i](y))
            if i == self.merge_layer:
                phi = y.clone()

        if self.last_op is not None:
            y = self.last_op(y)

        return y, phi
