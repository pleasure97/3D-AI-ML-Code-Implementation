import torch
import torch.nn as nn
import torch.nn.functional as F


def build_conv(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, batch_size, use_group_norm=False):
        super().__init__()

        self.conv1 = build_conv(in_channels, int(out_channels / 2), kernel_size, stride, padding)
        self.conv2 = build_conv(int(out_channels / 2), int(out_channels / 4), kernel_size, stride, padding)
        self.conv3 = build_conv(int(out_channels / 4), int(out_channels / 4), kernel_size, stride, padding)

        if not use_group_norm:
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.bn2 = nn.BatchNorm2d(int(out_channels / 2))
            self.bn3 = nn.BatchNorm2d(int(out_channels / 4))
            self.bn4 = nn.BatchNorm2d(in_channels)
        else:
            self.bn1 = nn.GroupNorm(batch_size, in_channels)
            self.bn2 = nn.GroupNorm(batch_size, int(out_channels / 2))
            self.bn3 = nn.GroupNorm(batch_size, int(out_channels / 4))
            self.bn4 = nn.GroupNorm(batch_size, in_channels)

        if in_channels != out_channels:
            self.down_sample = nn.Sequential(
                self.bn4,
                nn.ReLU(True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            )
        else:
            self.down_sample = None

    def forward(self, x):
        res = x

        out1 = self.conv1(F.relu(self.bn1(x), True))
        out2 = self.conv2(F.relu(self.bn2(out1), True))
        out3 = self.conv3(F.relu(self.bn3(out2), True))

        out = torch.cat([out1, out2, out3], 1)

        if self.down_sample is not None:
            res = self.down_sample(res)

        out += res

        return out


class HourGlassNetwork(nn.Module):
    def __init__(self, depth, in_channels, out_channels, kernel_size, stride, padding, batch_size, use_group_norm=False):
        super().__init__()
        self.name = 'hourglass'
        self.depth = depth

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.use_group_norm = use_group_norm
        self.batch_size = batch_size

        self._generate_block(self.depth)

    def _generate_block(self, depth):
