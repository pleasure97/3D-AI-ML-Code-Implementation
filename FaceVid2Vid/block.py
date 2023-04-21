import torch.nn as nn
import torch.nn.functional as F


class DownBlock2D(nn.Module):

    def __init__(self, input_channels, output_channels, filter_size=3):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.filter_size = filter_size

        self.conv = nn.Conv2d(input_channels, output_channels, (filter_size, filter_size), bias=False)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = F.relu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class DownBlock3D(nn.Module):

    def __init__(self, input_channels, output_channels, filter_size=3):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.filter_size = filter_size

        self.conv = nn.Conv3d(input_channels, output_channels, (filter_size, filter_size, filter_size), bias=False)
        self.bn = nn.BatchNorm3d(output_channels)
        self.relu = F.relu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class UpBlock2D(nn.Module):

    def __init__(self, input_channels, output_channels, filter_size=3):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.filter_size = filter_size

        self.conv = nn.Conv2d(input_channels, output_channels, (filter_size, filter_size), bias=False)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = F.relu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class UpBlock3D(nn.Module):

    def __init__(self, input_channels, output_channels, filter_size=3):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.filter_size = filter_size

        self.conv = nn.Conv3d(input_channels, output_channels, (filter_size, filter_size, filter_size), bias=False)
        self.bn = nn.BatchNorm3d(output_channels)
        self.relu = F.relu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class ResBlock2D(nn.Module):

    def __init__(self, num_channels, filter_size=3):
        super().__init__()
        self.num_channels = num_channels
        self.filter_size = filter_size

        self.conv = nn.Conv2d(num_channels, num_channels, (filter_size, filter_size), bias=False)
        self.bn = nn.BatchNorm2d(num_channels)
        self.relu = F.relu

    def forward(self, x):
        x0 = x

        for _ in range(2):
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)

        return x0 + x


class ResBottleneck(nn.Module):

    def __init__(self, input_channels, output_channels, filter_sizes):
        super().__init__()

        if filter_sizes is None:
            filter_sizes = [1, 3, 1]

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.filter_sizes = filter_sizes

        self.conv1 = nn.Conv2d(input_channels, output_channels // 4, (filter_sizes[0], filter_sizes[0]), bias=False)
        self.conv2 = nn.Conv2d(output_channels // 4, output_channels // 4, (filter_sizes[1], filter_sizes[1]),
                               bias=False)
        self.conv3 = nn.Conv2d(output_channels // 4, output_channels, (filter_sizes[2], filter_sizes[2]), bias=False)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.bn2 = nn.BatchNorm2d(output_channels // 4)
        self.bn3 = nn.BatchNorm2d(output_channels)
        self.relu = F.relu

    def forward(self, x):
        x0 = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        out = x0 + x
        out = self.relu(out)

        return out


class Reshape(nn.Module):

    def __init__(self, channel_dimension):
        super(Reshape, self).__init__()
        self.channel_dimension = channel_dimension

    def forward(self, x):
        depth_dimension = (x.numel() / self.channel_dimension) / x.numel()[-2]
        x = x.view(*x.shape[:-1], self.channel_dimension)
        x = x.view(*x.shape[:-2], depth_dimension, self.channel_dimension)

        return x
