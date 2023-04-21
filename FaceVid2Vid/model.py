import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from block import DownBlock2D, DownBlock3D, UpBlock2D, UpBlock3D, ResBlock2D, ResBottleneck, Reshape


class AppearanceFeatureExtractor(nn.Module):

    def __init__(self,
                 input_channel,
                 channel_sizes,
                 conv_filter_sizes,
                 num_res_block):
        super().__init__()

        if channel_sizes is None:
            channel_sizes = [64, 128, 256, 512, 32]
        if not isinstance(math.sqrt(channel_sizes[4], channel_sizes[5]), int):
            raise ValueError('check channel sizes so that channel sizes of ResBlock2D be int.')
        if conv_filter_sizes is None:
            conv_filter_sizes = [7, 1]
        if num_res_block is None:
            num_res_block = 6

        self.input_channel = input_channel
        self.channel_sizes = channel_sizes
        self.conv_filter_sizes = conv_filter_sizes

        self.conv1 = nn.Conv2d(input_channel, channel_sizes[0], (conv_filter_sizes[0], conv_filter_sizes[0]),
                               bias=False)
        self.bn = nn.BatchNorm2d(channel_sizes[0])
        self.relu = F.relu

        self.down1 = DownBlock2D(channel_sizes[0], channel_sizes[1])
        self.down2 = DownBlock2D(channel_sizes[1], channel_sizes[2])

        self.conv2 = nn.Conv2d(channel_sizes[2], channel_sizes[3], (conv_filter_sizes[1], conv_filter_sizes[1]))

        self.reshape = Reshape(channel_sizes[5])
        self.res = ResBlock2D(channel_sizes[5])
        self.num_res_block = num_res_block

    def forward(self, s):
        s = self.conv1(s)
        s = self.bn(s)
        s = self.relu(s)

        s = self.down1(s)
        s = self.down2(s)

        s = self.conv2(s)
        s = self.reshape(s)

        for i in range(self.num_res_block):
            s = self.res(s)

        return s


class CanonicalKeypointDetector(nn.Module):

    def __init__(self,
                 input_channel,
                 down_channels,
                 up_channels,
                 conv_channels,
                 conv_filters,
                 reshape_channel):
        super().__init__()

        if down_channels is None:
            down_channels = [64, 128, 256, 512, 1024]
        if up_channels is None:
            up_channels = [512, 256, 128, 64, 32]
        if conv_channels is None:
            conv_channels = [16384, 20, 180]
        if conv_filters is None:
            conv_filters = [1, 7, 7]
        if reshape_channel is None:
            reshape_channel = 1024

        self.input_channel = input_channel
        self.down_channels = down_channels
        self.up_channels = up_channels
        self.conv_channels = conv_channels
        self.reshape_channel = reshape_channel

        self.down_layers = nn.Sequential(
            *[DownBlock2D(input_channel, down_channels[i]) if i == 0
              else DownBlock2D(down_channels[i], down_channels[i + 1])
              for i in range(len(down_channels) - 1)]
        )

        self.conv1 = nn.Conv2d(down_channels[4], conv_channels[0], (conv_filters[0], conv_filters[0]))

        self.reshape = Reshape(reshape_channel)

        self.up_layers = nn.Sequential(
            *[UpBlock3D(reshape_channel, up_channels[i]) if i == 0
              else UpBlock3D(up_channels[i], up_channels[i + 1])
              for i in range(len(down_channels) - 1)]
        )

        self.conv2 = nn.Conv3d(up_channels[4], conv_channels[1], (conv_filters[1], conv_filters[1], conv_filters[1]))
        self.conv3 = nn.Conv3d(up_channels[4], conv_channels[2], (conv_filters[2], conv_filters[2], conv_filters[2]))

    def forward(self, x):
        x = self.down_layers(x)

        x = self.conv1(x)
        x = self.reshape(x)

        x = self.up_layers(x)

        keypoints = self.conv2(x)
        jacobians = self.conv3(x)

        return keypoints, jacobians


class HeadPoseOrExpressionDeformationEstimator(nn.Module):

    def __init__(self,
                 input_channel,
                 res_channels,
                 num_res_layers,
                 fc_channels,
                 conv_channel_size,
                 conv_filter_size,
                 pool_size):

        super().__init__()

        if res_channels is None:
            res_channels = [256, 512, 1024, 2048]
        if num_res_layers is None:
            num_res_layers = [3, 4, 6, 3]
        if fc_channels is None:
            fc_channels = {"yaw": 66, "pitch": 66, "roll": 66, "t": 3, "delta": 60}
        if conv_channel_size is None:
            conv_channel_size = 64
        if conv_filter_size is None:
            conv_filter_size = 7
        if pool_size is None:
            pool_size = 7

        self.conv = nn.Conv2d(input_channel, conv_channel_size, (conv_filter_size, conv_filter_size))
        self.bn = nn.BatchNorm2d(conv_channel_size)
        self.relu = F.relu

        self.conv1 = nn.Sequential(self.conv, self.bn, self.relu)

        self.res1 = nn.Sequential(*[ResBottleneck(res_channels[0]) for _ in range(num_res_layers[0])])
        self.res2 = nn.Sequential(*[ResBottleneck(res_channels[1]) for _ in range(num_res_layers[1])])
        self.res3 = nn.Sequential(*[ResBottleneck(res_channels[2]) for _ in range(num_res_layers[2])])
        self.res4 = nn.Sequential(*[ResBottleneck(res_channels[3]) for _ in range(num_res_layers[3])])

        self.pool = nn.AvgPool2d(pool_size)

        self.yaw = nn.Linear(res_channels[-1], fc_channels["yaw"])
        self.pitch = nn.Linear(res_channels[-1], fc_channels["pitch"])
        self.roll = nn.Linear(res_channels[-1], fc_channels["roll"])
        self.t = nn.Linear(res_channels[-1], fc_channels["t"])
        self.delta = nn.Linear(res_channels[-1], fc_channels["delta"])

        self.fc_channels = fc_channels

        self.idx_tensor = torch.FloatTensor(list(range(self.fc_channels["yaw"]))).unsqueeze(0).cuda()

    def forward(self, x):
        x = self.conv1(x)

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)

        x = self.pool(x)

        x = torch.mean(x, (2, 3))

        yaw, pitch, roll, t, delta = self.yaw(x), self.pitch(x), self.roll(x), self.t(x), self.delta(x)
        yaw, pitch, roll = torch.softmax(yaw, dim=1), torch.softmax(pitch, dim=1), torch.softmax(roll, dim=1)

        yaw = (yaw * self.idx_tensor).sum(dim=1)
        pitch = (pitch * self.idx_tensor).sum(dim=1)
        roll = (roll * self.idx_tensor).sum(dim=1)

        yaw = (yaw - self.fc_channels["yaw"] // 2) * 3 * np.pi / 180.
        pitch = (pitch - self.fc_channels["pitch"] // 2) * 3 * np.pi / 180.
        roll = (roll - self.fc_channels["roll"] // 2) * 3 * np.pi / 180.
        delta = delta.view(x.shape[0], -1, 3)

        return yaw, pitch, roll, t, delta


class MotionFieldEstimator(nn.Module):

    def __init__(self,
                 input_channel,
                 down_channels,
                 up_channels,
                 reshape_channel,
                 conv_filter_sizes,
                 conv_channels):

        super().__init__()

        if down_channels is None:
            down_channels = [64, 128, 256, 512, 1024]
        if up_channels is None:
            up_channels = [512, 256, 128, 64, 32]
        if reshape_channel is None:
            reshape_channel = 2192
        if conv_filter_sizes is None:
            conv_filter_sizes = [7, 7]
        if conv_channels is None:
            conv_channels = [21, 1]

        self.down_layers = nn.Sequential(
            *[DownBlock3D(input_channel, down_channels[i]) if i == 0
              else DownBlock3D(down_channels[i], down_channels[i + 1])
              for i in range(len(down_channels) - 1)]
        )

        self.up_layers = nn.Sequential(
            *[UpBlock3D(down_channels[-1], up_channels[i]) if i == 0
              else UpBlock3D(up_channels[i], up_channels[i + 1])
              for i in range(len(up_channels) - 1)]
        )

        self.conv_mask = nn.Conv3d(up_channels[-1],
                                   conv_channels[0],
                                   (conv_filter_sizes[0], conv_filter_sizes[0], conv_filter_sizes[0]))

        self.reshape_occlusion = Reshape(reshape_channel)

        self.conv_occlusion = nn.Conv2d(reshape_channel, conv_channels[1], (conv_filter_sizes[1], conv_filter_sizes[1]))

    def forward(self, x):
        x = self.down_layers(x)

        x = self.up_layers(x)

        mask = torch.softmax(self.conv_mask(x), dim=2)

        reshape_occlusion = self.reshape_occlusion(x)
        occlusion = self.conv_occlusion(reshape_occlusion)

        return mask, occlusion


class Generator(nn.Module):

    def __init__(self,
                 input_channel,
                 occlusion,
                 reshape_channel,
                 conv_channels,
                 conv_filters,
                 res_channel,
                 num_res_layers,
                 up_channels):

        super().__init__()

        self.occlusion = occlusion
        if reshape_channel is None:
            reshape_channel = 512
        if conv_channels is None:
            conv_channels = [256, 256, 3]
        if conv_filters is None:
            conv_filters = [3, 1, 7]
        if res_channel is None:
            res_channel = 256
        if num_res_layers is None:
            num_res_layers = 6
        if up_channels is None:
            up_channels = [128, 64]

        self.reshape = Reshape(reshape_channel)
        self.conv1 = nn.Sequential(
            nn.Conv2d(reshape_channel, conv_channels[0], (conv_filters[0], conv_filters[0])),
            nn.BatchNorm2d(conv_channels[0]),
            nn.ReLU())
        self.conv2 = nn.Conv2d(conv_channels[0], conv_channels[1], (conv_filters[1], conv_filters[1]))
        self.res_layers = nn.Sequential(*[ResBlock2D(res_channel) for _ in range(num_res_layers)])
        self.up_layers = nn.Sequential(
            UpBlock2D(res_channel, up_channels[0]),
            UpBlock2D(up_channels[0], up_channels[1]))
        self.conv3 = nn.Conv2d(up_channels[1], conv_channels[2], (conv_filters[2], conv_filters[2]))

    def forward(self, x):
        x = self.reshape(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.kron(x, self.occlusion)
        x = self.res_layers(x)
        x = self.up_layers(x)
        y = self.conv3(x)

        return y
