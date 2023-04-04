import torch
import torch.nn as nn
import torch.nn.functional as F


class DownBlock2D(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(DownBlock2D, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.conv = nn.Conv2d(input_channels, output_channels, 3, bias=False)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = F.relu()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class UpBlock2D(nn.Module):

    def __init__(self, num_channels):
        super(DownBlock2D, self).__init__()
        self.num_channels = num_channels

        self.conv = nn.Conv2d(num_channels, num_channels, 3, bias=False)
        self.bn = nn.BatchNorm2d(num_channels)
        self.relu = F.relu()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class ResBlock2D(nn.Module):

    def __init__(self, num_channels):
        super(ResBlock2D, self).__init__()
        self.num_channels = num_channels

        self.conv = nn.Conv2d(num_channels, num_channels, 3, bias=False)
        self.bn = nn.BatchNorm2d(num_channels)
        self.relu = F.relu()

    def forward(self, x):
        x0 = x

        for _ in range(2):
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)

        return x0 + x


class ResBottleneck(nn.Module):

    def __init__(self, num_channels):
        super(ResBlock2D, self).__init__()

        self.conv1 = nn.Conv2d(num_channels, num_channels // 4, 1, bias=False)
        self.conv2 = nn.Conv2d(num_channels // 4, num_channels // 16, 3, bias=False)
        self.conv3 = nn.Conv2d(num_channels // 16, num_channels // 16, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels // 4)
        self.bn3 = nn.BatchNorm2d(num_channels // 16)
        self.relu = F.relu()

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


class AppearanceFeatureExtractor(nn.Module):

    def __init__(self, num_channels):
        super(AppearanceFeatureExtractor, self).__init__()

        self.num_channels = num_channels
        self.conv1 = nn.Conv2d(num_channels, 64, 7, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = F.relu()

        self.conv2 = nn.Conv2d(256, 512, 1)


    def forward(self):
        pass


class CanonicalKeypointDetector(nn.Module):

    def __init__(self):
        pass

    def forward(self):
        pass


class HeadPoseEstimator(nn.Module):

    def __init__(self):
        pass

    def forward(self):
        pass


class ExpressionDeformationEstimator(nn.Module):

    def __init__(self):
        pass

    def forward(self):
        pass


class MotionFieldEstimator(nn.Module):

    def __init__(self):
        pass

    def forward(self):
        pass


class Generator(nn.Module):

    def __init__(self):
        pass

    def forward(self):
        pass
