import torch
import torch.nn as nn
import torch.nn.functional as F


def build_conv(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, block_options):
        super().__init__()

        kernel_size = block_options.get('kernel_size') if block_options is not None else 3
        stride = block_options.get('stride') if block_options is not None else 1
        padding = block_options.get('padding') if block_options is not None else 1
        batch_size = block_options.get('batch_size') if block_options is not None else 32
        use_group_norm = block_options.get('use_group_norm') if block_options is not None else False

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
    def __init__(self, depth, in_channels, out_channels, block_options):
        super().__init__()

        self.name = 'hourglass'
        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_options = block_options

        self._generate_block(self.depth)

    def _generate_block(self, depth):
        self.add_module('b1_' + str(depth), ConvBlock(self.in_channels, self.out_channels, **self.block_options))
        self.add_module('b2_' + str(depth), ConvBlock(self.in_channels, self.out_channels, **self.block_options))

        if depth > 1:
            self._generate_block(depth - 1)
        else:
            self.add_module('b2+_' + str(depth), ConvBlock(self.in_channels, self.out_channels, **self.block_options))

        self.add_module('b3_' + str(depth), ConvBlock(self.in_channels, self.out_channels, **self.block_options))

    def _forward(self, depth, x):
        up1 = x
        up1 = self._modules['b1_' + str(depth)](up1)

        down1 = F.avg_pool2d(x, 2, stride=2)
        down1 = self._modules['b2_' + str(depth)](down1)

        if depth > 1:
            down2 = self._forward(depth - 1, down1)
        else:
            down2 = down1
            down2 = self._modules['b2+_' + str(depth)](down2)

        down3 = down2
        down3 = self._modules['b3_' + str(depth)](down3)

        up2 = F.interpolate(down3, scale_factor=2, mode='bicubic', align_corners=True)

        return up1 + up2

    def forward(self, x):
        return self._forward(self.depth, x)


class HourGlassFilter(nn.Module):
    def __init__(self,
                 num_stack,
                 depth,
                 in_channel,
                 last_channel,
                 block_options,
                 down_type='conv64',
                 use_sigmoid=True,
                 ):
        super().__init__()

        self.num_stack = num_stack
        self.depth = depth
        self.last_channel = last_channel
        self.down_type = down_type
        self.use_sigmoid = use_sigmoid
        self.block_options = block_options
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padidng=3)

        self.use_group_norm = block_options.get('use_group_norm')

        if not self.use_group_norm:
            self.bn1 = nn.BatchNorm2d(64)
        else:
            self.bn1 = nn.GroupNorm(32, 64)

        if self.down_type == 'conv64':
            self.conv2 = ConvBlock(64, 64, **self.block_options)
            self.down_conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        elif self.down_type == 'conv128':
            self.conv2 = ConvBlock(128, 128, **self.block_options)
            self.down_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        elif self.down_type == 'avg_pool' or self.down_type == 'no_down':
            self.conv2 = ConvBlock(64, 128, **self.block_options)

        self.conv3 = ConvBlock(128, 128, **self.block_options)
        self.conv4 = ConvBlock(128, 256, **self.block_options)

        for stack in range(self.num_stack):
            self.add_module('m' + str(stack), HourGlassNetwork(self.depth, 256, **self.block_options))
            self.add_module('top_m_' + str(stack), ConvBlock(256, 256, **self.block_options))
            self.add_module('conv_last' + str(stack), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))

            if not self.use_group_norm:
                self.add_module('bn_end' + str(stack), nn.BatchNorm2d(256))
            else:
                self.add_module('bn_end' + str(stack), nn.GroupNorm(32, 256))

            self.add_module('l' + str(stack), nn.Conv2d(256, last_channel, kernel_size=1, stride=1, padding=0))

            if stack < self.num_stack - 1:
                self.add_module('bl' + str(stack), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
                self.add_module('al' + str(stack), nn.Conv2d(last_channel, 256, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), True)

        if self.down_type == 'avg_pool':
            x = F.avg_pool2d(self.conv2(x), 2, stride=2)
        elif self.down_type in ['conv64', 'conv128']:
            x = self.conv2(x)
            x = self.down_conv2(x)
        elif self.down_type == 'no_down':
            x = self.conv2(x)

        norm_x = x

        x = self.conv3(x)
        x = self.conv4(x)

        outputs = []

        for stack in range(self.num_stack):
            hourglass = self._modules['m' + str(stack)](x)

            hourglass = self._modules['top_m' + str(stack)](hourglass)
            hourglass = self._modules['conv_last' + str(stack)](hourglass)
            hourglass = self._modules['bn_end' + str(stack)](hourglass)
            hourglass = F.relu(hourglass)

            out1 = self._modules['l' + str(stack)](hourglass)

            if self.use_sigmoid:
                outputs.append(nn.Tanh()(out1))
            else:
                outputs.append(out1)

            if stack < self.num_stack - 1:
                hourglass = self._modules['bl' + str(stack)](hourglass)
                out2 = self._modules['al' + str(stack)](hourglass)
                x = x + hourglass + out2

        return outputs, norm_x



