from typing import TypeVar, Generic
import torch
import torchvision
from torch import nn
from torch import functional as F

T = TypeVar("T")
class BaseLoss(nn.Module, Generic[T]):
    config: T

    def __init__(self, config) -> None:
        super().__init__()
        self.config: T = config
        self.name = self.config.name

# Source code - https://gist.github.com/alper111/
class VGGLoss(nn.Module):
    def __init__(self, resize=True):
        super().__init__()
        blocks = []
        VGG16 = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
        blocks.append(VGG16.features[:4].eval())
        blocks.append(VGG16.features[4:9].eval())
        blocks.append(VGG16.features[9:16].eval())
        blocks.append(VGG16.features[16:23].eval())

        for block in blocks:
            for param in block.parameters():
                param.requires_grad = False

        self.blocks = nn.ModuleList(blocks)
        self.transform = F.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, source, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        print("source.shape =", source.shape)
        if source.dim() == 4:
            source = source.unsqueeze(2)
            target = target.unsqueeze(2)

        source = source.flatten(0, 1)
        target = target.flatten(0, 1)
        if source.shape[1] != 3:
            source = source.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        source = (source - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            source = self.transform(source, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.
        x = source
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += F.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += F.l1_loss(gram_x, gram_y)
        return loss
