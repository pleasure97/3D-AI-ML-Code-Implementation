import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

def get_point_distribution_loss(rays_o, rays_d, k:int, sigma_0: float=0.5):
  pass

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
        param.requires_grad=False

    self.blocks = nn.ModuleList(blocks)
    self.transform = F.interpolate
    self.resize = resize
    self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
    self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

  def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
    if input.shape[1] != 3:
      input = input.repeat(1, 3, 1, 1)
      target = target.repeat(1, 3, 1, 1)
    input = (input - self.mean) / self.std
    target = (target - self.mean) / self.std
    if self.resize:
      input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
      target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
    loss = 0.
    x = input
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

def get_denoising_loss(source: torch.Tensor, target: torch.Tensor, hyperparameter: float=0.8):
  L2_Loss = nn.MSELoss()
  VGG_Loss = VGGLoss()
  return L2_Loss(source, target) + hyperparameter * VGG_Loss(source, target)
