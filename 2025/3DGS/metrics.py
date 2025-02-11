import torch
from math import exp
import torch.nn.functional as F

def gaussian(window_size: int,
             sigma: float):
  """
  Generates a list of Tensor values drawn from a gaussian distribution with standard deviation.
  """
  gauss = torch.Tensor([exp(-(x - window_size // 2) **2 / float(2 * sigma ** 2)) for x in range(window_size)])
  return gauss / gauss.sum()

def create_window(window_size: int,
                  channel: int=1):

  # _1d_window : (window_size, 1)
  _1d_window = gaussian(window_size=window_size, sigma=1.5).unsqueeze(1)

  # _2d_window : (window_size, window_size)
  _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)

  # window : (channel, 1, window_size, window_size)
  window = torch.Tensor(_2d_window.expand(channel, 1, window_size, window_size).contiguous())

  return window

def ssim(img1, img2, window_size=11, size_average=True):

  try:
    _, channel, height, width = img1.size()
  except:
    channel, height, width = img1.size()

  window = create_window(window_size=window_size, channel=channel)

  if img1.is_cuda:
    window = window.cuda(img1.get_device())
  window = window.type_as(img1)

  mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
  mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

  sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1.pow(2)
  sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2.pow(2)
  sigma_1_2 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu_1 * mu_2

  C1, C2 = 0.01 **2, 0.03 ** 2

  ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma_1_2 + C2)) / ((mu1.pow(2) + mu2.pow(2) + C1) * (sigma1_sq + sigma2_sq + C2))

  if size_average:
    return ssim_map.mean()
  else:
    return ssim_map.mean(1).mean(1).mean(1)

def psnr(img1, img2):
  mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
  return 20 * torch.log10(1. / torch.sqrt(mse))
