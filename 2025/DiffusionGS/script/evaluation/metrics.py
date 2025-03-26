import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

def get_psnr(clean_image: torch.Tensor, noisy_image: torch.Tensor, max_value=1.) -> float:
    MSE = F.mse_loss(clean_image, noisy_image)
    if MSE == 0:
        return float('inf')

    psnr = 10 * torch.log10(max_value ** 2 / MSE)
    return psnr.item()

class LPIPS(nn.Module):
    def __init__(self, device: torch.device, model: str = 'vgg'):
        super().__init__()

        self.device = device

        if model == 'vgg':
            self.feature_extractor = torchvision.models.vgg16(pretrained=True).features.to(device)
            self.layer_ids = [3, 8, 15, 22]  # Feature Maps of VGG16
        elif model == 'alex':
            self.feature_extractor = torchvision.models.alexnet(pretrained=True).features.to(device)
            self.layer_ids = [2, 5, 8, 10]
        else:
            raise ValueError('Not Supported Model.')

        self.register_buffer('weights', torch.tensor([0.1, 0.2, 0.4, 0.3], device=device))

        # Make model's paramters untrainable
        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = False

    def extract_features(self, x):
        features = []
        for i, layer in enumerate(self.feature_extractor):
            x = layer(x)
            if i in self.layer_ids:
                features.append(x)
        return features

    def forward(self, image1: torch.Tensor, image2: torch.Tensor):
        image1, image2 = image1.to(self.device), image2.to(self.device)
        features1, features2 = self.extract_features(image1), self.extract_features(image2)
        loss = 0.

        for i in range(len(features1)):
            diff = (features1[i] - features2[i]) ** 2
            loss += self.weights[i] * diff.mean()

        return loss


def gaussian_window(window_size: int, sigma: float, device: torch.device) -> torch.Tensor:
  coordinates = torch.arange(window_size, device=device).float() - window_size // 2
  gaussian = torch.exp(-(coordinates ** 2) / (2 * sigma ** 2))
  gaussian /= gaussian.sum()
  return gaussian # [window_size]

def create_window(window_size: int, channel: int, device: torch.device) -> torch.Tensor:
  _1D_window = gaussian_window(window_size, 1.5, device).unsqueeze(1) # [window_size, 1]
  _2D_window = _1D_window @ _1D_window.T # [window_size, window_size]
  window = _2D_window.expand(channel, 1, window_size, window_size)  # [channel, 1, window_size, window_size)
  return window

def ssim(image1: torch.Tensor, image2: torch.Tensor, window_size: int=11, C1: float=0.01**2, C2: float=0.03**2) -> float:
  channel = image1.shape[1] # Check image1's shape is [BATCH_SIZE, CHANNELS, HEIGHT, WIDTH]
  window = create_window(window_size, channel, image1.device)

  mu1 = F.conv2d(image1, window, padding=window_size//2, groups=channel)
  mu2 = F.conv2d(image2, window, padding=window_size//2, groups=channel)

  mu1_square, mu2_square, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
  sigma1_square = F.conv2d(image1 ** 2, window, padding=window_size//2, groups=channel) - mu1_square
  sigma2_square = F.conv2d(image2 ** 2, window, padding=window_size//2, groups=channel) - mu2_square
  sigma1_sigma2 = F.conv2d(image1 * image2, window, padding=window_size//2, groups=channel) - mu1_mu2

  # luminance - Average of Pixel Coordinates
  luminance = (2 * mu1_mu2 + C1) / (mu1_square + mu2_square + C1)

  # contrast - Standard Deviation of Pixel Coordinates
  contrast = (2 * sigma1_sigma2 + C2) / (sigma1_square + sigma2_square + C2)

  # structure comparison - Correlation between Normalized Images
  C3 = C2 / 2
  structure = (sigma1_sigma2 + C3) / (sigma1_square.sqrt() * sigma2_square.sqrt() + C3)

  ssim_map = luminance * contrast * structure
  return ssim_map.mean()

def get_ssim(image1: torch.Tensor, image2: torch.Tensor) -> float:
  return 1 - ssim(image1, image2)