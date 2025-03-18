import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
  """ Turns a 2D input image into a 1D sequence learnable embeding vector.
  Args:
    in_channels (int) - Number of color channels for the input images. Defaults to 3.
    patch_size (int) - Size of patches to convert input images into. Defaults to 16.
    embedding_dim (int) - Size of embedding to turn image into. Defaults to 768.
  """
  def __init__(self, in_channels: int=3, patch_size: int=16, embedding_dim: int=768):
    super().__init__()

    self.in_channels = in_channels
    self.patch_size = patch_size
    self.embedding_dim = embedding_dim

    self.patchify = nn.Conv2d(in_channels=self.in_channels,
                              out_channels=self.embedding_dim,
                              kernel_size=self.patch_size,
                              stride=self.patch_size,
                              padding=0)

    self.flatten = nn.Flatten(start_dim=2, end_dim=3)

  def forward(self, x: torch.Tensor):
    image_resolution = x.shape[-1]
    assert image_resolution % self.patch_size == 0, \
      f"Input size must be divisible by patch size, image size : {image_resolution}, patch size : {self.patch_size}"

    x_patched = self.patchify(x)
    x_flattened = self.flatten(x_patched)

    return x_flattened.permute(0, 2, 1) # [batch_size, patch_size ** 2 * channel, embedding_dim] -> [batch_size, embedding_dim, patch_size ** 2 ]
