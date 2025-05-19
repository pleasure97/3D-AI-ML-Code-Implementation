from jaxtyping import Float
from torch import Tensor
from einops import rearrange
import torch
from PIL import Image
import numpy as np
from src.preprocess.types import BatchedViews, BatchedExample
from rembg import remove


def rescale(
        image: Float[Tensor, "3 height width"],
        shape: tuple[int, int]) -> Float[Tensor, "3 height width"]:
    height, width = shape
    image_new = (image * 255).clip(min=0, max=255).type(torch.uint8)
    image_new = rearrange(image_new, "c h w -> h w c").detach().cpu().numpy()
    image_new = Image.fromarray(image_new)
    image_new = image_new.resize((width, height), Image.LANCZOS)
    image_new = np.array(image_new) / 255
    # TODO - Check Device
    image_new = torch.tensor(image_new, dtype=image.dtype, device=image.device)
    image_new = rearrange(image_new, "h w c -> c h w")
    return image_new


def center_crop(
        images: Float[Tensor, "*#batch channel height width"],
        intrinsics: Float[Tensor, "*batch 3 3"],
        shape: tuple[int, int]
) -> tuple[Float[Tensor, "*#batch channel height_out width_out"], Float[Tensor, "*#batch 3 3"]]:
    *_, height_in, width_in = images.shape
    height_out, width_out = shape

    row = (height_in - height_out) // 2
    col = (width_in - width_out) // 2

    # Center crop the image.
    images = images[..., :, row : row + height_out, col : col + width_out]

    # Adjust the intrinsics to account for the cropping.
    intrinsics = intrinsics.clone()
    intrinsics[..., 0, 0] *= width_in / width_out
    intrinsics[..., 1, 1] *= height_in / height_out

    return images, intrinsics

def crop_and_scale(
        images: Float[Tensor, "*#batch channel height width"],
        intrinsics: Float[Tensor, "*#batch 3 3"],
        shape: tuple[int, int]
) -> tuple[Float[Tensor, "*#batch channel height_out width_out"], Float[Tensor, "*#batch 3 3"]]:
    *_, height_in, width_in = images.shape
    height_out, width_out = shape
    assert height_out <= height_in and width_out <= width_in

    scale_factor = max(height_out / height_in, width_out / width_in)
    scaled_height = round(height_in * scale_factor)
    scaled_width = round(width_in * scale_factor)
    assert scaled_height >= height_out and scaled_width >= width_out

    *batch, channel, height, width = images.shape
    images = images.reshape(-1, channel, height, width)
    images = torch.stack([rescale(image, (scaled_height, scaled_width)) for image in images])
    images = images.reshape(*batch, channel, scaled_height, scaled_width)

    return center_crop(images, intrinsics, shape)

def crop_views(views: BatchedViews, shape: tuple[int, int]) -> BatchedViews:
    images, intrinsics = crop_and_scale(views["image"], views["intrinsics"], shape)
    return {**views, "images": images, "intrinsics": intrinsics}

def crop_example(example: BatchedExample, shape: tuple[int, int]) -> BatchedExample:
    return {**example, "source": crop_views(example["source"], shape), "target": crop_views(example["target"], shape)}

def remove_background(images: Float[Tensor, "*#batch channel height width"]):
    *batch, channel, height, width = images.shape
    images = images.reshape(-1, channel, height, width)
    images = torch.stack([remove(image) for image in images])
    images = images.reshape(*batch, channel, height, width)

    return images