from dataclasses import dataclass
from diffusers import DiffusionPipeline, DDIMScheduler
from torch import Tensor
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from jaxtyping import Float


@dataclass
class DiffusionGeneratorConfig:
    model_id: str  # "ptx0/pseudo-journey-v2"
    timestep_spacing: str  # "trailing"
    total_timesteps: int  # 101
    num_timesteps: int  # 1


class DiffusionGenerator(DiffusionGeneratorConfig):
    def __init__(self, config: DiffusionGeneratorConfig, device="cuda", dtype=torch.float16):
        self.config = config

        self.device = device
        self.dtype = dtype
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.pipe = DiffusionPipeline.from_pretrained(self.config.model_id, dtype=dtype).to(device)
        self.scheduler = DDIMScheduler.from_config(
            self.pipe.scheduler.config,
            rescale_betas_zero_snr=True,
            timestep_spacing=self.config.timestep_spacing,
            prediction_type="sample")  # prediction type - x_0 prediction instead of epsilon prediction

        self.total_timesteps = self.config.total_timesteps
        self.num_timesteps = self.config.num_timesteps

    def generate(self, source_image: Float[Tensor, "batch channel height width"]) \
            -> list[Float[Tensor, "batch channel height width"]]:
        # Load and transform image
        timesteps = torch.arange(1, self.total_timesteps, self.num_timesteps)
        image = Image.open(source_image).convert("RGB")
        original_sample = self.transform(image).unsqueeze(0).to(self.device, dtype=self.dtype)

        # Generate noise
        noise = torch.randn_like(original_sample, dypte=self.dtype, device=self.device)

        # Apply noise at specified timesteps
        noisy_images = []
        for timestep in timesteps:
            noisy_image = self.scheduler.add_noise(original_sample, noise, timestep.unsqueeze(0))
            noisy_images.append(noisy_image)

        return noisy_images

    @staticmethod
    def visualize(self,
                  source_image: Float[Tensor, "batch channel height width"],
                  noisy_images: list[Float[Tensor, "batch channel height width"]]):
        timesteps = torch.arange(1, self.total_timesteps, self.num_timesteps)

        to_pil = transforms.ToPILImage()
        num_images = len(noisy_images) + 1
        fig, axs = plt.subplots(1, num_images, figsize=(3 * num_images, 3))

        # Original image
        axs[0].imshow(to_pil(source_image.squeeze(0).float().cpu().clamp(0, 1)))
        axs[0].set_title("Original")
        axs[0].axis("off")

        # Noisy images
        for i, (sample, t) in enumerate(zip(noisy_images, timesteps)):
            axs[i + 1].imshow(to_pil(sample.squeeze(0).float().cpu().clamp(0, 1)))
            axs[i + 1].set_title(f"Timestep {t.item()}")
            axs[i + 1].axis("off")

        plt.tight_layout()
        plt.show()
