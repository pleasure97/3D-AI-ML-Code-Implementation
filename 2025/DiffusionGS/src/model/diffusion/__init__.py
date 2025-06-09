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

        self.pipe = DiffusionPipeline.from_pretrained(self.config.model_id).to(device)
        self.pipe.to(device)
        self.pipe.enable_model_cpu_offload()
        self.scheduler = DDIMScheduler.from_config(
            self.pipe.scheduler.config,
            rescale_betas_zero_snr=True,
            timestep_spacing=self.config.timestep_spacing,
            prediction_type="sample")  # prediction type - x_0 prediction instead of epsilon prediction

        self.total_timesteps = self.config.total_timesteps
        self.num_timesteps = self.config.num_timesteps

    @torch.no_grad
    def generate(self, source_image: Float[Tensor, "batch channel height width"], timestep: int) \
            -> list[Float[Tensor, "batch channel height width"]]:
        # Load and transform image
        original_sample = source_image.to(device=self.device)
        batch_size = original_sample.shape[0]
        timesteps = torch.full((batch_size,), timestep, device=self.device)

        with torch.cuda.amp.autocast():
            # Generate noise
            noise = torch.randn_like(original_sample)
            noisy_image = self.scheduler.add_noise(original_sample, noise, timesteps)

        del original_sample, noise
        torch.cuda.empty_cache()

        return noisy_image

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
