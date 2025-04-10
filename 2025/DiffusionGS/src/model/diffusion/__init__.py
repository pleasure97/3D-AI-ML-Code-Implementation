from diffusers import DiffusionPipeline, DDIMScheduler
import torch
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

class NoisyImageGenerator:
    def __init__(self, model_id="ptx0/pseudo-journey-v2", device="cuda", dtype=torch.float16):
        self.device = device
        self.dtype = dtype
        self.pipe = DiffusionPipeline.from_pretrained(model_id, dtype=dtype).to(device)
        self.scheduler = DDIMScheduler.from_config(
            self.pipe.scheduler.config,
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing")
        self.transform = transforms.Compose([transforms.ToTensor()])

    def generate(self, image_path: Path, num_timesteps: int):
        # Load and transform image
        timesteps = torch.arange(1, 101, num_timesteps)
        image = Image.open(image_path).convert("RGB")
        original_sample = self.transform(image).unsqueeze(0).to(self.device, dtype=self.dtype)

        # Generate noise
        noise = torch.randn_like(original_sample, dypte=self.dtype, device=self.device)

        # Apply noise at specified timesteps
        noisy_samples = []
        for timestep in timesteps:
            noisy_sample = self.scheduler.add_noise(original_sample, noise, timestep.unsqueeze(0))
            noisy_samples.append(noisy_sample)

        return original_sample, noisy_samples

    def visualize(self, original_sample, noisy_samples, num_timesteps):
        timesteps = torch.arange(1, 101, num_timesteps)

        to_pil = transforms.ToPILImage()
        num_images = len(noisy_samples) + 1
        fig, axs = plt.subplots(1, num_images, figsize=(3 * num_images, 3))

        # Original image
        axs[0].imshow(to_pil(original_sample.squeeze(0).float().cpu().clamp(0, 1)))
        axs[0].set_title("Original")
        axs[0].axis("off")

        # Noisy images
        for i, (sample, t) in enumerate(zip(noisy_samples, timesteps)):
            axs[i + 1].imshow(to_pil(sample.squeeze(0).float().cpu().clamp(0, 1)))
            axs[i + 1].set_title(f"Timestep {t.item()}")
            axs[i + 1].axis("off")

        plt.tight_layout()
        plt.show()
