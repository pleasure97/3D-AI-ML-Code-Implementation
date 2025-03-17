from PIL import Image
import torchvision.transforms as transforms
import os

def crop_image(image_dir: str, crop_size: tuple=(800, 400)):

  transform = transforms.Compose([transforms.CenterCrop(crop_size)])

  for dirpath, dirnames, filenames in os.walk(image_dir):
        for filename in filenames:
            if filename.endswith(".jpg"):
                img_path = os.path.join(dirpath, filename)
                cropped_dir = os.path.join(dirpath, "cropped")
                os.makedirs(cropped_dir, exist_ok=True)

                img = Image.open(img_path)

                cropped_img = transform(img)

                output_path = os.path.join(cropped_dir, f"cropped_{filename}")
                print(f"output path is  : {output_path}")
                cropped_img.save(output_path)
