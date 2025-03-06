from PIL import Image
import torchvision.transforms as transforms
import os

def crop_image(image_dir: str, crop_size: tuple=(800, 400)):

  transform = transforms.Compose([transforms.CenterCrop(crop_size)])

  for dirpath, dirnames, filenames in os.walk(image_dir):
        for filename in filenames:
            if filename.endswith(".jpg"):
                img_path = os.path.join(dirpath, filename)

                img = Image.open(img_path)

                cropped_img = transform(img)

                output_path = os.path.join(dirpath, f"cropped_{filename}")
                print(f"output path is  : {output_path}")
                cropped_img.save(output_path)
                
from rembg import remove
from PIL import Image
import io

def remove_background(image_dir: str):
    for dirpath, _, filenames in os.walk(image_dir):
        for filename in filenames:
            if filename.startswith("cropped_") and filename.endswith(".jpg"):
                img_path = os.path.join(dirpath, filename)

                with open(img_path, 'rb') as input_file:
                    input_data = input_file.read()
                    output_data = remove(input_data) 

                output_image = Image.open(io.BytesIO(output_data))

                output_filename = filename.replace(".jpg", "_no_bg.png") 
                output_path = os.path.join(dirpath, output_filename)

                output_image.save(output_path)
                print(f"Saved background-removed image: {output_path}")
from rembg import remove
from PIL import Image
import io

def remove_background(image_dir: str, save_as_png: bool=False):
    for dirpath, _, filenames in os.walk(image_dir):
        for filename in filenames:
            if filename.startswith("cropped_") and filename.endswith(".jpg"):
                img_path = os.path.join(dirpath, filename)

                with open(img_path, 'rb') as input_file:
                    input_data = input_file.read()
                    output_data = remove(input_data, 
                                         alpha_matting=True,
                                         alpha_matting_foreground_threshold=240,
                                         alpha_matting_background_threshold=10, 
                                         alpha_matting_erode_size=5) 

                output_image = Image.open(io.BytesIO(output_data))

                if save_as_png:
                    output_filename = filename.replace(".jpg", "_no_bg.png")
                    output_path = os.path.join(dirpath, output_filename)
                    output_image.save(output_path, "PNG")
                else:
                    output_filename = filename.replace(".jpg", "_no_bg.jpg")
                    output_path = os.path.join(dirpath, output_filename)

                    rgb_image = Image.new("RGB", output_image.size, (255, 255, 255))
                    rgb_image.paste(output_image, mask=output_image.split()[3]) 
                    rgb_image.save(output_path, "JPEG", quality=95)
                print(f"Saved background-removed image: {output_path}")
