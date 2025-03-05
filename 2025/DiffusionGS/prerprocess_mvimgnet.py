from PIL import Image 
import torchvision.transforms as transforms
import os 

def crop_image(image_dir: str, crop_size: tuple=(200, 200)):

  transform = transforms.Compose([transforms.CenterCrop(crop_size)])

  for dirpath, dirnames, filenames in os.walk(image_dir):
      if "images" in dirpath: 
          for filename in filenames:
              if filename.endswith(".jpg"):
                  img_path = os.path.join(dirpath, filename)

                  img = Image.open(img_path)

                  cropped_img = transform(img)

                  output_path = os.path.join(dirpath, f"cropped_{filename}")
                  print(f"output path is  : {output_path}")
                  cropped_img.save(output_path)
