import argparse
import os
from os import listdir
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


def green_ch(path_dir, path_target):
  # pytorch provides a function to convert PIL images to tensors.
  pil2tensor = transforms.ToTensor()
  tensor2pil = transforms.ToPILImage()
  for images in os.listdir(path_dir): 
        #  read each image
        path_source = path_dir+images
        print(path_source)
        path_target = path_target
        
        if images.endswith(".jpg") or images.endswith(".png") or images.endswith(".JPG"):
          pil_image = Image.open(path_source)
          rgb_image = pil2tensor(pil_image)
          image_copy = rgb_image.clone()

          # Extract the green channel and enhance it 
          image_copy[1] = image_copy[1].mul(2.0).clamp(0.0, 1.0)
        
          # Save the green-channel image to the destination folder 
          img_g = tensor2pil(image_copy)
          img_g.save(os.path.join(path_target+images))
          print(path_target+images)


def main():
    parser = argparse.ArgumentParser(description="Extract and enhance green channel of images.")
    parser.add_argument('-n', '--num-processes', type=int, default=8, help='number of processes to use')
    parser.add_argument("--image-folder", type=str, required=True, help="Path to the source image directory")
    parser.add_argument("--output-folder", type=str, required=True, help="Path to the target directory for enhanced images")
    args = parser.parse_args()

    green_ch(args.image_folder, args.output_folder)

    print("Green channel extraction and enhancement completed successfully!")


if __name__ == "__main__":
    main()