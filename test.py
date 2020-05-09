# -*- coding: utf-8 -*-
import argparse
import torch
import torchvision.models
import torchvision.transforms as transforms
import glob
import os
from PIL import Image
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prepare_image(image):
    if image.mode != 'RGB':
        image = image.convert("RGB")
    Transform = transforms.Compose([
            transforms.Resize([224,224]),      
            transforms.ToTensor(),
            ])
    image = Transform(image)   
    image = image.unsqueeze(0)
    return image.to(device)

def predict(image, model):
    image = prepare_image(image)
    with torch.no_grad():
        preds = model(image)
    # print(r'Popularity score: %.2f' % preds.item() * 100)
    return '%.0f' % (preds.item() * 100)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='images/')
    config = parser.parse_args()
    image_paths = sorted(glob.glob(os.path.join(config.image_path, '*')))
    model = torchvision.models.resnet50()
    # model.avgpool = nn.AdaptiveAvgPool2d(1) # for any size of the input
    model.fc = torch.nn.Linear(in_features=2048, out_features=1)
    model.load_state_dict(torch.load('/content/Intrinsic-Image-Popularity/model/model-resnet50.pth', map_location=device)) 
    model.eval().to(device)

    if not os.path.isdir(config.image_path):
      print("ERROR: '" + config.image_path + "' is not a directory!")
      return

    count = 0
    max_count = 0

    image_names = []

    for image_path in image_paths:
      file_name, file_extension = os.path.splitext(os.path.split(image_path)[1])
      if (file_extension in ['.png'] and '_' not in file_name):
        max_count += 1
        image_names.append(file_name + file_extension)
    
    if len(image_names) == 0:
      print("ERROR: '" + config.image_path + "' does not contain any images to rename!")
      return
    
    print("Start renaming of " + str(len(image_names)) + " image(s)...")

    for image_name in image_names:
      count += 1

      full_path = os.path.join(config.image_path, image_name)

      image = Image.open(full_path)
      score = predict(image, model)

      os.rename(full_path, os.path.join(config.image_path, str(score) + '_' + image_name))
      
      print(str(count) + '/' + str(max_count))

if __name__ == '__main__':
    main()
