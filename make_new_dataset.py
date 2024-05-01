from PIL import Image
import torch
from torchvision.transforms import ToTensor
from dataset import LidarDataset  # Assuming this is your dataset class
import os
import numpy as np

original_image_dir = "NAPLab-LiDAR/images"
original_label_dir = "NAPLab-LiDAR/labels_yolo_v1.1"

train_image_dir = "CustomDataset/train/images"
train_label_dir = "CustomDataset/train/labels"
val_image_dir = "CustomDataset/val/images"
val_label_dir = "CustomDataset/val/labels"

# Assuming transforms are needed like normalization or ToTensor
transforms = ToTensor()

# Initialize your dataset
dataset = LidarDataset(image_dir=original_image_dir,
                       label_dir=original_label_dir,
                       transforms=transforms)

# Determine split index
val_split = 0.2
num_samples = len(dataset)
num_val = int(num_samples * val_split)
num_train = num_samples - num_val

for idx in range(num_samples):
    crop, adjusted_boxes = dataset[idx]

    # Determine whether the crop goes into training or validation set
    if idx < num_train:
        crop_path = os.path.join(train_image_dir, f'crop_{idx}.png')
        label_path = os.path.join(train_label_dir, f'crop_{idx}.txt')
    else:
        crop_path = os.path.join(val_image_dir, f'crop_{idx}.png')
        label_path = os.path.join(val_label_dir, f'crop_{idx}.txt')
    
    # Save the cropped image

    crop_uint8 = (crop.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    crop_image = Image.fromarray(crop_uint8)
    crop_image.save(crop_path)

    # Save the adjusted bounding boxes
    with open(label_path, 'w') as label_file:
        for box in adjusted_boxes:
            class_id, x_center, y_center, width, height = box.tolist()
            # Format as required by YOLO: class x_center y_center width height
            label_file.write(f'{int(class_id)} {x_center} {y_center} {width} {height}\n')
