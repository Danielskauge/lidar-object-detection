from PIL import Image
import torch
from torchvision.transforms import ToTensor
from dataset import LidarDataset  # Assuming this is your dataset class
import os
import numpy as np
from typing import Tuple, List
import torchvision.transforms.v2 as v2
import albumentations as A
from torchvision import tv_tensors
from albumentations.pytorch.transforms import ToTensorV2
import random
import shutil

def generate_augmented_versions(image: torch.Tensor, boxes: List[List[float]], multiplier: int, augment_transforms: v2.Compose) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
    augmented_crops = []
    augmented_targets = []

    canvas_size=image.shape[:2]
    boxes = boxes[:][2:]
    boxes = tv_tensors.BoundingBoxes(boxes, format="CXCYWH", canvas_size=canvas_size)   

    for _ in range(multiplier):

        target = {
            "boxes": boxes,
            "labels": torch.arange(boxes.shape[0])
        }
        aug_crop, aug_boxes = augment_transforms(image, boxes)

        augmented_crops.append(aug_crop)
        augmented_targets.append(aug_boxes)
    return augmented_crops, augmented_targets

if __name__ == "__main__":

    common_transform = v2.ToTensor()

    augment_transforms = v2.Compose([
        v2.RandomHorizontalFlip(p=0.8),
        v2.RandomResizedCrop(size=1, scale=(0.8, 1.0)),
        v2.RandomRotation(degrees=(-10, 10)),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
        ToTensor()
    ])

    original_image_dir = "NAPLab-LiDAR/images"
    original_label_dir = "NAPLab-LiDAR/labels_yolo_v1.1"

    train_image_dir = "BalancedDataset/train/images"
    train_label_dir = "BalancedDataset/train/labels"
    val_image_dir = "BalancedDataset/val/images"
    val_label_dir = "BalancedDataset/val/labels"

    # Clear existing files in the training and validation directories
    for folder in [train_image_dir, train_label_dir, val_image_dir, val_label_dir]:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

    # Initialize your dataset
    dataset = LidarDataset(image_dir=original_image_dir,
                        label_dir=original_label_dir,
                        transforms=ToTensor())

    class_names= {0: 'car', 1: 'truck', 2: 'bus', 3: 'motorcycle', 4: 'bicycle', 5: 'scooter', 6: 'person', 7: 'rider'}
    #class_multipliers = {'car': 1, 'truck': 1, 'bus': 1, 'motorcycle': 1, 'bicycle': 1, 'scooter': 1, 'person': 1, 'rider': 1}
    class_multipliers = {'car': 1, 'truck': 5, 'bus': 5, 'motorcycle': 1, 'bicycle': 3, 'scooter': 5, 'person': 1, 'rider': 3}

    all_crops = []
    all_targets = []

    num_original_crops = len(dataset)

    for idx in range(num_original_crops):
        print('id: ', idx)
        crop, targets = dataset[idx]

        largest_multiplier = 1

        for target in targets:
            class_id = int(target[0])
            if class_multipliers[class_names[class_id]] > largest_multiplier:
                largest_multiplier = class_multipliers[class_names[class_id]]

        for _ in range(largest_multiplier):
            all_crops.append(crop)
            all_targets.append(targets)

    num_crops_balanced_dataset = len(all_crops)
    val_split = 0.2
    num_val = int(num_crops_balanced_dataset * val_split)
    num_train = num_crops_balanced_dataset - num_val

    #change the paths of the images and labels to split them into trianing and validation folders
    indices = list(range(num_crops_balanced_dataset))
    random.shuffle(indices)  # Shuffle the indices to randomize the data split

    for idx, random_idx in enumerate(indices):
        print('idx new: ', idx)

        # Determine whether the crop goes into training or validation set
        if idx < num_train:
            crop_path = os.path.join(train_image_dir, f'crop_{random_idx}.png')
            label_path = os.path.join(train_label_dir, f'crop_{random_idx}.txt')
        else:
            crop_path = os.path.join(val_image_dir, f'crop_{random_idx}.png')
            label_path = os.path.join(val_label_dir, f'crop_{random_idx}.txt')
        
        crop = all_crops[random_idx]
        targets = all_targets[random_idx]
        crop_uint8 = (crop.numpy().transpose(1, 2, 0)*255).astype(np.uint8)
        crop_image = Image.fromarray(crop_uint8)
        crop_image.save(crop_path)

        # Save the adjusted bounding boxes
        with open(label_path, 'w') as label_file:
            for target in targets:
                class_id, x_center, y_center, width, height = target
                # Format as required by YOLO: class x_center y_center width height
                label_file.write(f'{int(class_id)} {x_center} {y_center} {width} {height}\n')
