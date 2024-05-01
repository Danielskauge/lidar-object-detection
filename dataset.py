from torch.utils.data import Dataset
import os
import torch
from typing import Tuple, List
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from typing import Dict, Any


class LidarDataset(Dataset):
    def __init__(self, image_dir: str, label_dir: str, transforms=None):
        """4
        Initializes the dataset with image and label directories.
        Args:
            image_dir (str): Directory containing images.
            label_dir (str): Directory containing labels in YOLO format.
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.images = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith('.PNG')]
        self.image_height = 128  # Assuming height is already suitable for square crops
        self.image_width = 1024
        self.num_crops = 8
        self.crop_size = self.image_width // self.num_crops  # Ensuring square crops

    def __len__(self):
        return len(self.images) * self.num_crops

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Returns a single square crop and its corresponding bounding boxes.
        """

        #process image
        image_idx = idx // self.num_crops
        crop_idx = idx % self.num_crops

        image_path = self.images[image_idx]
        image = Image.open(image_path)

        start_x = int(crop_idx * self.crop_size)
        end_x = int(start_x + self.crop_size)

        crop = image.crop((start_x, 0, end_x, self.image_height))
        crop = crop.resize((self.image_height, self.image_height), resample=Image.BILINEAR)

        if self.transforms:
            crop = self.transforms(crop)
        
        #process boxes
        label_file = os.path.splitext(os.path.basename(image_path))[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_file)
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found for image: {image_path}, label path: {label_path}")
        boxes_px = self.load_boxes_px(label_path)

        adjusted_boxes = self.adjust_boxes_for_crop(boxes_px, start_x)

        
        return crop, adjusted_boxes

    def load_boxes_px(self, label_path: str) -> torch.Tensor:
        """Load bounding boxes from a label file in YOLO format and convert them to absolute pixel coordinates."""
        boxes = []
        with open(label_path, 'r') as file:
            for line in file:
                class_id, x_center, y_center, width, height = map(float, line.split())
                x_center *= self.image_width
                y_center *= self.image_height
                width *= self.image_width
                height *= self.image_height
                boxes.append([class_id, x_center, y_center, width, height])
        return boxes
        

    def adjust_boxes_for_crop(self, boxes: List, start_x_px: int) -> List[torch.Tensor]:
        """Adjust box coordinates for a specific crop based on the start_x position in pixels,
        clipping boxes to crop boundaries, and converting coordinates back to relative format."""
        adjusted_boxes = []
        crop_end_x_px = start_x_px + self.crop_size

        for box in boxes:
            class_id, x_center_px, y_center_px, width_px, height_px = box

            # Calculate the left and right edges of the box in pixels
            box_left_px = x_center_px - width_px / 2
            box_right_px = x_center_px + width_px / 2

            # Check if any part of the box is within the crop
            if box_right_px > start_x_px and box_left_px < crop_end_x_px:
                # Clip the box edges to be within the crop
                clipped_left_px = max(box_left_px, start_x_px)
                clipped_right_px = min(box_right_px, crop_end_x_px)

                # Calculate new width based on clipped edges in pixels
                clipped_width_px = clipped_right_px - clipped_left_px

                # Recalculate the center based on the clipped edges in pixels
                new_x_center_px = clipped_left_px + clipped_width_px / 2
                new_x_center_ratio = (new_x_center_px - start_x_px) / self.crop_size
                new_y_center_ratio = y_center_px / self.image_height
                new_width_ratio = clipped_width_px / self.crop_size
                new_height_ratio = height_px / self.image_height

                box = torch.tensor([class_id, new_x_center_ratio, new_y_center_ratio, new_width_ratio, new_height_ratio])

                adjusted_boxes.append(box)
 
        return adjusted_boxes

     
if __name__ == "__main__":
    dataset = LidarDataset(image_dir="NAPLab-LiDAR/images", label_dir="NAPLab-LiDAR/labels_yolo_v1.1", transforms=transforms.ToTensor())
    print("Dataset Information:")
    print(f"Image directory: {dataset.image_dir}")
    print(f"Label directory: {dataset.label_dir}")
    print("Number of original images:", len(dataset.images))
    print("Number of crops:", dataset.__len__())
    print("Crop size:", dataset.crop_size)
    print("Example:")

    print("Image shape:", dataset.__getitem__(0)[0].size)
    print("Max intensity value:", dataset.__getitem__(0)[0].max().item())
    print("Min intensity value:", dataset.__getitem__(0)[0].min().item())
    import matplotlib.pyplot as plt
    
    fig, axs = plt.subplots(1, 8, figsize=(15, 3))  # Create a figure with 5 subplots in a row
    for i in range(8):
        image, boxes = dataset.__getitem__(i)
        axs[i].imshow(image, cmap='gray')
        for box in boxes:
            class_id, x_center, y_center, width, height = box
            x_center *= image.size[1]
            y_center *= image.size[0]
            width *= image.size[1]
            height *= image.size[0]
            rect = plt.Rectangle((x_center - width / 2, y_center - height / 2), width, height, linewidth=1, edgecolor='r', facecolor='none')
            axs[i].add_patch(rect)
        axs[i].axis('off')  # Turn off axis
    plt.show()

    for i in range(25):
        image, boxes = dataset.__getitem__(i)
        print("image size:", image.size)
