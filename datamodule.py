import lightning.pytorch as pl
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
import torch
from dataset import LidarDataset
import os
from torch.utils.data.dataloader import default_collate
from typing import Tuple, List


class LidarDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, image_dir, label_dir, train_split_ratio=0.8):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = os.cpu_count()
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.train_split_ratio = train_split_ratio

        train_dataset = LidarDataset(image_dir=self.image_dir, label_dir=self.label_dir, transforms=self.get_transforms("train"))
        val_dataset = LidarDataset(image_dir=self.image_dir, label_dir=self.label_dir, transforms=self.get_transforms("val"))
        indices = torch.randperm(len(train_dataset))
        val_size = int(len(train_dataset) * self.train_split_ratio)
        self.train_dataset = Subset(train_dataset, indices[-val_size:])
        self.val_dataset = Subset(val_dataset, indices[:-val_size])

    def safe_collate(self, batch):
        try:
            return default_collate(batch)
        except RuntimeError as e:
            print("Error in collating:", e)
            
            return None  # or handle appropriately

    def custom_collate(self, batch) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        batch: List[Tuple[torch.Tensor, List[torch.Tensor]]]
        """
        images, crops = zip(*batch)
        images_tensor = torch.stack(images)
        return images_tensor, crops

    def calculate_mean_std(self):
        loader = self.train_dataloader()
        total_images = 0
        sum_means = 0
        sum_vars = 0

        for images, _ in loader:
            print("mean - new image batch")
            batch_mean = images.mean(dim=[0,1,2])
            sum_means += batch_mean * images.size(0)
            total_images += images.size(0)
        global_mean = sum_means / total_images

        for images, _ in loader:
            print("std - new image batch")
            batch_var = ((images - global_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))**2).mean([0,1,2])
            sum_vars += batch_var * images.size(0)
        global_std = torch.sqrt(sum_vars / total_images)

        return global_mean, global_std
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True, collate_fn=self.custom_collate)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=False, collate_fn=self.custom_collate)

    def get_transforms(self,split):
        #Mean: 0.4694253206253052, Std: 0.2967234253883362
        
        shared_transforms = [
            transforms.ToTensor(),
            transforms.Resize((640, 640))
        ]
        
        if split == "train":
            return transforms.Compose([
                *shared_transforms,
                #transforms.RandomHorizontalFlip(),
            ])
            
        elif split == "val":
            return transforms.Compose([
                *shared_transforms,
                # ...
            ])
        elif split == "test":
            return transforms.Compose([
                *shared_transforms,
                # ...
            ])
        
if __name__ == "__main__":
    datamodule = LidarDataModule(batch_size=16, image_dir="NAPLab-LiDAR/images", label_dir="NAPLab-LiDAR/labels_yolo_v1.1")
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    print(f"Number of training samples: {len(train_loader.dataset)}")
    print(f"Number of validation samples: {len(val_loader.dataset)}")
    #mean, std = datamodule.calculate_mean_std()
    #print(f"Mean: {mean}, Std: {std}")

