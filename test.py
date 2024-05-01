import torch
from ultralytics import YOLO
from torch.utils.data import DataLoader
from datamodule import LidarDataModule


if __name__ == "__main__":
    # Load your model and dataset
    print("Loading model")
    model = YOLO('yolov8n.yaml')
    print("Loading datamodule")
    datamodule = LidarDataModule(batch_size=3, image_dir='NAPLab-LiDAR/images', label_dir='NAPLab-LiDAR/labels_yolo_v1.1')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        print(f"Epoch {epoch}")
        for images, targets in datamodule.train_dataloader():

            # Forward pass
            results = model(images, targets)

            print("RESULTS:", results)

            # Backward and optimize
            optimizer.zero_grad()
            #loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{10}], Loss: {loss.item()}')