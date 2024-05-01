from ultralytics import YOLO
import torch

# Create a YOLO model instance
model = YOLO()

# Check if it is an instance of torch.nn.Module
print(isinstance(model, torch.nn.Module))
