from ultralytics import YOLO
import time



# Load a pretrained YOLOv8n model
model = YOLO('runs/detect/balanced_lower_lr_SGD/weights/best.pt')

start_time = time.time()

# Run inference on 'bus.jpg' with arguments
model.predict('BalancedDataset/train/images/crop_3.png', imgsz=640)

end_time = time.time()

# Calculate and print the inference time
inference_time = end_time - start_time
print(f"Inference time: {inference_time} seconds")