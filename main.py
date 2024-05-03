
from ultralytics import YOLO

#project
project = "YOLOv8"
name = "baseline-1"

#training params
imgsz = 640
batch = 16
epochs = 100
freeze = 10
fraction = 1
patience = 10
optimizer = "Adam"


#augmentations
hsv_h = 0
hsv_s = 0
hsv_v = 0
translate = 0
scale = 0
mosaic = 0
erasing = 0
crop_fraction = 0

data_config = "configs/balanced.yaml"

model = YOLO("yolov8n.pt")

model.train(data=data_config, 
            patience=patience,
            epochs=epochs, 
            imgsz=imgsz, 
            batch=batch,
            project=project,
            name=name,
            freeze=freeze,
            hsv_h=hsv_h,
            hsv_s=hsv_s,
            hsv_v=hsv_v,
            translate=translate,
            scale=scale,
            mosaic=mosaic,
            erasing=erasing,
            crop_fraction=crop_fraction,
            fraction = fraction
            )
