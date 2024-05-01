
from ultralytics import YOLO

#training params
imgsz = 640
batch = 16
epochs = 100
optimizer = "Adam"
lr = 0.001
freeze = 10

#TODO: Check that everything is logged correclty to wandbd
#TODO: choose default params
#TODO: Check that the model is being saved correctly, try reloading from checkpoint
#TODO: Finish training to a baseline
#TODO: Evaulate the baseline to find possible improvmenets
#TODO: Plan experiments
#TODO: Make script for running all experiments

#augmentation params

model = YOLO("yolov8n.pt")

print(model)


model.train(data="configs/dataset.yaml", 
            epochs=epochs, 
            imgsz=imgsz, 
            batch=batch,
            project="YOLOv8",
            name="YOLOv8-1",
            freeze=freeze
            )
