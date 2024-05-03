
import os
import cv2
import pandas as pd
import wandb

# Initialize wandb
wandb.init(project='lidar_object_detection', entity='danielskauge')

def log_annotated_images(image_directory, label_directory, class_names):
    """Logs images with bounding box annotations to wandb.

    Args:
        image_directory (str): Directory containing the images.
        label_directory (str): Directory containing YOLO formatted label files.
        class_names (dict): Dictionary mapping class IDs to class names.
    """
    for label_file in os.listdir(label_directory):
        if label_file.endswith(".txt"):
            img_file = label_file.replace(".txt", ".PNG")
            img_path = os.path.join(image_directory, img_file)
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                height, width, _ = img.shape

                label_path = os.path.join(label_directory, label_file)
                boxes = []
                with open(label_path, "r") as file:
                    for line in file:
                        class_id, x_center, y_center, bbox_width, bbox_height = map(float, line.strip().split())
                        x_min = int((x_center - bbox_width / 2) * width)
                        y_min = int((y_center - bbox_height / 2) * height)
                        x_max = int((x_center + bbox_width / 2) * width)
                        y_max = int((y_center + bbox_height / 2) * height)

                        boxes.append({
                            "position": {
                                "minX": x_min,
                                "maxX": x_max,
                                "minY": y_min,
                                "maxY": y_max,
                            },
                            "class_id": int(class_id),
                            "box_caption": class_names[int(class_id)],
                            "domain": "pixel",
                        })

                # Log the image with bounding boxes to wandb
                wandb.log({"annotated_images": [wandb.Image(img, boxes={
                    "predictions": {
                        "box_data": boxes,
                        "class_labels": class_names
                    }
                })]})

class_names = {0: 'car', 1: 'truck', 2: 'bus', 3: 'motorcycle', 4: 'bicycle', 5: 'scooter', 6: 'person', 7: 'rider'}
image_directory = 'NAPLab-LiDAR/images'
label_directory = 'NAPLab-LiDAR/labels_yolo_v1.1'
log_annotated_images(image_directory, label_directory, class_names)

