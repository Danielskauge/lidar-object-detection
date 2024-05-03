import yaml
import argparse
from ultralytics import YOLO


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model with YAML configuration')
    parser.add_argument('config', type=str, help='Path to the configuration YAML file')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    model = YOLO("yolov8n.pt")

    
    model.train(data="configs/balanced_dataset.yaml", **config)
