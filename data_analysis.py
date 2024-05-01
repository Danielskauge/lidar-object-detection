import os
import random
from typing import List, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def load_data(img_dir: str, label_dir: str) -> pd.DataFrame:
    """Load and parse YOLO annotation files into a pandas DataFrame.

    Args:
        img_dir (str): The directory containing the image files.
        label_dir (str): The directory containing the YOLO label files.

    Returns:
        pd.DataFrame: A DataFrame containing columns for filename, class_id, and image dimensions.
    """

    data = []
    
    for label_file in os.listdir(label_dir):
        if label_file.endswith(".txt"):
            img_file = label_file.replace(".txt",".PNG")
            img_path = os.path.join(img_dir, img_file)
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                height, width, _ = img.shape
                label_path = os.path.join(label_dir, label_file)
                with open(label_path,"r") as file:
                    for line in file:
                        components = line.strip().split()
                        class_id, x_center, y_center, bbox_width, bbox_height = map(float, components)
                        #Convert YOLO format to absolute bounding box dimentions
                        bbox_width_abs = bbox_width * width
                        bbox_height_abs = bbox_height * height
                        data.append({
                            'filename': img_file,
                            'class_id': int(class_id),
                            'x_center': x_center,  
                            'y_center': y_center, 
                            'bb_width': bbox_width_abs,
                            'bb_height': bbox_height_abs
                        })
    return pd.DataFrame(data)

def plot_class_distribution(df: pd.DataFrame, class_names: dict):
    """Plot the distribution of classes as a bar chart.

    Args:
        df (pd.DataFrame): The DataFrame with class information.
        class_names (dict): A dictionary mapping class IDs to their names.
    """
    class_counts = df['class_id'].value_counts().sort_index()
    class_names_sorted = list(map(lambda class_id: class_names[class_id], class_counts.index))
    plt.figure(figsize=(10, 6))
    class_counts.plot(kind='bar')
    plt.title('Distribution of Classes')
    plt.xlabel('Class ID')
    plt.ylabel('Frequency')
    plt.xticks(ticks=np.arange(len(class_names_sorted)), labels=class_names_sorted, rotation=0)  # Adjust rotation if needed
    plt.savefig('plots/class_distribution.png')

def display_random_samples(df: pd.DataFrame, directory: str, class_names: dict, num_samples: int = 3):
    """Display random samples of images for each class, showing bounding boxes, with class names as column headers.

    Args:
        df (pd.DataFrame): The DataFrame with image and class information.
        directory (str): The directory containing the images.
        class_names (dict): A dictionary mapping class IDs to their names.
        num_samples (int, optional): Number of samples to display per class. Defaults to 3.
    """
    plt.figure(figsize=(num_samples * len(df['class_id'].unique()), 15))
    for index, class_id in enumerate(df['class_id'].unique()):
        sample_images = df[df['class_id'] == class_id].sample(n=num_samples, random_state=42)

        for i, row in enumerate(sample_images.iterrows(), 1):
            filename, bb_width, bb_height = row[1]['filename'], row[1]['bb_width'], row[1]['bb_height']
            img = cv2.imread(os.path.join(directory, filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax = plt.subplot(num_samples, len(df['class_id'].unique()), i * len(df['class_id'].unique()) + index)
            ax.imshow(img)
            # Drawing the bounding box
            start_point = (
                int(row[1]['x_center'] * img.shape[1] - bb_width / 2),
                int(row[1]['y_center'] * img.shape[0] - bb_height / 2)
            )
            end_point = (
                int(start_point[0] + bb_width),
                int(start_point[1] + bb_height)
            )
            cv2.rectangle(img, start_point, end_point, (255, 0, 0), 2)
            ax.imshow(img)
            ax.set_title(f"{class_names[class_id]} - {int(bb_width)}x{int(bb_height)}")
            ax.axis('off')
    plt.tight_layout()
    plt.savefig('plots/random_samples.png')

if __name__ == "__main__":

    class_names = {0: 'car', 1: 'truck', 2: 'bus', 3: 'motorcycle', 4: 'bicycle', 5: 'scooter', 6: 'person', 7: 'rider'}
    image_directory = 'NAPLab-LiDAR/images'
    label_directory = 'NAPLab-LiDAR/labels_yolo_v1.1'
    data_df = load_data(image_directory, label_directory)
    plot_class_distribution(data_df, class_names)
    #display_random_samples(data_df, image_directory, class_names)


        
