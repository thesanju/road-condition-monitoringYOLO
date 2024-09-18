# import YOLO from ultralytics and torch
from ultralytics import YOLO
import torch

# Disable cuDNN for stability in some environments (optional, not necessary in windows)
torch.backends.cudnn.enabled = False


# Load the pre-trained YOLOv8 small model (yolov8s.pt)
# 'yolov8m.pt'  for medium and yolov8l for large
model = YOLO("yolov8s.pt")


# Train the YOLO model using custom data
# Parameters:
# - data: Path to the YAML file that defines the dataset (train, val paths, class names, etc.)
# - batch: Number of images per batch (8 images in each batch)
# - imgsz: Input image size (640x640 pixels)
# - epochs: Number of epochs (how many times the model will see the entire dataset)
# - workers: Number of CPU threads used for loading the data (1 worker thread here)

model.train(data = "data/data.yaml", batch=8, imgsz=640, epochs=10, workers=1)