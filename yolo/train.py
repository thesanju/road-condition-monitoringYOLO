from ultralytics import YOLO
import torch

torch.backends.cudnn.enabled = False



model = YOLO("yolov8m.pt")

model.train(data = "data/data.yaml", batch=8, imgsz=640, epochs=10, workers=1)