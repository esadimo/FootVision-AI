"""
Diagnostic script to inspect player crops and kit colors from SNMOT-062.
"""
import os
import sys
import cv2
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ultralytics import YOLO
from src.teams.crop_extractor import extract_torso_crop, remove_grass_mask

img_dir = "data/raw/SNMOT-062/img1"
files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")])

model = YOLO("yolov8n.pt")

# Inspect frame 100
frame = cv2.imread(os.path.join(img_dir, files[100]))
results = model(frame, conf=0.20, classes=[0], verbose=False)[0]

crops = []
os.makedirs("outputs/crop_samples", exist_ok=True)

for i, box in enumerate(results.boxes):
    coords = box.xyxy[0].cpu().numpy().tolist()
    x1, y1, x2, y2 = [int(v) for v in coords]
    
    # Save full crop and torso crop
    full_crop = frame[y1:y2, x1:x2]
    torso_crop = extract_torso_crop(frame, (x1, y1, x2, y2))
    
    if full_crop is not None and full_crop.size > 0:
        cv2.imwrite(f"outputs/crop_samples/player_{i:02d}_full.jpg", full_crop)
    if torso_crop is not None and torso_crop.size > 0:
        cv2.imwrite(f"outputs/crop_samples/player_{i:02d}_torso.jpg", torso_crop)

print(f"Saved {len(results.boxes)} player crops to outputs/crop_samples/")
