"""
Analyze color distribution of extracted crops in LAB and HSV space.
"""
import os
import cv2
import numpy as np

crop_dir = "outputs/crop_samples"
torso_files = sorted([f for f in os.listdir(crop_dir) if f.endswith("_torso.jpg")])

print(f"{'Crop':<20} | {'Median BGR':<18} | {'Median LAB':<18} | {'Median HSV':<18}")
print("-" * 80)

for f in torso_files:
    img = cv2.imread(os.path.join(crop_dir, f))
    if img is None:
        continue
    # remove pitch grass if green
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([30, 40, 40]), np.array([85, 255, 255]))
    non_grass = img[mask == 0]
    if len(non_grass) < 10:
        non_grass = img.reshape(-1, 3)
        
    bgr_med = np.median(non_grass, axis=0).astype(int)
    
    lab_pix = cv2.cvtColor(non_grass.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    lab_med = np.median(lab_pix, axis=0).astype(int)
    
    hsv_pix = cv2.cvtColor(non_grass.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    hsv_med = np.median(hsv_pix, axis=0).astype(int)
    
    print(f"{f:<20} | {str(bgr_med):<18} | {str(lab_med):<18} | {str(hsv_med):<18}")
