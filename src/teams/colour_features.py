"""
src.teams.colour_features — Robust jersey color representation in CIE-LAB and HSV space.
"""

import cv2
import numpy as np
from typing import Optional


def extract_player_feature_vector(
    crop_bgr: np.ndarray,
    grass_mask_thresh: tuple = ((30, 40, 40), (85, 255, 255))
) -> Optional[np.ndarray]:
    """
    Extracts a 4-dimensional normalized color feature vector from a torso crop:
        [L_norm, a_norm, b_norm, Saturation_norm]

    Parameters
    ----------
    crop_bgr : np.ndarray
        Torso crop image in BGR.
    grass_mask_thresh : tuple of (lower, upper)
        HSV thresholds for pitch green grass.

    Returns
    -------
    np.ndarray of shape (4,) or None
        Normalized feature vector.
    """
    if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.shape[0] < 4 or crop_bgr.shape[1] < 4:
        return None

    # 1. Mask out pitch grass green pixels
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    lower_green = np.array(grass_mask_thresh[0], dtype=np.uint8)
    upper_green = np.array(grass_mask_thresh[1], dtype=np.uint8)
    grass_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Non-grass pixels
    non_grass_bgr = crop_bgr[grass_mask == 0]
    non_grass_hsv = hsv[grass_mask == 0]

    # If almost everything was masked, use inner central 60% box as fallback
    if len(non_grass_bgr) < 8:
        ch, cw = crop_bgr.shape[:2]
        inner_crop = crop_bgr[int(ch*0.2):int(ch*0.8), int(cw*0.2):int(cw*0.8)]
        if inner_crop.size == 0:
            return None
        non_grass_bgr = inner_crop.reshape(-1, 3)
        non_grass_hsv = cv2.cvtColor(inner_crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)

    # 2. Convert to LAB color space
    lab_pixels = cv2.cvtColor(non_grass_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)

    # 3. Compute robust median values
    l_med, a_med, b_med = np.median(lab_pixels, axis=0)
    s_med = np.median(non_grass_hsv[:, 1])  # Saturation from HSV

    # 4. Normalize components into stable [0, 1] range:
    # L ranges [0, 255]
    # a, b range [0, 255] with neutral gray around 128
    # S ranges [0, 255]
    l_norm = float(l_med) / 255.0
    a_norm = (float(a_med) - 128.0) / 128.0
    b_norm = (float(b_med) - 128.0) / 128.0
    s_norm = float(s_med) / 255.0

    return np.array([l_norm, a_norm, b_norm, s_norm], dtype=np.float32)


def get_crop_representative_bgr(crop_bgr: np.ndarray) -> np.ndarray:
    """Returns the median BGR color of non-grass jersey pixels for visualization."""
    if crop_bgr is None or crop_bgr.size == 0:
        return np.array([128, 128, 128], dtype=np.uint8)
    
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    grass_mask = cv2.inRange(hsv, np.array([30, 40, 40]), np.array([85, 255, 255]))
    non_grass = crop_bgr[grass_mask == 0]
    if len(non_grass) < 8:
        non_grass = crop_bgr.reshape(-1, 3)
        
    return np.median(non_grass, axis=0).astype(np.uint8)
