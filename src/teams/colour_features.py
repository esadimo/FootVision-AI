"""
src.teams.colour_features — Extraction of normalized color metrics (LAB and HSV) from chest crops.
"""

import cv2
import numpy as np
from typing import Optional, Dict


def extract_chest_color_metrics(crop_bgr: np.ndarray) -> Optional[Dict[str, float]]:
    """
    Extracts median color properties from a player's chest crop.

    Returns
    -------
    dict with keys:
        'L': Lightness in [0, 1]
        'a': Green-Red axis in [-1, 1] (negative = green, positive = red)
        'b': Blue-Yellow axis in [-1, 1] (negative = blue, positive = yellow)
        'H': Hue in [0, 180] (HSV degrees/2: 20-38 is Yellow, 38-85 is Green)
        'S': Saturation in [0, 1]
        'V': Value in [0, 1]
        'median_bgr': (B, G, R) tuple
    """
    if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.shape[0] < 2 or crop_bgr.shape[1] < 2:
        return None

    # Convert to LAB
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    l_med, a_med, b_med = np.median(lab.reshape(-1, 3), axis=0)

    # Convert to HSV
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h_med, s_med, v_med = np.median(hsv.reshape(-1, 3), axis=0)

    # Median BGR
    bgr_med = np.median(crop_bgr.reshape(-1, 3), axis=0).astype(int)

    return {
        "L": float(l_med) / 255.0,
        "a": (float(a_med) - 128.0) / 128.0,
        "b": (float(b_med) - 128.0) / 128.0,
        "H": float(h_med),
        "S": float(s_med) / 255.0,
        "V": float(v_med) / 255.0,
        "median_bgr": (int(bgr_med[0]), int(bgr_med[1]), int(bgr_med[2]))
    }
