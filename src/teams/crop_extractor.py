"""
src.teams.crop_extractor — Torso crop extraction and pitch grass background removal.
"""

import cv2
import numpy as np
from typing import Optional, Tuple


def extract_torso_crop(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    top_ratio: float = 0.15,
    bottom_ratio: float = 0.55
) -> Optional[np.ndarray]:
    """
    Crops the player's upper-torso region from a bounding box, avoiding the head
    and legs/grass to isolate jersey color.

    Parameters
    ----------
    frame : np.ndarray
        Full BGR image frame.
    bbox : tuple (x1, y1, x2, y2)
        Player bounding box coordinates.
    top_ratio : float
        Fraction of height to trim from the top (skips head/hair).
    bottom_ratio : float
        Fraction of height to trim up to (skips shorts, legs, socks, pitch grass).

    Returns
    -------
    np.ndarray or None
        Cropped BGR image of the player's jersey region, or None if invalid.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h_img, w_img = frame.shape[:2]

    # Clip to frame boundary
    x1 = max(0, min(w_img - 1, x1))
    x2 = max(0, min(w_img, x2))
    y1 = max(0, min(h_img - 1, y1))
    y2 = max(0, min(h_img, y2))

    box_w = x2 - x1
    box_h = y2 - y1

    if box_w < 8 or box_h < 15:
        return None

    # Calculate vertical torso bounds
    torso_y1 = int(y1 + box_h * top_ratio)
    torso_y2 = int(y1 + box_h * bottom_ratio)
    
    # Avoid side edge background pixels (trim 10% from each side)
    torso_x1 = int(x1 + box_w * 0.10)
    torso_x2 = int(x2 - box_w * 0.10)

    if (torso_y2 - torso_y1) < 4 or (torso_x2 - torso_x1) < 4:
        return None

    return frame[torso_y1:torso_y2, torso_x1:torso_x2].copy()


def remove_grass_mask(
    crop_bgr: np.ndarray,
    lower_green_hsv: Tuple[int, int, int] = (30, 40, 40),
    upper_green_hsv: Tuple[int, int, int] = (85, 255, 255)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a boolean mask excluding pitch grass green pixels from the crop.

    Parameters
    ----------
    crop_bgr : np.ndarray
        Torso crop image in BGR.
    lower_green_hsv : tuple
        Lower HSV threshold for grass green.
    upper_green_hsv : tuple
        Upper HSV threshold for grass green.

    Returns
    -------
    non_grass_pixels : np.ndarray
        Array of shape (N, 3) with BGR values of non-grass pixels.
    mask : np.ndarray
        Binary mask (255 = jersey pixel, 0 = grass pixel).
    """
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    grass_mask = cv2.inRange(hsv, np.array(lower_green_hsv), np.array(upper_green_hsv))
    
    # Invert mask: 255 for non-grass, 0 for grass
    jersey_mask = cv2.bitwise_not(grass_mask)
    
    non_grass_pixels = crop_bgr[jersey_mask > 0]
    
    # If almost all pixels were flagged as grass (e.g. green kit), fallback to entire crop
    if len(non_grass_pixels) < 0.15 * crop_bgr.shape[0] * crop_bgr.shape[1]:
        non_grass_pixels = crop_bgr.reshape(-1, 3)
        
    return non_grass_pixels, jersey_mask
