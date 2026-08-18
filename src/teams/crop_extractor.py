"""
src.teams.crop_extractor — Spatial chest extraction preserving jersey colors (including green kits).
"""

import cv2
import numpy as np
from typing import Optional, Tuple


def extract_torso_crop(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    top_ratio: float = 0.15,
    bottom_ratio: float = 0.50,
    side_margin: float = 0.20
) -> Optional[np.ndarray]:
    """
    Extracts the inner-chest jersey crop of a player.
    Uses conservative side margins (20% from each side) to avoid the background pitch,
    and focuses on the upper 15%-50% region (chest area) to avoid shorts/legs.

    Parameters
    ----------
    frame : np.ndarray
        Full BGR image frame.
    bbox : tuple (x1, y1, x2, y2)
        Player bounding box coordinates.
    top_ratio : float
        Top offset (skips head/hair).
    bottom_ratio : float
        Bottom offset (stops before shorts).
    side_margin : float
        Horizontal trimming from left and right to avoid background pitch.

    Returns
    -------
    np.ndarray or None
        Cropped BGR image of the chest jersey.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h_img, w_img = frame.shape[:2]

    # Clip to image boundaries
    x1 = max(0, min(w_img - 1, x1))
    x2 = max(0, min(w_img, x2))
    y1 = max(0, min(h_img - 1, y1))
    y2 = max(0, min(h_img, y2))

    box_w = x2 - x1
    box_h = y2 - y1

    if box_w < 6 or box_h < 12:
        return None

    # Focus on the pure upper-chest zone
    chest_y1 = int(y1 + box_h * top_ratio)
    chest_y2 = int(y1 + box_h * bottom_ratio)
    
    # Trim sides aggressively to avoid pitch grass leakage
    chest_x1 = int(x1 + box_w * side_margin)
    chest_x2 = int(x2 - box_w * side_margin)

    if (chest_y2 - chest_y1) < 3 or (chest_x2 - chest_x1) < 3:
        return None

    return frame[chest_y1:chest_y2, chest_x1:chest_x2].copy()
