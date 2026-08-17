"""
src.teams.colour_features — Extraction of jersey dominant color vectors and histograms.
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from typing import Optional


def extract_dominant_color(
    pixels_bgr: np.ndarray,
    n_colors: int = 2,
    color_space: str = "hsv"
) -> Optional[np.ndarray]:
    """
    Extracts the dominant non-background color representation from a crop pixel array.

    Parameters
    ----------
    pixels_bgr : np.ndarray
        Array of shape (N, 3) representing BGR pixel values.
    n_colors : int
        Number of clusters to discover within the crop.
    color_space : str
        Target color space for feature representation ('hsv', 'lab', or 'bgr').

    Returns
    -------
    np.ndarray or None
        Normalized 1D feature vector of length 3 representing the dominant color.
    """
    if pixels_bgr is None or len(pixels_bgr) < 8:
        return None

    # Convert color space
    if color_space.lower() == "hsv":
        pixels_converted = cv2.cvtColor(pixels_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    elif color_space.lower() == "lab":
        pixels_converted = cv2.cvtColor(pixels_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    else:
        pixels_converted = pixels_bgr.astype(np.float32)

    # Use K-Means on the crop pixels to find dominant color centroid
    if len(pixels_converted) <= n_colors:
        return np.mean(pixels_converted, axis=0)

    try:
        kmeans = KMeans(n_clusters=min(n_colors, len(pixels_converted)), n_init=3, random_state=42)
        labels = kmeans.fit_predict(pixels_converted)
        
        # Pick the largest cluster (highest pixel count)
        counts = np.bincount(labels)
        dominant_idx = np.argmax(counts)
        dominant_color = kmeans.cluster_centers_[dominant_idx]
        return dominant_color
    except Exception:
        return np.median(pixels_converted, axis=0)


def extract_color_histogram(
    crop_bgr: np.ndarray,
    mask: Optional[np.ndarray] = None,
    bins: int = 16
) -> np.ndarray:
    """
    Extracts a normalized 2D Hue-Saturation color histogram.
    """
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], mask, [bins, bins], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist.flatten()
