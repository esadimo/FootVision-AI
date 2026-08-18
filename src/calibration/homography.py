"""
src.calibration.homography — Planar homography computation and point projection.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


def compute_homography(
    image_points: List[Tuple[float, float]],
    pitch_points:  List[Tuple[float, float]]
) -> np.ndarray:
    """
    Computes the 3x3 homography matrix H that maps image pixel coordinates
    to real-world pitch metric coordinates.

    Uses DLT + Levenberg-Marquardt non-linear optimization for geometric error minimization.

    Parameters
    ----------
    image_points : list of (x_px, y_px)
        Landmark pixel coordinates in the camera frame (native resolution).
    pitch_points : list of (X_m, Y_m)
        Corresponding landmark coordinates in pitch meter space.

    Returns
    -------
    H : np.ndarray of shape (3, 3)
        Optimized homography matrix.
    """
    if len(image_points) < 4 or len(pitch_points) < 4:
        raise ValueError("At least 4 point correspondences are required for homography computation.")
    if len(image_points) != len(pitch_points):
        raise ValueError("Image points and pitch points must have the same length.")

    src = np.array(image_points, dtype=np.float64)
    dst = np.array(pitch_points, dtype=np.float64)

    # If exactly 4 points, standard DLT
    if len(image_points) == 4:
        H, _ = cv2.findHomography(src, dst, method=0)
    else:
        # If > 4 points, use RANSAC with small inlier threshold for robust outlier rejection,
        # then refine with all inliers using least-squares
        H, inliers = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if H is not None and inliers is not None:
            inlier_mask = inliers.ravel() == 1
            if np.sum(inlier_mask) >= 4:
                # Refine with all inliers
                H, _ = cv2.findHomography(src[inlier_mask], dst[inlier_mask], method=0)

    if H is None:
        raise RuntimeError("Homography computation failed. Check that your landmark points are valid.")
    return H


def project_point(H: np.ndarray, x_px: float, y_px: float) -> Tuple[float, float]:
    """
    Projects a single pixel coordinate (x_px, y_px) to pitch metric space
    using the homography matrix H.

    Parameters
    ----------
    H : np.ndarray (3, 3)
        Homography matrix from image → pitch.
    x_px : float
        Pixel x-coordinate in the image.
    y_px : float
        Pixel y-coordinate in the image.

    Returns
    -------
    (X_m, Y_m) : tuple of float
        Corresponding coordinate in pitch meter space.
    """
    pt = np.array([[[x_px, y_px]]], dtype=np.float64)
    result = cv2.perspectiveTransform(pt, H)
    X_m = float(result[0, 0, 0])
    Y_m = float(result[0, 0, 1])
    return (X_m, Y_m)


def project_points_batch(H: np.ndarray,
                          points_px: List[Tuple[float, float]]
                          ) -> List[Tuple[float, float]]:
    """
    Projects a list of pixel coordinates to pitch metric space in a single batch call.
    """
    if not points_px:
        return []
    pts = np.array([[p] for p in points_px], dtype=np.float64)
    result = cv2.perspectiveTransform(pts, H)
    return [(float(r[0][0]), float(r[0][1])) for r in result]


def save_homography(H: np.ndarray, path: str) -> None:
    """Saves the homography matrix to a .npy file."""
    np.save(path, H)
    print(f"  [Calibration] Homography matrix saved to: {path}")


def load_homography(path: str) -> np.ndarray:
    """Loads the homography matrix from a .npy file."""
    H = np.load(path)
    print(f"  [Calibration] Homography matrix loaded from: {path}")
    return H


def compute_reprojection_error(H: np.ndarray,
                                image_points: List[Tuple[float, float]],
                                pitch_points:  List[Tuple[float, float]]) -> float:
    """
    Computes the mean reprojection error (in meters) to validate calibration quality.
    """
    projected = project_points_batch(H, image_points)
    errors = [np.sqrt((p[0] - g[0])**2 + (p[1] - g[1])**2)
              for p, g in zip(projected, pitch_points)]
    return float(np.mean(errors))
