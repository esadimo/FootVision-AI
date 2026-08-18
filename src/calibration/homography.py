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


import json


def save_homography(H: np.ndarray, path: str) -> None:
    """Saves a single homography matrix to a .npy or .json file."""
    if path.endswith(".json"):
        data = {"keyframes": [{"frame_idx": 0, "H": H.tolist()}]}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    else:
        np.save(path, H)
    print(f"  [Calibration] Homography matrix saved to: {path}")


def save_multikeyframe_homography(keyframes: List[dict], path: str) -> None:
    """
    Saves multi-keyframe homographies to a JSON file.
    Each dict in keyframes: {'frame_idx': int, 'H': np.ndarray (or list)}
    """
    serializable = []
    for kf in keyframes:
        h_matrix = kf['H'].tolist() if isinstance(kf['H'], np.ndarray) else kf['H']
        serializable.append({
            "frame_idx": int(kf['frame_idx']),
            "H": h_matrix
        })
    # Sort by frame index
    serializable.sort(key=lambda x: x['frame_idx'])
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"keyframes": serializable}, f, indent=2)
    print(f"  [Calibration] Multi-keyframe homography ({len(serializable)} frames) saved to: {path}")


def load_homography(path: str):
    """
    Loads homography calibration from file.
    Supports both single static .npy and multi-keyframe .json formats.
    """
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        keyframes = []
        for kf in data.get("keyframes", []):
            keyframes.append({
                "frame_idx": kf["frame_idx"],
                "H": np.array(kf["H"], dtype=np.float64)
            })
        keyframes.sort(key=lambda x: x["frame_idx"])
        print(f"  [Calibration] Loaded {len(keyframes)} keyframe homographies from: {path}")
        return keyframes
    else:
        H = np.load(path)
        print(f"  [Calibration] Static homography loaded from: {path}")
        return H


def get_homography_for_frame(calibration_data, frame_idx: int, img_w: int = 1920, img_h: int = 1080) -> np.ndarray:
    """
    Retrieves or smoothly interpolates the homography matrix for a given frame_idx.

    If calibration_data is a single 3x3 ndarray, returns it directly.
    If calibration_data is a list of keyframe dicts, smoothly interpolates between
    the enclosing keyframes using 4-point virtual anchor projective blending.
    """
    if isinstance(calibration_data, np.ndarray):
        return calibration_data

    if not isinstance(calibration_data, list) or len(calibration_data) == 0:
        raise ValueError("Invalid calibration data.")

    keyframes = calibration_data
    if len(keyframes) == 1:
        return keyframes[0]["H"]

    # Boundary conditions
    if frame_idx <= keyframes[0]["frame_idx"]:
        return keyframes[0]["H"]
    if frame_idx >= keyframes[-1]["frame_idx"]:
        return keyframes[-1]["H"]

    # Find bounding keyframes
    kf_prev = keyframes[0]
    kf_next = keyframes[-1]
    for i in range(len(keyframes) - 1):
        if keyframes[i]["frame_idx"] <= frame_idx <= keyframes[i + 1]["frame_idx"]:
            kf_prev = keyframes[i]
            kf_next = keyframes[i + 1]
            break

    f_prev = kf_prev["frame_idx"]
    f_next = kf_next["frame_idx"]
    if f_prev == f_next:
        return kf_prev["H"]

    alpha = (frame_idx - f_prev) / (f_next - f_prev)  # 0.0 -> 1.0

    # Smooth virtual 4-corner metric interpolation (guaranteed non-degenerate)
    corners_img = np.array([
        [[0.0, 0.0]],
        [[float(img_w), 0.0]],
        [[float(img_w), float(img_h)]],
        [[0.0, float(img_h)]]
    ], dtype=np.float64)

    pts_m_prev = cv2.perspectiveTransform(corners_img, kf_prev["H"])
    pts_m_next = cv2.perspectiveTransform(corners_img, kf_next["H"])

    # Linearly blend metric coordinates
    pts_m_interp = (1.0 - alpha) * pts_m_prev + alpha * pts_m_next

    # Solve intermediate homography
    H_interp, _ = cv2.findHomography(corners_img, pts_m_interp, method=0)
    return H_interp


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
