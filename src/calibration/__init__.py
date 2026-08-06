"""
src.calibration — Pitch detection and camera calibration modules.

Modules
-------
pitch_model : Defines the standard top-down pitch template with known landmark
              coordinates in metres and in normalised [0, 1] space.
keypoints   : Detect or manually collect pitch landmark correspondences
              (penalty-area corners, centre circle, touchlines).
homography  : Compute and apply the homography matrix that maps image pixel
              coordinates to real-world pitch coordinates.
validation  : Visualize and quantitatively validate a computed homography by
              overlaying a projected pitch grid on the broadcast frame.
"""
