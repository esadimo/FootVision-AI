"""
FootVision AI — Phase 9 (Step 1)
Interactive Landmark Picker for Pitch Calibration

PURPOSE:
    This script opens the first frame of your sequence and lets you click
    on known pitch marking intersections to calibrate the homography.
    Once you've clicked enough landmarks, press ENTER to compute and save H.

HOW TO USE:
    1. Run this script.
    2. A window opens showing frame 1 of your sequence.
    3. A panel on the right shows which landmark to click NEXT (with a hint description).
    4. Click precisely on the pitch marking intersection shown.
    5. Repeat until you've clicked all landmarks (or as many as you can see).
    6. Press ENTER to compute and save the homography matrix.
    7. Press Z to undo the last click.
    8. Press ESC to exit without saving.

LANDMARK SEQUENCE (click these in order):
    1. Top-left corner of pitch
    2. Top-right corner of pitch
    3. Bottom-left corner of pitch
    4. Bottom-right corner of pitch
    5. Left penalty area — top-right corner (intersection at 16.5m from goal)
    6. Left penalty area — bottom-right corner
    7. Right penalty area — top-left corner
    8. Right penalty area — bottom-left corner
    9. Center spot

    (More you add, the better the calibration quality.)

Usage:
    python scripts/phase9_pick_landmarks.py [--seq_dir data/raw/SNMOT-062]
"""

import os
import sys
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
import numpy as np

from src.calibration.pitch_model import PITCH_LANDMARKS, draw_pitch
from src.calibration.homography import compute_homography, save_homography, compute_reprojection_error

# Ordered list of landmarks to click and their display names
LANDMARK_SEQUENCE = [
    ("top_left_corner",             "1. TOP-LEFT corner of pitch"),
    ("top_right_corner",            "2. TOP-RIGHT corner of pitch"),
    ("bottom_left_corner",          "3. BOTTOM-LEFT corner of pitch"),
    ("bottom_right_corner",         "4. BOTTOM-RIGHT corner of pitch"),
    ("left_penalty_top_right",      "5. Left penalty box — TOP-RIGHT corner (16.5m from goal, 13.84m from top)"),
    ("left_penalty_bottom_right",   "6. Left penalty box — BOTTOM-RIGHT corner"),
    ("right_penalty_top_left",      "7. Right penalty box — TOP-LEFT corner"),
    ("right_penalty_bottom_left",   "8. Right penalty box — BOTTOM-LEFT corner"),
    ("center_spot",                 "9. CENTER SPOT (center of pitch)"),
    ("halfway_top",                 "10. Halfway line — TOP touchline intersection"),
    ("halfway_bottom",              "11. Halfway line — BOTTOM touchline intersection"),
    ("left_penalty_top_left",       "12. Left penalty box — TOP-LEFT (on goal line)"),
    ("left_penalty_bottom_left",    "13. Left penalty box — BOTTOM-LEFT (on goal line)"),
]

# UI Colors
COLOR_CLICKED  = (0, 255, 0)
COLOR_TARGET   = (0, 215, 255)
COLOR_TEXT     = (255, 255, 255)
CIRCLE_RADIUS  = 7


class LandmarkPicker:
    def __init__(self, frame: np.ndarray):
        self.frame       = frame.copy()
        self.clicked     = []    # [(x_px, y_px), ...]
        self.current_idx = 0
        self.done        = False

    def _redraw(self):
        canvas = self.frame.copy()
        h, w = canvas.shape[:2]

        # Draw already-clicked landmarks
        for i, (px, py) in enumerate(self.clicked):
            cv2.circle(canvas, (px, py), CIRCLE_RADIUS, COLOR_CLICKED, -1)
            cv2.circle(canvas, (px, py), CIRCLE_RADIUS + 2, (255, 255, 255), 1)
            name_short = LANDMARK_SEQUENCE[i][0].replace("_", " ")
            cv2.putText(canvas, f"{i+1}", (px + 8, py - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_CLICKED, 1, cv2.LINE_AA)

        # Top HUD
        cv2.rectangle(canvas, (0, 0), (w, 52), (20, 20, 20), cv2.FILLED)
        if self.current_idx < len(LANDMARK_SEQUENCE):
            _, desc = LANDMARK_SEQUENCE[self.current_idx]
            cv2.putText(canvas, f"Click: {desc}", (12, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, COLOR_TARGET, 1, cv2.LINE_AA)
        else:
            cv2.putText(canvas, "All landmarks collected!  Press ENTER to save, Z to undo.",
                        (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 1, cv2.LINE_AA)

        status = (f"  Clicked: {len(self.clicked)}/{len(LANDMARK_SEQUENCE)}  |  "
                  f"[ENTER] Save  [Z] Undo  [ESC] Cancel")
        cv2.putText(canvas, status, (12, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow("Phase 9 — Landmark Picker", canvas)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_idx < len(LANDMARK_SEQUENCE):
                self.clicked.append((x, y))
                self.current_idx += 1
                self._redraw()

    def run(self):
        cv2.namedWindow("Phase 9 — Landmark Picker", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Phase 9 — Landmark Picker", self.mouse_callback)
        self._redraw()

        while True:
            key = cv2.waitKey(50) & 0xFF

            if key == 27:   # ESC
                print("  Calibration cancelled.")
                break
            elif key == 13 or key == 10:  # ENTER
                if len(self.clicked) < 4:
                    print("  Need at least 4 clicks. Keep clicking.")
                else:
                    self.done = True
                    break
            elif key == ord('z') or key == ord('Z'):  # Undo
                if self.clicked:
                    self.clicked.pop()
                    self.current_idx -= 1
                    self._redraw()

        cv2.destroyAllWindows()
        return self.done


def main():
    parser = argparse.ArgumentParser(description="Phase 9 Step 1: Click pitch landmarks to calibrate homography.")
    parser.add_argument("--seq_dir",    default="data/raw/SNMOT-062")
    parser.add_argument("--frame_idx", type=int, default=0,
                        help="Which frame index to use for calibration (default: 0 = first frame)")
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()

    img_dir = os.path.join(args.seq_dir, "img1")
    files   = sorted([f for f in os.listdir(img_dir)
                      if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    if not files:
        print(f"[ERROR] No images found in {img_dir}")
        sys.exit(1)

    frame_path = os.path.join(img_dir, files[args.frame_idx])
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"[ERROR] Could not read frame: {frame_path}")
        sys.exit(1)

    print(f"\n  Phase 9 — Pitch Landmark Picker")
    print(f"  Frame: {frame_path}")
    print(f"  Resolution: {frame.shape[1]}x{frame.shape[0]}")
    print(f"\n  Instructions:")
    print(f"  - Click precisely on each pitch marking intersection listed at the top of the window")
    print(f"  - Press ENTER when finished (minimum 4 points required)")
    print(f"  - Press Z to undo the last click")
    print(f"  - Press ESC to cancel")
    print()

    picker = LandmarkPicker(frame)
    success = picker.run()

    if not success or len(picker.clicked) < 4:
        print("  Calibration not saved.")
        return

    # Collect the correspondences
    n = len(picker.clicked)
    image_points = picker.clicked[:n]
    pitch_points = [PITCH_LANDMARKS[LANDMARK_SEQUENCE[i][0]] for i in range(n)]

    print(f"\n  Computing homography from {n} point correspondences...")
    H = compute_homography(image_points, pitch_points)

    # Compute and report reprojection error
    err = compute_reprojection_error(H, image_points, pitch_points)
    print(f"  Mean reprojection error: {err:.3f} meters")
    if err > 3.0:
        print(f"  [WARNING] Error > 3m. Consider re-clicking landmarks more precisely.")
    else:
        print(f"  [Good] Calibration quality is acceptable.")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    h_path = os.path.join(args.output_dir, "homography.npy")
    save_homography(H, h_path)

    # Show result: project clicked points onto pitch
    pitch_canvas = draw_pitch(800, 520)
    for (X_m, Y_m) in pitch_points:
        px = int(X_m / 105.0 * 800)
        py = int(Y_m /  68.0 * 520)
        cv2.circle(pitch_canvas, (px, py), 5, (0, 215, 255), -1)

    cv2.imshow("Phase 9 — Pitch Calibration Preview", pitch_canvas)
    print(f"\n  Preview showing projected landmarks. Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"\n  Homography saved to: {h_path}")
    print(f"  Now run: python scripts/phase9_pitch_radar.py\n")


if __name__ == "__main__":
    main()
