"""
FootVision AI — Phase 9: Precision Pitch Calibrator with Magnifier Loupe

FEATURES:
    - 4x / 6x High-Resolution Magnifier Loupe: Shows exact 1080p pixel grid and crosshairs.
    - Live Projected Pitch Wireframe: Real-time yellow field markings overlaid on video.
    - Per-Point Error Diagnostics: Shows exact residual error in meters for every landmark.
    - Non-linear Levenberg-Marquardt Homography Optimization.

HOW TO USE:
    1. Click any landmark on the Pitch Map (Right Panel) or Video Frame (Left Panel).
    2. Use the Magnifier Loupe in the corner of the Video Frame to click the exact pixel intersection.
    3. Pair 4 to 8 landmarks across the visible pitch area.
    4. Observe the yellow pitch wireframe overlay on the video to verify alignment.
    5. Press [ENTER] or [S] to save the calibrated matrix to outputs/homography.npy.
    6. Press [Z] to Undo last pair | [R] to Reset | [ESC] to Exit.

Usage:
    python scripts/phase9_pick_landmarks.py [--seq_dir data/raw/SNMOT-062] [--frame_idx 0]
"""

import os
import sys
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
import numpy as np

from src.calibration.pitch_model import PITCH_LANDMARKS, PITCH_LENGTH, PITCH_WIDTH, draw_pitch
from src.calibration.homography import compute_homography, save_homography, compute_reprojection_error, project_point


# ─── Layout Dimensions ────────────────────────────────────────────────────────
IMG_DISPLAY_W = 1200
IMG_DISPLAY_H = 675    # 16:9

PITCH_DISPLAY_W = 580
PITCH_DISPLAY_H = 376  # 105:68

HEADER_H = 65
TOTAL_W  = IMG_DISPLAY_W + PITCH_DISPLAY_W
TOTAL_H  = max(IMG_DISPLAY_H, PITCH_DISPLAY_H) + HEADER_H + 50  # Extra space for diagnostics

# Magnifier Loupe Config
LOUPE_SIZE = 220       # Display size of loupe box in pixels
LOUPE_ZOOM = 4.0       # 4x optical magnification
LOUPE_CROP = int(LOUPE_SIZE / LOUPE_ZOOM)  # 55x55 native pixel crop

# Palette colors (BGR)
COLOR_BG       = (25, 25, 25)
COLOR_TEXT     = (240, 240, 240)
COLOR_ACCENT   = (0, 215, 255)   # Yellow/Gold
COLOR_ACTIVE   = (0, 255, 255)   # Bright yellow
COLOR_PAIRED   = (0, 255, 0)     # Bright green
COLOR_GRID     = (0, 255, 255)   # Yellow for wireframe
COLOR_ERROR_HI = (0, 80, 255)    # Orange/Red for high error


class PrecisionCalibrator:
    def __init__(self, raw_frame: np.ndarray):
        self.raw_frame = raw_frame
        self.raw_h, self.raw_w = raw_frame.shape[:2]

        self.scale_x = self.raw_w / IMG_DISPLAY_W
        self.scale_y = self.raw_h / IMG_DISPLAY_H

        # State
        self.pairs = []            # list of dict
        self.pending_pitch = None
        self.pending_img = None
        self.hover_landmark = None
        self.mouse_img_pos = None  # (raw_x, raw_y, disp_x, disp_y)
        self.show_preview = True   # Live preview on by default
        self.H = None
        self.reproj_err = None
        self.point_errors = []

        # Build landmark pixel positions on pitch canvas
        self.pitch_landmarks_px = {}
        for name, (xm, ym) in PITCH_LANDMARKS.items():
            px = int(xm / PITCH_LENGTH * PITCH_DISPLAY_W)
            py = int(ym / PITCH_WIDTH  * PITCH_DISPLAY_H)
            self.pitch_landmarks_px[name] = (px, py)

    def _find_nearest_pitch_landmark(self, px: int, py: int, max_dist: int = 20):
        best_name = None
        best_dist = float('inf')
        for name, (lx, ly) in self.pitch_landmarks_px.items():
            dist = np.hypot(px - lx, py - ly)
            if dist < best_dist and dist <= max_dist:
                best_dist = dist
                best_name = name
        return best_name

    def mouse_callback(self, event, x, y, flags, param):
        y_adj = y - HEADER_H

        # Track mouse over video frame for Magnifier Loupe
        if 0 <= y_adj < IMG_DISPLAY_H and 0 <= x < IMG_DISPLAY_W:
            raw_x = int(x * self.scale_x)
            raw_y = int(y_adj * self.scale_y)
            self.mouse_img_pos = (raw_x, raw_y, x, y_adj)
            self.hover_landmark = None
        elif 0 <= y_adj < PITCH_DISPLAY_H and IMG_DISPLAY_W <= x < TOTAL_W:
            self.mouse_img_pos = None
            pitch_x = x - IMG_DISPLAY_W
            pitch_y = y_adj
            self.hover_landmark = self._find_nearest_pitch_landmark(pitch_x, pitch_y)
        else:
            self.mouse_img_pos = None
            self.hover_landmark = None

        if event == cv2.EVENT_LBUTTONDOWN:
            # Click on Broadcast Frame
            if 0 <= y_adj < IMG_DISPLAY_H and 0 <= x < IMG_DISPLAY_W:
                raw_x = int(x * self.scale_x)
                raw_y = int(y_adj * self.scale_y)

                if self.pending_pitch is not None:
                    name, (xm, ym), (p_px, p_py) = self.pending_pitch
                    self.pairs.append({
                        'name': name,
                        'pitch_m': (xm, ym),
                        'pitch_px': (p_px, p_py),
                        'img_px_raw': (raw_x, raw_y),
                        'img_px_disp': (x, y_adj)
                    })
                    self.pending_pitch = None
                    self._update_homography()
                else:
                    self.pending_img = (raw_x, raw_y, x, y_adj)

            # Click on Pitch Diagram
            elif 0 <= y_adj < PITCH_DISPLAY_H and IMG_DISPLAY_W <= x < TOTAL_W:
                pitch_x = x - IMG_DISPLAY_W
                pitch_y = y_adj
                near_name = self._find_nearest_pitch_landmark(pitch_x, pitch_y)

                if near_name is not None:
                    xm, ym = PITCH_LANDMARKS[near_name]
                    p_px, p_py = self.pitch_landmarks_px[near_name]

                    # Re-click removes old pair
                    self.pairs = [p for p in self.pairs if p['name'] != near_name]

                    if self.pending_img is not None:
                        raw_x, raw_y, disp_x, disp_y = self.pending_img
                        self.pairs.append({
                            'name': near_name,
                            'pitch_m': (xm, ym),
                            'pitch_px': (p_px, p_py),
                            'img_px_raw': (raw_x, raw_y),
                            'img_px_disp': (disp_x, disp_y)
                        })
                        self.pending_img = None
                        self._update_homography()
                    else:
                        self.pending_pitch = (near_name, (xm, ym), (p_px, p_py))

    def _update_homography(self):
        if len(self.pairs) >= 4:
            img_pts = [p['img_px_raw'] for p in self.pairs]
            pitch_pts = [p['pitch_m'] for p in self.pairs]
            try:
                self.H = compute_homography(img_pts, pitch_pts)
                self.reproj_err = compute_reprojection_error(self.H, img_pts, pitch_pts)
                # Compute individual point errors
                self.point_errors = []
                for p in self.pairs:
                    proj = project_point(self.H, p['img_px_raw'][0], p['img_px_raw'][1])
                    err = np.hypot(proj[0] - p['pitch_m'][0], proj[1] - p['pitch_m'][1])
                    self.point_errors.append(err)
            except Exception:
                self.H = None
                self.reproj_err = None
                self.point_errors = []
        else:
            self.H = None
            self.reproj_err = None
            self.point_errors = []

    def _draw_loupe(self, canvas: np.ndarray):
        """Draws a 4x magnified loupe of the cursor area on the 1080p frame."""
        if self.mouse_img_pos is None:
            return

        raw_x, raw_y, disp_x, disp_y = self.mouse_img_pos
        half = LOUPE_CROP // 2

        # Crop from raw 1080p frame
        y1 = max(0, raw_y - half)
        y2 = min(self.raw_h, raw_y + half)
        x1 = max(0, raw_x - half)
        x2 = min(self.raw_w, raw_x + half)

        crop = self.raw_frame[y1:y2, x1:x2]
        if crop.size == 0:
            return

        loupe = cv2.resize(crop, (LOUPE_SIZE, LOUPE_SIZE), interpolation=cv2.INTER_NEAREST)

        # Draw crosshairs in the center of the loupe
        cx, cy = LOUPE_SIZE // 2, LOUPE_SIZE // 2
        cv2.line(loupe, (cx - 15, cy), (cx + 15, cy), COLOR_ACTIVE, 1)
        cv2.line(loupe, (cx, cy - 15), (cx, cy + 15), COLOR_ACTIVE, 1)
        cv2.circle(loupe, (cx, cy), 3, (0, 0, 255), 1)

        # Loupe placement: bottom-left corner of video frame (or top-left if cursor is in bottom-left)
        if disp_x < LOUPE_SIZE + 20 and disp_y > IMG_DISPLAY_H - LOUPE_SIZE - 20:
            lx, ly = 15, HEADER_H + 15
        else:
            lx, ly = 15, HEADER_H + IMG_DISPLAY_H - LOUPE_SIZE - 15

        # Border & Title
        cv2.rectangle(canvas, (lx - 2, ly - 22), (lx + LOUPE_SIZE + 2, ly + LOUPE_SIZE + 2), (255, 255, 255), cv2.FILLED)
        cv2.rectangle(canvas, (lx, ly), (lx + LOUPE_SIZE, ly + LOUPE_SIZE), (0, 0, 0), cv2.FILLED)
        cv2.putText(canvas, f"4x Loupe ({raw_x}, {raw_y})", (lx + 4, ly - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

        canvas[ly:ly + LOUPE_SIZE, lx:lx + LOUPE_SIZE] = loupe

    def _draw_projected_wireframe(self, canvas: np.ndarray):
        if self.H is None:
            return

        try:
            H_inv = np.linalg.inv(self.H)
        except np.linalg.LinAlgError:
            return

        def pitch2disp(xm, ym):
            pt = np.array([[[xm, ym]]], dtype=np.float64)
            res = cv2.perspectiveTransform(pt, H_inv)
            raw_x, raw_y = res[0, 0, 0], res[0, 0, 1]
            disp_x = int(raw_x / self.scale_x)
            disp_y = int(raw_y / self.scale_y) + HEADER_H
            return (disp_x, disp_y)

        line_segments = [
            # Outer touchlines & goal lines
            ((0, 0), (105, 0)), ((105, 0), (105, 68)), ((105, 68), (0, 68)), ((0, 68), (0, 0)),
            # Halfway line
            ((52.5, 0), (52.5, 68)),
            # Left penalty box
            ((0, 13.84), (16.5, 13.84)), ((16.5, 13.84), (16.5, 54.16)), ((16.5, 54.16), (0, 54.16)),
            # Right penalty box
            ((105, 13.84), (88.5, 13.84)), ((88.5, 13.84), (88.5, 54.16)), ((88.5, 54.16), (105, 54.16)),
            # Left 6-yard box
            ((0, 24.84), (5.5, 24.84)), ((5.5, 24.84), (5.5, 43.16)), ((5.5, 43.16), (0, 43.16)),
            # Right 6-yard box
            ((105, 24.84), (99.5, 24.84)), ((99.5, 24.84), (99.5, 43.16)), ((99.5, 43.16), (105, 43.16)),
        ]

        for (p1, p2) in line_segments:
            pts = []
            for alpha in np.linspace(0, 1, 20):
                xm = p1[0] * (1 - alpha) + p2[0] * alpha
                ym = p1[1] * (1 - alpha) + p2[1] * alpha
                pts.append(pitch2disp(xm, ym))

            for i in range(len(pts) - 1):
                pa, pb = pts[i], pts[i + 1]
                if (0 <= pa[0] < IMG_DISPLAY_W and HEADER_H <= pa[1] < HEADER_H + IMG_DISPLAY_H and
                    0 <= pb[0] < IMG_DISPLAY_W and HEADER_H <= pb[1] < HEADER_H + IMG_DISPLAY_H):
                    cv2.line(canvas, pa, pb, COLOR_GRID, 1, cv2.LINE_AA)

        # Center circle
        circle_pts = []
        for theta in np.linspace(0, 2 * np.pi, 48):
            xm = 52.5 + 9.15 * np.cos(theta)
            ym = 34.0 + 9.15 * np.sin(theta)
            circle_pts.append(pitch2disp(xm, ym))

        for i in range(len(circle_pts)):
            pa = circle_pts[i]
            pb = circle_pts[(i + 1) % len(circle_pts)]
            if (0 <= pa[0] < IMG_DISPLAY_W and HEADER_H <= pa[1] < HEADER_H + IMG_DISPLAY_H and
                0 <= pb[0] < IMG_DISPLAY_W and HEADER_H <= pb[1] < HEADER_H + IMG_DISPLAY_H):
                cv2.line(canvas, pa, pb, COLOR_GRID, 1, cv2.LINE_AA)

    def render(self) -> np.ndarray:
        canvas = np.full((TOTAL_H, TOTAL_W, 3), COLOR_BG, dtype=np.uint8)

        # ── 1. Top HUD Header ─────────────────────────────────────────────────
        cv2.rectangle(canvas, (0, 0), (TOTAL_W, HEADER_H), (15, 15, 15), cv2.FILLED)
        cv2.line(canvas, (0, HEADER_H), (TOTAL_W, HEADER_H), (60, 60, 60), 1)

        n_pairs = len(self.pairs)
        if n_pairs < 4:
            status = f"Paired: {n_pairs}/4 minimum. Click a landmark on the Pitch or Video Frame."
            status_col = (0, 180, 255)
        else:
            err_str = f"{self.reproj_err:.2f}m" if self.reproj_err is not None else "N/A"
            status = f"Paired: {n_pairs} landmarks | Mean Reproj Error: {err_str} | [ENTER/S] Save | [P] Toggle Grid"
            status_col = (0, 255, 0) if (self.reproj_err and self.reproj_err < 1.5) else (0, 215, 255)

        cv2.putText(canvas, status, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, status_col, 1, cv2.LINE_AA)

        controls = "[4x Loupe Active]  |  [Z] Undo  |  [R] Reset  |  [ENTER/S] Save Calibration  |  [ESC] Exit"
        cv2.putText(canvas, controls, (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 180, 180), 1, cv2.LINE_AA)

        if self.pending_pitch is not None:
            prompt = f"Selected: '{self.pending_pitch[0]}' -> USE LOUPE TO CLICK EXACT PIXEL ON VIDEO"
            cv2.putText(canvas, prompt, (TOTAL_W - 750, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.46, COLOR_ACTIVE, 1, cv2.LINE_AA)
        elif self.pending_img is not None:
            prompt = "Video point clicked -> NOW CLICK MATCHING LANDMARK ON PITCH MAP"
            cv2.putText(canvas, prompt, (TOTAL_W - 750, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.46, COLOR_ACTIVE, 1, cv2.LINE_AA)
        elif self.hover_landmark is not None:
            xm, ym = PITCH_LANDMARKS[self.hover_landmark]
            hover_str = f"Hover: {self.hover_landmark} ({xm:.1f}m, {ym:.1f}m)"
            cv2.putText(canvas, hover_str, (TOTAL_W - 550, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)

        # ── 2. Left Panel: Broadcast Video Frame ──────────────────────────────
        img_disp = cv2.resize(self.raw_frame, (IMG_DISPLAY_W, IMG_DISPLAY_H), interpolation=cv2.INTER_AREA)
        canvas[HEADER_H:HEADER_H + IMG_DISPLAY_H, 0:IMG_DISPLAY_W] = img_disp

        # Projected Wireframe
        if self.show_preview and self.H is not None:
            self._draw_projected_wireframe(canvas)

        # Paired points on Broadcast Frame
        for idx, p in enumerate(self.pairs):
            disp_x, disp_y = p['img_px_disp']
            cy = disp_y + HEADER_H
            cx = disp_x
            cv2.circle(canvas, (cx, cy), 6, COLOR_PAIRED, -1)
            cv2.circle(canvas, (cx, cy), 8, (255, 255, 255), 1)
            err_text = f" #{idx + 1}"
            if idx < len(self.point_errors):
                err_text += f" ({self.point_errors[idx]:.1f}m)"
            cv2.putText(canvas, err_text, (cx + 8, cy - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_PAIRED, 1, cv2.LINE_AA)

        # Pending image point
        if self.pending_img is not None:
            _, _, disp_x, disp_y = self.pending_img
            cy = disp_y + HEADER_H
            cx = disp_x
            cv2.circle(canvas, (cx, cy), 6, COLOR_ACTIVE, -1)
            cv2.circle(canvas, (cx, cy), 8, (255, 255, 255), 1)

        # Draw 4x Magnifier Loupe
        self._draw_loupe(canvas)

        # ── 3. Right Panel: 2D Pitch Diagram ──────────────────────────────────
        pitch_base = draw_pitch(PITCH_DISPLAY_W, PITCH_DISPLAY_H, line_thickness=1)
        pitch_y_start = HEADER_H + 10
        pitch_x_start = IMG_DISPLAY_W

        canvas[pitch_y_start:pitch_y_start + PITCH_DISPLAY_H, pitch_x_start:pitch_x_start + PITCH_DISPLAY_W] = pitch_base

        paired_names = {p['name']: idx for idx, p in enumerate(self.pairs)}

        for name, (px, py) in self.pitch_landmarks_px.items():
            cx = pitch_x_start + px
            cy = pitch_y_start + py

            if name in paired_names:
                idx = paired_names[name]
                cv2.circle(canvas, (cx, cy), 6, COLOR_PAIRED, -1)
                cv2.circle(canvas, (cx, cy), 8, (255, 255, 255), 1)
                cv2.putText(canvas, str(idx + 1), (cx + 7, cy - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
            elif self.pending_pitch is not None and self.pending_pitch[0] == name:
                cv2.circle(canvas, (cx, cy), 7, COLOR_ACTIVE, -1)
                cv2.circle(canvas, (cx, cy), 9, (255, 255, 255), 1)
            elif self.hover_landmark == name:
                cv2.circle(canvas, (cx, cy), 6, COLOR_ACCENT, -1)
            else:
                cv2.circle(canvas, (cx, cy), 4, (120, 120, 120), -1)
                cv2.circle(canvas, (cx, cy), 5, (200, 200, 200), 1)

        # Diagnostics & Tips Box
        diag_y = pitch_y_start + PITCH_DISPLAY_H + 15
        cv2.rectangle(canvas, (pitch_x_start + 10, diag_y), (TOTAL_W - 10, TOTAL_H - 10), (35, 35, 35), cv2.FILLED)

        cv2.putText(canvas, "Precision Alignment Guide:", (pitch_x_start + 20, diag_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLOR_ACCENT, 1, cv2.LINE_AA)

        tips = [
            "1. Use the 4x Loupe to click right on line intersections (T-junctions & corners).",
            "2. Spread points across the field: near sideline, far sideline, box corners.",
            "3. If a point shows high error (>2m), press [Z] to undo and re-click precisely.",
            "4. Yellow wireframe lines should align seamlessly with the green pitch."
        ]
        for i, tip in enumerate(tips):
            cv2.putText(canvas, tip, (pitch_x_start + 20, diag_y + 40 + i * 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.line(canvas, (IMG_DISPLAY_W, HEADER_H), (IMG_DISPLAY_W, TOTAL_H), (70, 70, 70), 2)
        return canvas

    def run(self) -> bool:
        win_name = "FootVision AI — Precision Pitch Calibrator"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, TOTAL_W, TOTAL_H)
        cv2.setMouseCallback(win_name, self.mouse_callback)

        saved = False
        while True:
            display = self.render()
            cv2.imshow(win_name, display)
            key = cv2.waitKey(25) & 0xFF

            if key == 27:  # ESC
                print("\n  Calibration exited without saving.")
                break
            elif key == ord('z') or key == ord('Z'):
                if self.pending_pitch is not None or self.pending_img is not None:
                    self.pending_pitch = None
                    self.pending_img = None
                elif self.pairs:
                    removed = self.pairs.pop()
                    print(f"  [Undo] Removed pair: {removed['name']}")
                    self._update_homography()
            elif key == ord('r') or key == ord('R'):
                self.pairs.clear()
                self.pending_pitch = None
                self.pending_img = None
                self._update_homography()
                print("  [Reset] Cleared all landmark pairs.")
            elif key == ord('p') or key == ord('P'):
                self.show_preview = not self.show_preview
                print(f"  Pitch wireframe preview: {'ON' if self.show_preview else 'OFF'}")
            elif key == 13 or key == 10 or key == ord('s') or key == ord('S'):
                if len(self.pairs) < 4:
                    print(f"  [Error] At least 4 landmark pairs required (currently {len(self.pairs)}).")
                else:
                    saved = True
                    break

        cv2.destroyAllWindows()
        return saved


def main():
    parser = argparse.ArgumentParser(description="FootVision AI — Precision Pitch Calibrator")
    parser.add_argument("--seq_dir",    default="data/raw/SNMOT-062")
    parser.add_argument("--frame_idx", type=int, default=0)
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

    print(f"\n=======================================================")
    print(f"  FootVision AI — Precision Calibrator (with 4x Loupe)")
    print(f"=======================================================")
    print(f"  Frame       : {frame_path}")
    print(f"  Resolution  : {frame.shape[1]}x{frame.shape[0]}")
    print(f"=======================================================\n")

    calibrator = PrecisionCalibrator(frame)
    saved = calibrator.run()

    if saved and calibrator.H is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        h_path = os.path.join(args.output_dir, "homography.npy")
        save_homography(calibrator.H, h_path)

        print(f"\n  [SUCCESS] Calibrated with {len(calibrator.pairs)} landmarks!")
        print(f"  Mean Reprojection Error: {calibrator.reproj_err:.3f} meters")
        print(f"  Matrix saved to: {h_path}")
        print(f"\n  Next step: run the smoothed pipeline:")
        print(f"    python scripts/phase9_pitch_radar.py --no_viewer\n")


if __name__ == "__main__":
    main()
