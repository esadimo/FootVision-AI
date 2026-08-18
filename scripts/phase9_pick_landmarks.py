"""
FootVision AI — Phase 9: Interactive Dual-Panel Pitch Calibration Tool

PURPOSE:
    Calibrate the camera-to-pitch homography matrix (H) for any camera shot
    (including zoomed-in, half-pitch, or partial views).

HOW IT WORKS:
    1. The window shows the Broadcast Frame on the LEFT and a 2D FIFA Pitch Map on the RIGHT.
    2. Click any visible landmark on the Pitch Map (or near one) -> it highlights in yellow.
    3. Click the matching marking intersection on the Broadcast Frame -> they pair up!
    4. (Alternatively: click the broadcast frame first, then click the pitch landmark).
    5. Repeat for any 4 or more landmarks visible in your camera view (e.g. penalty box corners,
       penalty spot, center circle, halfway line).
    6. Press [P] to PREVIEW the projected pitch grid overlay on the video frame.
    7. Press [ENTER] or [S] to SAVE the homography matrix to outputs/homography.npy.
    8. Press [Z] to Undo the last pair | [R] to Reset all pairs | [ESC] to Exit.

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


# ─── Display Layout Dimensions ────────────────────────────────────────────────
IMG_DISPLAY_W = 1200
IMG_DISPLAY_H = 675   # 16:9 ratio

PITCH_DISPLAY_W = 580
PITCH_DISPLAY_H = 376  # 105:68 ratio

HEADER_H = 65
TOTAL_W  = IMG_DISPLAY_W + PITCH_DISPLAY_W
TOTAL_H  = max(IMG_DISPLAY_H, PITCH_DISPLAY_H) + HEADER_H

# Palette colors (BGR)
COLOR_BG       = (25, 25, 25)
COLOR_TEXT     = (240, 240, 240)
COLOR_ACCENT   = (0, 215, 255)   # Yellow/Gold
COLOR_ACTIVE   = (0, 255, 255)   # Bright yellow
COLOR_PAIRED   = (0, 255, 0)     # Green
COLOR_GRID     = (0, 255, 255)   # Cyan/Yellow for wireframe


class DualPanelCalibrator:
    def __init__(self, raw_frame: np.ndarray):
        self.raw_frame = raw_frame
        self.raw_h, self.raw_w = raw_frame.shape[:2]

        # Display scalers
        self.scale_x = self.raw_w / IMG_DISPLAY_W
        self.scale_y = self.raw_h / IMG_DISPLAY_H

        # State
        self.pairs = []            # list of dict: {'name': str, 'pitch_m': (X,Y), 'pitch_px': (px,py), 'img_px_raw': (x,y), 'img_px_disp': (x,y)}
        self.pending_pitch = None  # (name, (X_m, Y_m), (canvas_x, canvas_y)) if pitch clicked first
        self.pending_img   = None  # (raw_x, raw_y, disp_x, disp_y) if frame clicked first
        self.hover_landmark = None # name of landmark mouse is currently near on pitch canvas
        self.show_preview = False
        self.H = None
        self.reproj_err = None

        # Build landmark pixel positions on pitch canvas
        self.pitch_landmarks_px = {}
        for name, (xm, ym) in PITCH_LANDMARKS.items():
            px = int(xm / PITCH_LENGTH * PITCH_DISPLAY_W)
            py = int(ym / PITCH_WIDTH  * PITCH_DISPLAY_H)
            self.pitch_landmarks_px[name] = (px, py)

    def _find_nearest_pitch_landmark(self, px: int, py: int, max_dist: int = 22):
        best_name = None
        best_dist = float('inf')
        for name, (lx, ly) in self.pitch_landmarks_px.items():
            dist = np.hypot(px - lx, py - ly)
            if dist < best_dist and dist <= max_dist:
                best_dist = dist
                best_name = name
        return best_name

    def mouse_callback(self, event, x, y, flags, param):
        # Adjust for header offset
        y_adj = y - HEADER_H

        # 1. Hover tracking on pitch panel
        if 0 <= y_adj < PITCH_DISPLAY_H and IMG_DISPLAY_W <= x < TOTAL_W:
            pitch_x = x - IMG_DISPLAY_W
            pitch_y = y_adj
            self.hover_landmark = self._find_nearest_pitch_landmark(pitch_x, pitch_y)
        else:
            self.hover_landmark = None

        # 2. Left click handling
        if event == cv2.EVENT_LBUTTONDOWN:
            # Click on Broadcast Frame (LEFT PANEL)
            if 0 <= y_adj < IMG_DISPLAY_H and 0 <= x < IMG_DISPLAY_W:
                raw_x = int(x * self.scale_x)
                raw_y = int(y_adj * self.scale_y)

                if self.pending_pitch is not None:
                    # Complete pair: Pitch was clicked first, now Frame clicked
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
                    # Frame clicked first
                    self.pending_img = (raw_x, raw_y, x, y_adj)

            # Click on Pitch Diagram (RIGHT PANEL)
            elif 0 <= y_adj < PITCH_DISPLAY_H and IMG_DISPLAY_W <= x < TOTAL_W:
                pitch_x = x - IMG_DISPLAY_W
                pitch_y = y_adj
                near_name = self._find_nearest_pitch_landmark(pitch_x, pitch_y)

                if near_name is not None:
                    xm, ym = PITCH_LANDMARKS[near_name]
                    p_px, p_py = self.pitch_landmarks_px[near_name]

                    # Check if already paired
                    if any(p['name'] == near_name for p in self.pairs):
                        # Remove existing pair to allow re-clicking
                        self.pairs = [p for p in self.pairs if p['name'] != near_name]

                    if self.pending_img is not None:
                        # Complete pair: Frame was clicked first, now Pitch clicked
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
                        # Pitch clicked first
                        self.pending_pitch = (near_name, (xm, ym), (p_px, p_py))

    def _update_homography(self):
        if len(self.pairs) >= 4:
            img_pts = [p['img_px_raw'] for p in self.pairs]
            pitch_pts = [p['pitch_m'] for p in self.pairs]
            try:
                self.H = compute_homography(img_pts, pitch_pts)
                self.reproj_err = compute_reprojection_error(self.H, img_pts, pitch_pts)
            except Exception:
                self.H = None
                self.reproj_err = None
        else:
            self.H = None
            self.reproj_err = None

    def _draw_projected_wireframe(self, canvas: np.ndarray):
        """Projects known pitch lines onto the broadcast display canvas using H_inv."""
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

        # Lines to draw
        line_segments = [
            # Boundary
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

        # Draw line segments with interpolation for clipping
        for (p1, p2) in line_segments:
            # Interpolate points along segment
            pts = []
            for alpha in np.linspace(0, 1, 15):
                xm = p1[0] * (1 - alpha) + p2[0] * alpha
                ym = p1[1] * (1 - alpha) + p2[1] * alpha
                pts.append(pitch2disp(xm, ym))

            for i in range(len(pts) - 1):
                pt_a = pts[i]
                pt_b = pts[i + 1]
                # Only draw if within broadcast frame area
                if (0 <= pt_a[0] < IMG_DISPLAY_W and HEADER_H <= pt_a[1] < HEADER_H + IMG_DISPLAY_H and
                    0 <= pt_b[0] < IMG_DISPLAY_W and HEADER_H <= pt_b[1] < HEADER_H + IMG_DISPLAY_H):
                    cv2.line(canvas, pt_a, pt_b, COLOR_GRID, 1, cv2.LINE_AA)

        # Draw center circle as interpolated polygon
        circle_pts = []
        for theta in np.linspace(0, 2 * np.pi, 40):
            xm = 52.5 + 9.15 * np.cos(theta)
            ym = 34.0 + 9.15 * np.sin(theta)
            circle_pts.append(pitch2disp(xm, ym))

        for i in range(len(circle_pts)):
            pt_a = circle_pts[i]
            pt_b = circle_pts[(i + 1) % len(circle_pts)]
            if (0 <= pt_a[0] < IMG_DISPLAY_W and HEADER_H <= pt_a[1] < HEADER_H + IMG_DISPLAY_H and
                0 <= pt_b[0] < IMG_DISPLAY_W and HEADER_H <= pt_b[1] < HEADER_H + IMG_DISPLAY_H):
                cv2.line(canvas, pt_a, pt_b, COLOR_GRID, 1, cv2.LINE_AA)

    def render(self) -> np.ndarray:
        canvas = np.full((TOTAL_H, TOTAL_W, 3), COLOR_BG, dtype=np.uint8)

        # ── 1. Top HUD Header ─────────────────────────────────────────────────
        cv2.rectangle(canvas, (0, 0), (TOTAL_W, HEADER_H), (15, 15, 15), cv2.FILLED)
        cv2.line(canvas, (0, HEADER_H), (TOTAL_W, HEADER_H), (60, 60, 60), 1)

        # Status text
        n_pairs = len(self.pairs)
        if n_pairs < 4:
            status = f"Paired: {n_pairs}/4 minimum required. Click a landmark on the Pitch or Video Frame."
            status_col = (0, 180, 255)
        else:
            err_str = f"{self.reproj_err:.2f}m" if self.reproj_err is not None else "N/A"
            status = f"Paired: {n_pairs} landmarks | Reprojection Error: {err_str} | [ENTER/S] Save | [P] Toggle Preview"
            status_col = (0, 255, 0) if (self.reproj_err and self.reproj_err < 3.0) else (0, 215, 255)

        cv2.putText(canvas, status, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, status_col, 1, cv2.LINE_AA)

        controls = "[P] Preview Grid  |  [Z] Undo  |  [R] Reset All  |  [ENTER/S] Save Calibration  |  [ESC] Exit"
        cv2.putText(canvas, controls, (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 180, 180), 1, cv2.LINE_AA)

        # Active prompt on right side of header
        if self.pending_pitch is not None:
            prompt = f"Selected: '{self.pending_pitch[0]}' -> NOW CLICK MATCHING POINT ON VIDEO FRAME"
            cv2.putText(canvas, prompt, (TOTAL_W - 750, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_ACTIVE, 1, cv2.LINE_AA)
        elif self.pending_img is not None:
            prompt = "Video point clicked -> NOW CLICK MATCHING LANDMARK ON PITCH MAP"
            cv2.putText(canvas, prompt, (TOTAL_W - 750, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_ACTIVE, 1, cv2.LINE_AA)
        elif self.hover_landmark is not None:
            xm, ym = PITCH_LANDMARKS[self.hover_landmark]
            hover_str = f"Hover: {self.hover_landmark} ({xm:.1f}m, {ym:.1f}m)"
            cv2.putText(canvas, hover_str, (TOTAL_W - 550, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        # ── 2. Left Panel: Broadcast Video Frame ──────────────────────────────
        img_disp = cv2.resize(self.raw_frame, (IMG_DISPLAY_W, IMG_DISPLAY_H), interpolation=cv2.INTER_AREA)
        canvas[HEADER_H:HEADER_H + IMG_DISPLAY_H, 0:IMG_DISPLAY_W] = img_disp

        # Draw wireframe preview if active
        if self.show_preview and self.H is not None:
            self._draw_projected_wireframe(canvas)

        # Draw paired points on Broadcast Frame
        for idx, p in enumerate(self.pairs):
            disp_x, disp_y = p['img_px_disp']
            cy = disp_y + HEADER_H
            cx = disp_x
            cv2.circle(canvas, (cx, cy), 6, COLOR_PAIRED, -1)
            cv2.circle(canvas, (cx, cy), 8, (255, 255, 255), 1)
            cv2.putText(canvas, str(idx + 1), (cx + 8, cy - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_PAIRED, 1, cv2.LINE_AA)

        # Draw pending image point
        if self.pending_img is not None:
            _, _, disp_x, disp_y = self.pending_img
            cy = disp_y + HEADER_H
            cx = disp_x
            cv2.circle(canvas, (cx, cy), 6, COLOR_ACTIVE, -1)
            cv2.circle(canvas, (cx, cy), 8, (255, 255, 255), 1)

        # ── 3. Right Panel: 2D Pitch Diagram ──────────────────────────────────
        pitch_base = draw_pitch(PITCH_DISPLAY_W, PITCH_DISPLAY_H, line_thickness=1)
        pitch_y_start = HEADER_H + 10
        pitch_x_start = IMG_DISPLAY_W

        canvas[pitch_y_start:pitch_y_start + PITCH_DISPLAY_H, pitch_x_start:pitch_x_start + PITCH_DISPLAY_W] = pitch_base

        # Draw all candidate landmark target dots on Pitch Map
        paired_names = {p['name']: idx for idx, p in enumerate(self.pairs)}

        for name, (px, py) in self.pitch_landmarks_px.items():
            cx = pitch_x_start + px
            cy = pitch_y_start + py

            if name in paired_names:
                # Already paired
                idx = paired_names[name]
                cv2.circle(canvas, (cx, cy), 6, COLOR_PAIRED, -1)
                cv2.circle(canvas, (cx, cy), 8, (255, 255, 255), 1)
                cv2.putText(canvas, str(idx + 1), (cx + 7, cy - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
            elif self.pending_pitch is not None and self.pending_pitch[0] == name:
                # Currently active/selected
                cv2.circle(canvas, (cx, cy), 7, COLOR_ACTIVE, -1)
                cv2.circle(canvas, (cx, cy), 9, (255, 255, 255), 1)
            elif self.hover_landmark == name:
                # Mouse hovering near
                cv2.circle(canvas, (cx, cy), 6, COLOR_ACCENT, -1)
            else:
                # Available landmark
                cv2.circle(canvas, (cx, cy), 4, (120, 120, 120), -1)
                cv2.circle(canvas, (cx, cy), 5, (200, 200, 200), 1)

        # Panel title and instruction box
        info_y = pitch_y_start + PITCH_DISPLAY_H + 20
        cv2.rectangle(canvas, (pitch_x_start + 10, info_y), (TOTAL_W - 10, TOTAL_H - 10), (35, 35, 35), cv2.FILLED)
        cv2.putText(canvas, "Visible Features Guide:", (pitch_x_start + 20, info_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_ACCENT, 1, cv2.LINE_AA)

        tips = [
            "- Zoomed/Half pitch? Click ONLY the markings visible in your shot!",
            "- Great choices: Penalty box corners, 6-yard box, penalty spot, center circle.",
            "- Minimum 4 points needed across the field for an accurate plane transformation.",
            "- Press [P] to see the yellow pitch wireframe projected over the video."
        ]
        for i, tip in enumerate(tips):
            cv2.putText(canvas, tip, (pitch_x_start + 20, info_y + 44 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1, cv2.LINE_AA)

        # Vertical divider line between panels
        cv2.line(canvas, (IMG_DISPLAY_W, HEADER_H), (IMG_DISPLAY_W, TOTAL_H), (70, 70, 70), 2)

        return canvas

    def run(self) -> bool:
        win_name = "FootVision AI — Pitch Calibration (Interactive Dual-Panel)"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, TOTAL_W, TOTAL_H)
        cv2.setMouseCallback(win_name, self.mouse_callback)

        saved = False
        while True:
            display = self.render()
            cv2.imshow(win_name, display)
            key = cv2.waitKey(30) & 0xFF

            if key == 27:  # ESC
                print("\n  Calibration exited without saving.")
                break
            elif key == ord('z') or key == ord('Z'):  # Undo
                if self.pending_pitch is not None or self.pending_img is not None:
                    self.pending_pitch = None
                    self.pending_img = None
                elif self.pairs:
                    removed = self.pairs.pop()
                    print(f"  [Undo] Removed pair: {removed['name']}")
                    self._update_homography()
            elif key == ord('r') or key == ord('R'):  # Reset
                self.pairs.clear()
                self.pending_pitch = None
                self.pending_img = None
                self._update_homography()
                print("  [Reset] Cleared all landmark pairs.")
            elif key == ord('p') or key == ord('P'):  # Toggle preview
                if len(self.pairs) >= 4:
                    self.show_preview = not self.show_preview
                    print(f"  Pitch wireframe preview: {'ON' if self.show_preview else 'OFF'}")
                else:
                    print("  Need at least 4 pairs to preview homography wireframe.")
            elif key == 13 or key == 10 or key == ord('s') or key == ord('S'):  # ENTER / S
                if len(self.pairs) < 4:
                    print(f"  [Error] At least 4 landmark pairs required (currently {len(self.pairs)}).")
                else:
                    saved = True
                    break

        cv2.destroyAllWindows()
        return saved


def main():
    parser = argparse.ArgumentParser(description="FootVision AI — Phase 9 Interactive Pitch Calibrator")
    parser.add_argument("--seq_dir",    default="data/raw/SNMOT-062")
    parser.add_argument("--frame_idx", type=int, default=0,
                        help="Frame index to use for calibration (default: 0)")
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
    print(f"  FootVision AI — Interactive Pitch Calibrator")
    print(f"=======================================================")
    print(f"  Frame       : {frame_path}")
    print(f"  Resolution  : {frame.shape[1]}x{frame.shape[0]}")
    print(f"\n  HOW TO USE:")
    print(f"  1. Click any visible pitch landmark on the RIGHT panel (Pitch Map).")
    print(f"  2. Click the matching marking on the LEFT panel (Video Frame).")
    print(f"  3. Repeat for any 4+ visible landmarks in your shot.")
    print(f"  4. Press [P] to preview the projected pitch wireframe overlay.")
    print(f"  5. Press [ENTER] or [S] to save the homography matrix.")
    print(f"=======================================================\n")

    calibrator = DualPanelCalibrator(frame)
    saved = calibrator.run()

    if saved and calibrator.H is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        h_path = os.path.join(args.output_dir, "homography.npy")
        save_homography(calibrator.H, h_path)

        print(f"\n  [SUCCESS] Homography computed from {len(calibrator.pairs)} landmarks!")
        print(f"  Mean Reprojection Error: {calibrator.reproj_err:.3f} meters")
        print(f"  Matrix saved to: {h_path}")
        print(f"\n  Next step: run the full pipeline:")
        print(f"    python scripts/phase9_pitch_radar.py --no_viewer\n")


if __name__ == "__main__":
    main()
