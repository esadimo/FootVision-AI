"""
FootVision AI — Phase 9: Multi-Keyframe Precision Calibrator (Panning Camera Support)

PURPOSE:
    Calibrate pitch homography across panning camera movements (e.g. Right Half -> Midfield -> Left Half).
    The pipeline smoothly interpolates between keyframes to track the moving camera angle!

HOW TO USE:
    1. The window starts at Frame 0 (or your first keyframe).
    2. Click visible landmarks on the Pitch Map (Right Panel) and Video Frame (Left Panel) with the 4x Loupe.
    3. Press [N] or [RIGHT ARROW] to jump to the next keyframe (e.g. Frame 375 or Frame 749).
    4. Calibrate the visible markings in that new camera angle.
    5. Press [P] to toggle the live yellow pitch wireframe.
    6. Press [ENTER] or [S] to SAVE all keyframes to outputs/homography_keyframes.json.
    7. Controls:
       - [N] / [B] or [<] / [>] : Next / Previous Keyframe (or jump by 100 frames)
       - [Z] : Undo last landmark pair
       - [R] : Reset current keyframe pairs
       - [ENTER] or [S] : Save calibration
       - [ESC] : Exit

Usage:
    python scripts/phase9_pick_landmarks.py [--seq_dir data/raw/SNMOT-062] [--keyframes 0 375 749]
"""

import os
import sys
import argparse
import json

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
import numpy as np

from src.calibration.pitch_model import PITCH_LANDMARKS, PITCH_LENGTH, PITCH_WIDTH, draw_pitch
from src.calibration.homography import (
    compute_homography,
    save_homography,
    save_multikeyframe_homography,
    compute_reprojection_error,
    project_point
)


# ─── Layout Dimensions ────────────────────────────────────────────────────────
IMG_DISPLAY_W = 1200
IMG_DISPLAY_H = 675    # 16:9

PITCH_DISPLAY_W = 580
PITCH_DISPLAY_H = 376  # 105:68

HEADER_H = 65
TOTAL_W  = IMG_DISPLAY_W + PITCH_DISPLAY_W
TOTAL_H  = max(IMG_DISPLAY_H, PITCH_DISPLAY_H) + HEADER_H + 55

# Magnifier Loupe Config
LOUPE_SIZE = 220
LOUPE_ZOOM = 4.0
LOUPE_CROP = int(LOUPE_SIZE / LOUPE_ZOOM)

# Colors (BGR)
COLOR_BG       = (25, 25, 25)
COLOR_TEXT     = (240, 240, 240)
COLOR_ACCENT   = (0, 215, 255)   # Gold
COLOR_ACTIVE   = (0, 255, 255)   # Bright Yellow
COLOR_PAIRED   = (0, 255, 0)     # Bright Green
COLOR_GRID     = (0, 255, 255)   # Yellow wireframe


class MultiKeyframeCalibrator:
    def __init__(self, frame_paths: list, initial_keyframes: list):
        self.frame_paths = frame_paths
        self.total_frames = len(frame_paths)
        self.keyframe_indices = sorted(list(set(initial_keyframes)))
        self.current_kf_ptr = 0
        self.current_frame_idx = self.keyframe_indices[0]

        # Keyframe storage: {frame_idx: {'pairs': [...], 'H': np.ndarray, 'err': float}}
        self.calibrations = {}
        for kf in self.keyframe_indices:
            self.calibrations[kf] = {'pairs': [], 'H': None, 'err': None, 'point_errors': []}

        # Load current frame
        self._load_frame(self.current_frame_idx)

        # Interactive state
        self.pending_pitch = None
        self.pending_img = None
        self.hover_landmark = None
        self.mouse_img_pos = None
        self.show_preview = True

        # Pitch landmarks pixel lookup
        self.pitch_landmarks_px = {}
        for name, (xm, ym) in PITCH_LANDMARKS.items():
            px = int(xm / PITCH_LENGTH * PITCH_DISPLAY_W)
            py = int(ym / PITCH_WIDTH  * PITCH_DISPLAY_H)
            self.pitch_landmarks_px[name] = (px, py)

    def _load_frame(self, frame_idx: int):
        self.current_frame_idx = frame_idx
        path = self.frame_paths[frame_idx]
        self.raw_frame = cv2.imread(path)
        self.raw_h, self.raw_w = self.raw_frame.shape[:2]
        self.scale_x = self.raw_w / IMG_DISPLAY_W
        self.scale_y = self.raw_h / IMG_DISPLAY_H
        if frame_idx not in self.calibrations:
            self.calibrations[frame_idx] = {'pairs': [], 'H': None, 'err': None, 'point_errors': []}
            if frame_idx not in self.keyframe_indices:
                self.keyframe_indices.append(frame_idx)
                self.keyframe_indices.sort()

    @property
    def cur_data(self):
        return self.calibrations[self.current_frame_idx]

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
        y_adj = y - HEADER_H

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
            # Click Broadcast Frame
            if 0 <= y_adj < IMG_DISPLAY_H and 0 <= x < IMG_DISPLAY_W:
                raw_x = int(x * self.scale_x)
                raw_y = int(y_adj * self.scale_y)

                if self.pending_pitch is not None:
                    name, (xm, ym), (p_px, p_py) = self.pending_pitch
                    self.cur_data['pairs'].append({
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

            # Click Pitch Diagram
            elif 0 <= y_adj < PITCH_DISPLAY_H and IMG_DISPLAY_W <= x < TOTAL_W:
                pitch_x = x - IMG_DISPLAY_W
                pitch_y = y_adj
                near_name = self._find_nearest_pitch_landmark(pitch_x, pitch_y)

                if near_name is not None:
                    xm, ym = PITCH_LANDMARKS[near_name]
                    p_px, p_py = self.pitch_landmarks_px[near_name]

                    # Replace existing
                    self.cur_data['pairs'] = [p for p in self.cur_data['pairs'] if p['name'] != near_name]

                    if self.pending_img is not None:
                        raw_x, raw_y, disp_x, disp_y = self.pending_img
                        self.cur_data['pairs'].append({
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
        pairs = self.cur_data['pairs']
        if len(pairs) >= 4:
            img_pts = [p['img_px_raw'] for p in pairs]
            pitch_pts = [p['pitch_m'] for p in pairs]
            try:
                H = compute_homography(img_pts, pitch_pts)
                err = compute_reprojection_error(H, img_pts, pitch_pts)
                pt_errs = []
                for p in pairs:
                    proj = project_point(H, p['img_px_raw'][0], p['img_px_raw'][1])
                    e = np.hypot(proj[0] - p['pitch_m'][0], proj[1] - p['pitch_m'][1])
                    pt_errs.append(e)
                self.cur_data['H'] = H
                self.cur_data['err'] = err
                self.cur_data['point_errors'] = pt_errs
            except Exception:
                self.cur_data['H'] = None
                self.cur_data['err'] = None
                self.cur_data['point_errors'] = []
        else:
            self.cur_data['H'] = None
            self.cur_data['err'] = None
            self.cur_data['point_errors'] = []

    def _draw_loupe(self, canvas: np.ndarray):
        if self.mouse_img_pos is None:
            return
        raw_x, raw_y, disp_x, disp_y = self.mouse_img_pos
        half = LOUPE_CROP // 2

        y1 = max(0, raw_y - half)
        y2 = min(self.raw_h, raw_y + half)
        x1 = max(0, raw_x - half)
        x2 = min(self.raw_w, raw_x + half)

        crop = self.raw_frame[y1:y2, x1:x2]
        if crop.size == 0:
            return

        loupe = cv2.resize(crop, (LOUPE_SIZE, LOUPE_SIZE), interpolation=cv2.INTER_NEAREST)
        cx, cy = LOUPE_SIZE // 2, LOUPE_SIZE // 2
        cv2.line(loupe, (cx - 15, cy), (cx + 15, cy), COLOR_ACTIVE, 1)
        cv2.line(loupe, (cx, cy - 15), (cx, cy + 15), COLOR_ACTIVE, 1)
        cv2.circle(loupe, (cx, cy), 3, (0, 0, 255), 1)

        if disp_x < LOUPE_SIZE + 20 and disp_y > IMG_DISPLAY_H - LOUPE_SIZE - 20:
            lx, ly = 15, HEADER_H + 15
        else:
            lx, ly = 15, HEADER_H + IMG_DISPLAY_H - LOUPE_SIZE - 15

        cv2.rectangle(canvas, (lx - 2, ly - 22), (lx + LOUPE_SIZE + 2, ly + LOUPE_SIZE + 2), (255, 255, 255), cv2.FILLED)
        cv2.rectangle(canvas, (lx, ly), (lx + LOUPE_SIZE, ly + LOUPE_SIZE), (0, 0, 0), cv2.FILLED)
        cv2.putText(canvas, f"4x Loupe ({raw_x}, {raw_y})", (lx + 4, ly - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        canvas[ly:ly + LOUPE_SIZE, lx:lx + LOUPE_SIZE] = loupe

    def _draw_projected_wireframe(self, canvas: np.ndarray):
        H = self.cur_data['H']
        if H is None:
            return
        try:
            H_inv = np.linalg.inv(H)
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
            ((0, 0), (105, 0)), ((105, 0), (105, 68)), ((105, 68), (0, 68)), ((0, 68), (0, 0)),
            ((52.5, 0), (52.5, 68)),
            ((0, 13.84), (16.5, 13.84)), ((16.5, 13.84), (16.5, 54.16)), ((16.5, 54.16), (0, 54.16)),
            ((105, 13.84), (88.5, 13.84)), ((88.5, 13.84), (88.5, 54.16)), ((88.5, 54.16), (105, 54.16)),
            ((0, 24.84), (5.5, 24.84)), ((5.5, 24.84), (5.5, 43.16)), ((5.5, 43.16), (0, 43.16)),
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

        pairs = self.cur_data['pairs']
        n_pairs = len(pairs)
        err = self.cur_data['err']
        err_str = f"{err:.2f}m" if err is not None else "N/A"

        title = f"KEYFRAME [Frame {self.current_frame_idx + 1}/{self.total_frames}] | Paired: {n_pairs}/4 min | Error: {err_str}"
        title_col = (0, 255, 0) if (err is not None and err < 1.5) else (0, 215, 255)
        cv2.putText(canvas, title, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, title_col, 1, cv2.LINE_AA)

        controls = "[N/B] Next/Prev Keyframe | [Z] Undo | [R] Reset Frame | [ENTER/S] Save All Keyframes | [ESC] Exit"
        cv2.putText(canvas, controls, (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 180, 180), 1, cv2.LINE_AA)

        if self.pending_pitch is not None:
            prompt = f"Selected: '{self.pending_pitch[0]}' -> USE LOUPE TO CLICK ON VIDEO"
            cv2.putText(canvas, prompt, (TOTAL_W - 680, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_ACTIVE, 1, cv2.LINE_AA)
        elif self.pending_img is not None:
            prompt = "Video point clicked -> NOW CLICK MATCHING LANDMARK ON PITCH MAP"
            cv2.putText(canvas, prompt, (TOTAL_W - 680, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_ACTIVE, 1, cv2.LINE_AA)

        # ── 2. Left Panel: Broadcast Video Frame ──────────────────────────────
        img_disp = cv2.resize(self.raw_frame, (IMG_DISPLAY_W, IMG_DISPLAY_H), interpolation=cv2.INTER_AREA)
        canvas[HEADER_H:HEADER_H + IMG_DISPLAY_H, 0:IMG_DISPLAY_W] = img_disp

        if self.show_preview and self.cur_data['H'] is not None:
            self._draw_projected_wireframe(canvas)

        for idx, p in enumerate(pairs):
            disp_x, disp_y = p['img_px_disp']
            cy = disp_y + HEADER_H
            cx = disp_x
            cv2.circle(canvas, (cx, cy), 6, COLOR_PAIRED, -1)
            cv2.circle(canvas, (cx, cy), 8, (255, 255, 255), 1)
            e_txt = f" #{idx+1}"
            if idx < len(self.cur_data['point_errors']):
                e_txt += f" ({self.cur_data['point_errors'][idx]:.1f}m)"
            cv2.putText(canvas, e_txt, (cx + 8, cy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_PAIRED, 1, cv2.LINE_AA)

        if self.pending_img is not None:
            _, _, disp_x, disp_y = self.pending_img
            cy = disp_y + HEADER_H
            cx = disp_x
            cv2.circle(canvas, (cx, cy), 6, COLOR_ACTIVE, -1)
            cv2.circle(canvas, (cx, cy), 8, (255, 255, 255), 1)

        self._draw_loupe(canvas)

        # ── 3. Right Panel: 2D Pitch Diagram ──────────────────────────────────
        pitch_base = draw_pitch(PITCH_DISPLAY_W, PITCH_DISPLAY_H, line_thickness=1)
        pitch_y_start = HEADER_H + 10
        pitch_x_start = IMG_DISPLAY_W
        canvas[pitch_y_start:pitch_y_start + PITCH_DISPLAY_H, pitch_x_start:pitch_x_start + PITCH_DISPLAY_W] = pitch_base

        paired_names = {p['name']: idx for idx, p in enumerate(pairs)}

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

        # ── 4. Bottom Keyframe Status Strip ───────────────────────────────────
        strip_y = pitch_y_start + PITCH_DISPLAY_H + 12
        cv2.rectangle(canvas, (pitch_x_start + 10, strip_y), (TOTAL_W - 10, TOTAL_H - 8), (35, 35, 35), cv2.FILLED)
        cv2.putText(canvas, "Keyframe Status (Panning Coverage):", (pitch_x_start + 20, strip_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLOR_ACCENT, 1, cv2.LINE_AA)

        kf_info_lines = []
        for kf in self.keyframe_indices:
            data = self.calibrations.get(kf, {})
            p_cnt = len(data.get('pairs', []))
            e = data.get('err', None)
            e_str = f"{e:.2f}m" if e is not None else "uncalibrated"
            is_active = (kf == self.current_frame_idx)
            prefix = "-> " if is_active else "   "
            kf_info_lines.append(f"{prefix}Frame {kf+1}: {p_cnt} pts ({e_str})")

        for i, line in enumerate(kf_info_lines[:3]):
            col = COLOR_ACTIVE if "->" in line else (200, 200, 200)
            cv2.putText(canvas, line, (pitch_x_start + 20, strip_y + 40 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)

        cv2.line(canvas, (IMG_DISPLAY_W, HEADER_H), (IMG_DISPLAY_W, TOTAL_H), (70, 70, 70), 2)
        return canvas

    def run(self) -> bool:
        win_name = "FootVision AI — Multi-Keyframe Pitch Calibrator"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, TOTAL_W, TOTAL_H)
        cv2.setMouseCallback(win_name, self.mouse_callback)

        saved = False
        while True:
            display = self.render()
            cv2.imshow(win_name, display)
            key = cv2.waitKey(25) & 0xFF

            if key == 27:  # ESC
                print("\n  Exited without saving.")
                break
            elif key == ord('n') or key == ord('N') or key == ord('.'):  # Next keyframe
                self.current_kf_ptr = (self.current_kf_ptr + 1) % len(self.keyframe_indices)
                self._load_frame(self.keyframe_indices[self.current_kf_ptr])
                self.pending_pitch = None
                self.pending_img = None
            elif key == ord('b') or key == ord('B') or key == ord(','):  # Prev keyframe
                self.current_kf_ptr = (self.current_kf_ptr - 1) % len(self.keyframe_indices)
                self._load_frame(self.keyframe_indices[self.current_kf_ptr])
                self.pending_pitch = None
                self.pending_img = None
            elif key == ord('z') or key == ord('Z'):  # Undo
                if self.pending_pitch is not None or self.pending_img is not None:
                    self.pending_pitch = None
                    self.pending_img = None
                elif self.cur_data['pairs']:
                    rem = self.cur_data['pairs'].pop()
                    print(f"  [Undo] Frame {self.current_frame_idx + 1}: Removed {rem['name']}")
                    self._update_homography()
            elif key == ord('r') or key == ord('R'):  # Reset
                self.cur_data['pairs'].clear()
                self.pending_pitch = None
                self.pending_img = None
                self._update_homography()
                print(f"  [Reset] Cleared pairs for Frame {self.current_frame_idx + 1}")
            elif key == ord('p') or key == ord('P'):  # Toggle preview
                self.show_preview = not self.show_preview
            elif key == 13 or key == 10 or key == ord('s') or key == ord('S'):  # ENTER / S
                # Check that at least one keyframe has >= 4 pairs
                valid_kfs = [kf for kf in self.keyframe_indices if self.calibrations[kf]['H'] is not None]
                if not valid_kfs:
                    print("\n  [Error] You must calibrate at least ONE keyframe with >= 4 pairs before saving!")
                else:
                    saved = True
                    break

        cv2.destroyAllWindows()
        return saved


def main():
    parser = argparse.ArgumentParser(description="FootVision AI — Multi-Keyframe Pitch Calibrator")
    parser.add_argument("--seq_dir",    default="data/raw/SNMOT-062")
    parser.add_argument("--keyframes",  type=int, nargs="+", default=[0, 375, 749],
                        help="Keyframe indices to calibrate (default: 0 375 749)")
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()

    img_dir = os.path.join(args.seq_dir, "img1")
    files   = sorted([f for f in os.listdir(img_dir)
                      if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    if not files:
        print(f"[ERROR] No images found in {img_dir}")
        sys.exit(1)

    frame_paths = [os.path.join(img_dir, f) for f in files]
    total_frames = len(frame_paths)

    # Clamp keyframes to valid range
    valid_kfs = [min(max(0, k), total_frames - 1) for k in args.keyframes]
    valid_kfs = sorted(list(set(valid_kfs)))

    print(f"\n=======================================================")
    print(f"  FootVision AI — Multi-Keyframe Calibrator")
    print(f"=======================================================")
    print(f"  Sequence    : {args.seq_dir} ({total_frames} frames)")
    print(f"  Keyframes   : {[k+1 for k in valid_kfs]} (1-indexed)")
    print(f"\n  HOW TO USE:")
    print(f"  1. Calibrate Frame {valid_kfs[0]+1} (Right Half) by pairing 4+ visible landmarks.")
    print(f"  2. Press [N] to jump to Frame {valid_kfs[1]+1} (Midfield) and pair its visible landmarks.")
    print(f"  3. Press [N] to jump to Frame {valid_kfs[2]+1} (Left Half) and pair its visible landmarks.")
    print(f"  4. Press [ENTER] or [S] to SAVE the full panning calibration.")
    print(f"=======================================================\n")

    calibrator = MultiKeyframeCalibrator(frame_paths, valid_kfs)
    saved = calibrator.run()

    if saved:
        os.makedirs(args.output_dir, exist_ok=True)
        valid_calibs = []
        for kf in calibrator.keyframe_indices:
            data = calibrator.calibrations.get(kf, {})
            if data.get('H') is not None:
                valid_calibs.append({'frame_idx': kf, 'H': data['H']})

        # Save both multi-keyframe JSON and first keyframe npy (for compatibility)
        json_path = os.path.join(args.output_dir, "homography_keyframes.json")
        save_multikeyframe_homography(valid_calibs, json_path)

        npy_path = os.path.join(args.output_dir, "homography.npy")
        save_homography(valid_calibs[0]['H'], npy_path)

        print(f"\n  [SUCCESS] Calibrated {len(valid_calibs)} keyframes across the sequence!")
        print(f"  Multi-keyframe JSON saved to: {json_path}")
        print(f"\n  Now run the full smoothed panning radar:")
        print(f"    python scripts/phase9_pitch_radar.py --no_viewer\n")


if __name__ == "__main__":
    main()
