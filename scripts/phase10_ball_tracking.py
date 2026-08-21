"""
FootVision AI -- Phase 10: Ball Detection & Tracking

WORKFLOW:
    1. Loads YOLO model (detecting only class 32 - sports ball).
    2. Runs BallDetector with low confidence thresholds to catch blurred balls.
    3. Runs BallTracker to interpolate missing states when ball is occluded/fast.
    4. Renders tracking trail and confidence tags.
    5. Outputs MP4 and CSV.

Usage:
    python scripts/phase10_ball_tracking.py [--seq_dir data/raw/SNMOT-062]
"""

import os
import sys
import argparse
import csv
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
import numpy as np
from ultralytics import YOLO

from src.detection.ball_detector import BallDetector
from src.tracking.ball_tracker import BallTracker
from src.visualization.overlays import show_frame, close_all_windows

# ─── Constants ────────────────────────────────────────────────────────────────
MODEL_PATH        = "yolov8n.pt"
BALL_CLASS_ID     = 32
CONF_THRESHOLD    = 0.05   # Lower to catch motion blur
MAX_GAP_FRAMES    = 15     # Allow interpolating up to ~0.6 seconds of occlusion
FPS               = 25.0


def draw_ball_hud(frame: np.ndarray,
                  frame_num: int,
                  timestamp: float,
                  ball_pos: tuple,
                  is_interpolated: bool,
                  conf: float) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    
    # Top info bar
    cv2.rectangle(out, (0, 0), (w, 42), (20, 20, 20), cv2.FILLED)
    
    status = "DETECTED" if ball_pos and not is_interpolated else ("INTERPOLATED" if ball_pos else "LOST")
    col = (0, 255, 0) if status == "DETECTED" else ((0, 165, 255) if status == "INTERPOLATED" else (0, 0, 255))
    
    info = (f"Phase 10: Ball Tracking  |  Frame {frame_num}  |  {timestamp:.2f}s  |  "
            f"State: {status}")
    if status == "DETECTED" and conf > 0:
        info += f" ({conf*100:.0f}%)"
        
    cv2.putText(out, info, (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
    return out


def draw_ball_overlay(frame: np.ndarray, 
                      ball_pos: tuple, 
                      is_interpolated: bool, 
                      trail: list) -> np.ndarray:
    if not ball_pos:
        return frame
        
    out = frame.copy()
    cx, cy = map(int, ball_pos)
    
    # Draw trail
    if len(trail) > 1:
        for i in range(1, len(trail)):
            pt1 = tuple(map(int, trail[i-1]))
            pt2 = tuple(map(int, trail[i]))
            # Fade trail based on age
            thickness = 2
            color = (0, 140, 255) # Orange tail
            cv2.line(out, pt1, pt2, color, thickness, cv2.LINE_AA)

    # Draw current ball position
    # Orange circle for detected, Dashed/Yellowish for interpolated
    color = (0, 215, 255) if not is_interpolated else (0, 255, 255)
    radius = 12
    thickness = 2 if not is_interpolated else 1
    
    cv2.circle(out, (cx, cy), radius, color, thickness, cv2.LINE_AA)
    cv2.circle(out, (cx, cy), 2, (255, 255, 255), -1, cv2.LINE_AA)
    
    if is_interpolated:
        cv2.putText(out, "EST", (cx + 15, cy + 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

    return out


def main():
    parser = argparse.ArgumentParser(description="Phase 10: Ball Detection & Tracking")
    parser.add_argument("--seq_dir",     default="data/raw/SNMOT-062")
    parser.add_argument("--model",       default=MODEL_PATH)
    parser.add_argument("--conf",        type=float, default=CONF_THRESHOLD)
    parser.add_argument("--output_dir",  default="outputs")
    parser.add_argument("--no_viewer",   action="store_true")
    args = parser.parse_args()

    img_dir = os.path.join(args.seq_dir, "img1")
    frames  = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    if not frames:
        print(f"[ERROR] No images found in {img_dir}")
        sys.exit(1)

    seq_name = os.path.basename(args.seq_dir)
    total_frames = len(frames)
    
    print(f"\n  Phase 10 -- Ball Detection & Tracking")
    print(f"  Sequence : {seq_name} ({total_frames} frames)")
    print(f"  Model    : {args.model}")
    print()

    # Load modules
    model = YOLO(args.model)
    detector = BallDetector(conf_threshold=args.conf)
    tracker = BallTracker(max_gap_frames=MAX_GAP_FRAMES, history_len=20)

    # Setup writers
    os.makedirs(args.output_dir, exist_ok=True)
    vid_path = os.path.join(args.output_dir, f"{seq_name}_phase10_ball.mp4")
    csv_path = os.path.join(args.output_dir, f"{seq_name}_phase10_ball.csv")

    first = cv2.imread(os.path.join(img_dir, frames[0]))
    fh, fw = first.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(vid_path, fourcc, FPS, (fw, fh))
    
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame_number", "timestamp_s", "ball_detected", 
                         "x_center", "y_center", "is_interpolated", "confidence"])

    t0 = time.time()
    
    for fi, fname in enumerate(frames):
        frame_path = os.path.join(img_dir, fname)
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        frame_num = fi + 1
        timestamp = fi / FPS

        # Run YOLO strictly for the ball class
        results = model.predict(frame, classes=[BALL_CLASS_ID], conf=args.conf, verbose=False)
        
        # 1. Detect
        best_ball = detector.extract_best_ball(results[0].boxes)
        
        conf = 0.0
        bbox = None
        if best_ball is not None:
            conf, bbox = best_ball

        # 2. Track & Interpolate
        pos, is_interp = tracker.update(bbox)
        
        # 3. Save to CSV
        detected_flag = 1 if pos else 0
        px, py = pos if pos else (-1.0, -1.0)
        csv_writer.writerow([frame_num, f"{timestamp:.4f}", detected_flag, 
                             f"{px:.2f}", f"{py:.2f}", 1 if is_interp else 0, f"{conf:.3f}"])

        # 4. Render
        trail = tracker.get_trail(length=20)
        ann_frame = draw_ball_overlay(frame, pos, is_interp, trail)
        final_frame = draw_ball_hud(ann_frame, frame_num, timestamp, pos, is_interp, conf)

        writer.write(final_frame)

        if not args.no_viewer:
            show_frame(final_frame, "Phase 10 -- Ball Tracking")

        if frame_num % 50 == 0 or frame_num == total_frames:
            elapsed = time.time() - t0
            fps_now = frame_num / elapsed if elapsed > 0 else 0
            eta = (total_frames - frame_num) / fps_now if fps_now > 0 else 0
            status = "TRACKING" if pos else "LOST"
            print(f"  Frame {frame_num:4d}/{total_frames}  |  {fps_now:.1f} FPS  |  ETA: {eta:.0f}s  |  State: {status}")

    # Cleanup
    csv_file.close()
    writer.release()
    close_all_windows()
    
    elapsed = time.time() - t0
    print(f"\n  Phase 10 Ball Tracking Complete in {elapsed:.1f}s")
    print(f"  Output Video: {vid_path}")
    print(f"  Output CSV  : {csv_path}\n")

if __name__ == "__main__":
    main()
