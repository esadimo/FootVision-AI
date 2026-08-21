"""
FootVision AI -- Phase 11: Possession Estimation

WORKFLOW:
    Reads pre-computed player pitch coordinates (Phase 9) and ball pixel coordinates (Phase 10).
    Dynamically projects the ball to the pitch using the homography (Phase 9).
    Calculates ball-to-player proximity to establish possession.
    Outputs annotated video and a possession timeline CSV.

Usage:
    python scripts/phase11_possession.py [--seq_dir data/raw/SNMOT-062]
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

from src.calibration.homography import load_homography, project_point, get_homography_for_frame
from src.analytics.possession_estimator import PossessionEstimator
from src.visualization.overlays import show_frame, close_all_windows

# Configuration
FPS = 25.0
PROXIMITY_M = 2.5
SMOOTHING_FRAMES = 3


def draw_hud(frame: np.ndarray, 
             frame_num: int, 
             timestamp: float, 
             possession_team: str, 
             percentages: dict) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    
    cv2.rectangle(out, (0, 0), (w, 42), (15, 15, 15), cv2.FILLED)
    
    # Base text
    info = f"Phase 11: Possession  |  Frame {frame_num}  |  {timestamp:.2f}s"
    cv2.putText(out, info, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    
    # Possession Stats
    stat_text = f"POSSESSION: {percentages['Team A']:.1f}% (A) - {percentages['Team B']:.1f}% (B)"
    (tw, th), _ = cv2.getTextSize(stat_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    
    # Current possessor colored background
    bg_color = (15, 15, 15)
    text_color = (255, 255, 255)
    if possession_team == "Team A":
        bg_color = (200, 130, 130) # Light navy/white
        text_color = (0, 0, 0)
    elif possession_team == "Team B":
        bg_color = (80, 200, 80) # Green
        text_color = (0, 0, 0)
        
    x_offset = w - tw - 20
    cv2.rectangle(out, (x_offset - 5, 5), (w - 10, 37), bg_color, cv2.FILLED)
    cv2.putText(out, stat_text, (x_offset, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 1, cv2.LINE_AA)
    
    # Draw Possession Timeline Bar at the bottom
    bar_h = 8
    cv2.rectangle(out, (0, h - bar_h), (w, h), (50, 50, 50), cv2.FILLED)
    pct_a = percentages["Team A"] / 100.0 if percentages["Team A"] + percentages["Team B"] > 0 else 0.5
    sep_x = int(w * pct_a)
    
    if sep_x > 0:
        cv2.rectangle(out, (0, h - bar_h), (sep_x, h), (200, 130, 130), cv2.FILLED)
    if sep_x < w:
        cv2.rectangle(out, (sep_x, h - bar_h), (w, h), (80, 200, 80), cv2.FILLED)
        
    return out


def draw_connections(frame: np.ndarray, 
                     ball_px: tuple, 
                     players_px: dict, 
                     closest_id: int, 
                     closest_dist_m: float,
                     thresh_m: float) -> np.ndarray:
    out = frame.copy()
    if not ball_px or closest_id is None:
        return out
        
    bx, by = map(int, ball_px)
    
    # Draw line to closest player
    if closest_id in players_px:
        px, py = map(int, players_px[closest_id])
        
        # Color based on whether it's within threshold
        color = (0, 255, 0) if closest_dist_m <= thresh_m else (0, 0, 255)
        thickness = 2 if closest_dist_m <= thresh_m else 1
        
        cv2.line(out, (bx, by), (px, py), color, thickness, cv2.LINE_AA)
        
        # Distance text
        mid_x, mid_y = (bx + px) // 2, (by + py) // 2
        cv2.putText(out, f"{closest_dist_m:.1f}m", (mid_x + 5, mid_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
                    
    return out


def load_player_data(csv_path: str):
    # Dict[frame_num, Dict[track_id, (team_label, pitch_x_m, pitch_y_m, foot_x_px, foot_y_px)]]
    data = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fnum = int(row["frame_number"])
            tid = int(row["track_id"])
            if fnum not in data:
                data[fnum] = {}
            data[fnum][tid] = (
                row["team_label"],
                float(row["pitch_x_m"]),
                float(row["pitch_y_m"]),
                float(row["foot_x_px"]),
                float(row["foot_y_px"])
            )
    return data


def load_ball_data(csv_path: str):
    # Dict[frame_num, (x_center_px, y_center_px, is_interpolated)]
    data = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fnum = int(row["frame_number"])
            if int(row["ball_detected"]) == 1:
                data[fnum] = (
                    float(row["x_center"]),
                    float(row["y_center"]),
                    int(row["is_interpolated"]) == 1
                )
    return data


def main():
    parser = argparse.ArgumentParser(description="Phase 11: Possession Estimation")
    parser.add_argument("--seq_dir", default="data/raw/SNMOT-062")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--no_viewer", action="store_true")
    args = parser.parse_args()

    seq_name = os.path.basename(args.seq_dir)
    img_dir = os.path.join(args.seq_dir, "img1")
    frames = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png"))])
    total_frames = len(frames)

    # 1. Check for required files
    player_csv = os.path.join(args.output_dir, f"{seq_name}_phase9_pitch_coords.csv")
    ball_csv = os.path.join(args.output_dir, f"{seq_name}_phase10_ball.csv")
    homography_file = os.path.join(args.output_dir, "homography_keyframes.json")
    
    if not os.path.exists(homography_file):
        homography_file = os.path.join(args.output_dir, "homography.npy")

    for fpath in [player_csv, ball_csv, homography_file]:
        if not os.path.exists(fpath):
            print(f"[ERROR] Missing required file: {fpath}")
            print("Please ensure Phase 9 and Phase 10 have been completed.")
            sys.exit(1)

    print(f"\n  Phase 11 -- Possession Estimation")
    print(f"  Sequence : {seq_name} ({total_frames} frames)")
    print("  Loading data from previous phases...")

    player_data = load_player_data(player_csv)
    ball_data = load_ball_data(ball_csv)
    calib_data = load_homography(homography_file)

    estimator = PossessionEstimator(proximity_threshold_m=PROXIMITY_M, smoothing_frames=SMOOTHING_FRAMES)

    first = cv2.imread(os.path.join(img_dir, frames[0]))
    fh, fw = first.shape[:2]

    vid_path = os.path.join(args.output_dir, f"{seq_name}_phase11_possession.mp4")
    out_csv = os.path.join(args.output_dir, f"{seq_name}_phase11_possession.csv")
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(vid_path, fourcc, FPS, (fw, fh))
    
    csv_file = open(out_csv, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame_number", "timestamp_s", "possession_team", "closest_player_id", "closest_distance_m", "team_A_pct", "team_B_pct"])

    t0 = time.time()
    
    for fi, fname in enumerate(frames):
        frame_num = fi + 1
        timestamp = fi / FPS
        
        frame = cv2.imread(os.path.join(img_dir, fname))
        
        # Get frame data
        p_data = player_data.get(frame_num, {})
        b_data = ball_data.get(frame_num, None)
        
        # Prepare estimator inputs
        # players = {track_id: (team_label, x_m, y_m)}
        players_m = {tid: (vals[0], vals[1], vals[2]) for tid, vals in p_data.items()}
        players_px = {tid: (vals[3], vals[4]) for tid, vals in p_data.items()}
        
        ball_m = None
        ball_px = None
        if b_data:
            bx_px, by_px, _ = b_data
            ball_px = (bx_px, by_px)
            
            # Project ball pixel to pitch meters using homography
            # Wait, the ball is usually slightly off the ground, but for 2D proximity we project its center 
            # to the ground plane anyway.
            H = get_homography_for_frame(calib_data, fi, fw, fh)
            ball_m = project_point(H, bx_px, by_px)

        # Update Possession
        poss_team, closest_id, closest_dist = estimator.update(ball_m, players_m)
        pcts = estimator.get_possession_percentages()
        
        # Write CSV
        csv_writer.writerow([
            frame_num, f"{timestamp:.4f}", poss_team, 
            closest_id if closest_id is not None else -1, 
            f"{closest_dist:.2f}" if closest_dist != float('inf') else -1,
            f"{pcts['Team A']:.2f}", f"{pcts['Team B']:.2f}"
        ])

        # Draw overlays
        ann_frame = draw_connections(frame, ball_px, players_px, closest_id, closest_dist, PROXIMITY_M)
        final_frame = draw_hud(ann_frame, frame_num, timestamp, poss_team, pcts)
        
        writer.write(final_frame)
        
        if not args.no_viewer:
            show_frame(final_frame, "Phase 11 -- Possession Estimation")
            
        if frame_num % 50 == 0 or frame_num == total_frames:
            elapsed = time.time() - t0
            fps_now = frame_num / elapsed if elapsed > 0 else 0
            eta = (total_frames - frame_num) / fps_now if fps_now > 0 else 0
            print(f"  Frame {frame_num:4d}/{total_frames}  |  {fps_now:.0f} FPS  |  ETA: {eta:.0f}s  |  Poss: {poss_team}")

    csv_file.close()
    writer.release()
    close_all_windows()
    
    elapsed = time.time() - t0
    print(f"\n  Phase 11 Complete in {elapsed:.1f}s")
    print(f"  Output Video: {vid_path}")
    print(f"  Output CSV  : {out_csv}\n")

if __name__ == "__main__":
    main()
