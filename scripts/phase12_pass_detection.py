"""
FootVision AI -- Phase 12: Pass Detection

WORKFLOW:
    Reads possession timeline and player pitch coordinates.
    Identifies discrete "Pass" and "Turnover" events.
    Generates a Pass Network diagram (static radar map).
    Generates an annotated video with event pop-ups.

Usage:
    python scripts/phase12_pass_detection.py [--seq_dir data/raw/SNMOT-062]
"""

import os
import sys
import argparse
import csv
import time
from collections import defaultdict

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
import numpy as np

from src.analytics.pass_detector import PassDetector
from src.visualization.pitch_plots import draw_pitch
from src.visualization.overlays import show_frame, close_all_windows

# Constants
FPS = 25.0
POPUP_DURATION_FRAMES = int(FPS * 1.5) # Show popup for 1.5 seconds

# Colors matching pitch_plots.py
TEAM_A_COLOR = (200, 200, 255) # Light Red/Navy equivalent mapped in BGR -> actually it was Light Red/White
TEAM_B_COLOR = (50, 200, 50)   # Green


def draw_event_popup(frame: np.ndarray, active_event: dict) -> np.ndarray:
    if not active_event:
        return frame
        
    out = frame.copy()
    h, w = out.shape[:2]
    
    evt_type = active_event['event_type']
    from_p = active_event['from_player']
    to_p = active_event['to_player']
    from_t = active_event['from_team']
    to_t = active_event['to_team']
    
    if evt_type == "Pass":
        text = f"COMPLETED PASS: {from_t} (#{from_p} -> #{to_p})"
        bg_color = (130, 200, 130) if from_t == "Team B" else (200, 130, 130)
    else:
        text = f"TURNOVER! {from_t} #{from_p} intercepted by {to_t} #{to_p}"
        bg_color = (0, 0, 255) # Red for turnover

    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    
    # Draw centered popup at the bottom third
    x = (w - tw) // 2
    y = int(h * 0.85)
    
    cv2.rectangle(out, (x - 10, y - th - 10), (x + tw + 10, y + 10), bg_color, cv2.FILLED)
    cv2.rectangle(out, (x - 10, y - th - 10), (x + tw + 10, y + 10), (255, 255, 255), 2)
    cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    return out


def generate_pass_network(player_csv: str, events: list, output_path: str):
    """Draws a top-down tactical pass network."""
    # 1. Calculate average positions for all players
    player_positions = defaultdict(list)
    player_teams = {}
    
    with open(player_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = int(row['track_id'])
            x = float(row['pitch_x_m'])
            y = float(row['pitch_y_m'])
            team = row['team_label']
            if team in ["Team A", "Team B"] and 0 <= x <= 105 and 0 <= y <= 68:
                player_positions[tid].append((x, y))
                player_teams[tid] = team

    avg_pos = {}
    for tid, pos_list in player_positions.items():
        if len(pos_list) > 10: # Only plot players visible for at least 10 frames
            avg_x = sum(p[0] for p in pos_list) / len(pos_list)
            avg_y = sum(p[1] for p in pos_list) / len(pos_list)
            avg_pos[tid] = (avg_x, avg_y)

    # 2. Count passes between players
    pass_counts = defaultdict(int)
    for evt in events:
        if evt['event_type'] == "Pass":
            pair = tuple(sorted((evt['from_player'], evt['to_player'])))
            pass_counts[pair] += 1

    # 3. Draw Canvas
    PITCH_LENGTH = 105.0
    PITCH_WIDTH = 68.0
    margin = 40
    canvas_w = 800
    canvas_h = int(canvas_w * (PITCH_WIDTH / PITCH_LENGTH))
    
    canvas = draw_pitch(canvas_w, canvas_h)
    
    def m_to_px(xm, ym):
        px = int((xm / PITCH_LENGTH) * canvas_w)
        py = int((ym / PITCH_WIDTH) * canvas_h)
        return px, py

    # 4. Draw Edges (Passes)
    for (p1, p2), count in pass_counts.items():
        if p1 in avg_pos and p2 in avg_pos:
            pt1 = m_to_px(*avg_pos[p1])
            pt2 = m_to_px(*avg_pos[p2])
            thickness = min(1 + count, 5) # Scale thickness by volume
            color = (255, 255, 255)       # White lines
            cv2.line(canvas, pt1, pt2, color, thickness, cv2.LINE_AA)

    # 5. Draw Nodes (Players)
    for tid, (xm, ym) in avg_pos.items():
        pt = m_to_px(xm, ym)
        team = player_teams[tid]
        
        # Determine color (BGR)
        # Assuming Team A is light red visually, Team B is green
        node_color = (130, 130, 200) if team == "Team A" else (80, 200, 80)
        
        # Circle
        cv2.circle(canvas, pt, 12, node_color, -1, cv2.LINE_AA)
        cv2.circle(canvas, pt, 12, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Text (Track ID)
        (tw, th), _ = cv2.getTextSize(str(tid), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.putText(canvas, str(tid), (pt[0] - tw//2, pt[1] + th//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # Add title
    cv2.putText(canvas, "Pass Network & Average Positions", (20, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(output_path, canvas)
    print(f"  [Output] Pass Network saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 12: Pass Detection")
    parser.add_argument("--seq_dir", default="data/raw/SNMOT-062")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--no_viewer", action="store_true")
    args = parser.parse_args()

    seq_name = os.path.basename(args.seq_dir)
    img_dir = os.path.join(args.seq_dir, "img1")
    frames = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png"))])
    total_frames = len(frames)

    # Inputs
    player_csv = os.path.join(args.output_dir, f"{seq_name}_phase9_pitch_coords.csv")
    possession_csv = os.path.join(args.output_dir, f"{seq_name}_phase11_possession.csv")
    
    for fpath in [player_csv, possession_csv]:
        if not os.path.exists(fpath):
            print(f"[ERROR] Missing required file: {fpath}")
            sys.exit(1)

    print(f"\n  Phase 12 -- Pass Detection")
    print(f"  Sequence : {seq_name} ({total_frames} frames)")
    
    # 1. Detect Passes
    detector = PassDetector()
    events = detector.detect_passes(possession_csv, player_csv)
    print(f"  Detected {len(events)} events (Passes & Turnovers).")

    # 2. Write Events CSV
    out_csv = os.path.join(args.output_dir, f"{seq_name}_phase12_events.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=['start_frame', 'end_frame', 'from_player', 'to_player', 'from_team', 'to_team', 'event_type'])
        writer.writeheader()
        writer.writerows(events)
    print(f"  [Output] Event log saved to: {out_csv}")

    # 3. Generate Pass Network Diagram
    network_img = os.path.join(args.output_dir, f"{seq_name}_phase12_pass_network.jpg")
    generate_pass_network(player_csv, events, network_img)

    # 4. Generate Annotated Video
    vid_path = os.path.join(args.output_dir, f"{seq_name}_phase12_passes.mp4")
    
    first = cv2.imread(os.path.join(img_dir, frames[0]))
    fh, fw = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(vid_path, fourcc, FPS, (fw, fh))

    # Map frames to active events for fast lookup
    active_events = {}
    for evt in events:
        # Show popup starting from end_frame (when the pass is completed)
        # for a duration of POPUP_DURATION_FRAMES
        start = evt['end_frame']
        end = min(total_frames, start + POPUP_DURATION_FRAMES)
        for fnum in range(start, end + 1):
            active_events[fnum] = evt

    t0 = time.time()
    for fi, fname in enumerate(frames):
        frame_num = fi + 1
        frame = cv2.imread(os.path.join(img_dir, fname))
        
        # Get active event popup
        current_event = active_events.get(frame_num, None)
        final_frame = draw_event_popup(frame, current_event)
        
        # Add basic HUD
        cv2.putText(final_frame, f"Phase 12: Passes | Frame {frame_num}", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                    
        writer.write(final_frame)
        
        if not args.no_viewer:
            show_frame(final_frame, "Phase 12 -- Pass Detection")
            
        if frame_num % 100 == 0 or frame_num == total_frames:
            print(f"  Rendering Video... Frame {frame_num}/{total_frames}")

    writer.release()
    close_all_windows()
    
    elapsed = time.time() - t0
    print(f"\n  Phase 12 Video Render Complete in {elapsed:.1f}s")
    print(f"  Output Video: {vid_path}\n")

if __name__ == "__main__":
    main()
