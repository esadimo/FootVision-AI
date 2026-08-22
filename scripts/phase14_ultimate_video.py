"""
FootVision AI -- Phase 14: Ultimate Presentable Dashboard Video

Combines Phase 8 (Teams), Phase 9 (Minimap Radar), Phase 10 (Ball Tracking), 
Phase 11 (Possession), Phase 12 (Events), and Phase 13 (Team Metrics) into 
a single, highly-polished broadcast video overlay.

Usage:
    python scripts/phase14_ultimate_video.py [--seq_dir data/raw/SNMOT-062]
"""

import os
import sys
import argparse
import csv
import time
from collections import defaultdict, deque

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
import numpy as np

from src.visualization.pitch_plots import draw_pitch
from src.visualization.overlays import show_frame, close_all_windows
from src.calibration.homography import load_homography, get_homography_for_frame, project_point

FPS = 25.0
TEAM_A_COLOR = (200, 130, 130) # BGR
TEAM_B_COLOR = (80, 200, 80)
REF_COLOR = (0, 215, 255)
STAFF_COLOR = (128, 128, 128)

def hex_to_bgr(h):
    return tuple(int(h[i:i+2], 16) for i in (4, 2, 0))

# ─── Data Loaders ─────────────────────────────────────────────────────────────
def load_phase9(path):
    data = defaultdict(list)
    with open(path, 'r') as f:
        for row in csv.DictReader(f):
            data[int(row['frame_number'])].append({
                'track_id': int(row['track_id']),
                'team': row['team_label'],
                'bbox': (float(row['bbox_x1']), float(row['bbox_y1']), float(row['bbox_x2']), float(row['bbox_y2'])),
                'foot': (float(row['foot_x_px']), float(row['foot_y_px'])),
                'pitch': (float(row['pitch_x_m']), float(row['pitch_y_m']))
            })
    return data

def load_phase10(path):
    data = {}
    with open(path, 'r') as f:
        for row in csv.DictReader(f):
            if int(row['ball_detected']) == 1:
                data[int(row['frame_number'])] = (float(row['x_center']), float(row['y_center']))
    return data

def load_phase11(path):
    data = {}
    with open(path, 'r') as f:
        for row in csv.DictReader(f):
            data[int(row['frame_number'])] = {
                'poss_team': row['possession_team'],
                'closest_id': int(row['closest_player_id']),
                'dist_m': float(row['closest_distance_m']),
                'pct_A': float(row['team_A_pct']),
                'pct_B': float(row['team_B_pct'])
            }
    return data

def load_phase12(path):
    # Returns active events per frame
    events = defaultdict(list)
    with open(path, 'r') as f:
        for row in csv.DictReader(f):
            end_f = int(row['end_frame'])
            # Show event popup for 40 frames (~1.5s) after pass completion
            for i in range(end_f, end_f + 40):
                events[i].append(row)
    return events

def load_phase13(path):
    data = {}
    with open(path, 'r') as f:
        for row in csv.DictReader(f):
            data[int(row['frame_number'])] = {
                'A_w': float(row['Team A_width_m']), 'A_d': float(row['Team A_depth_m']),
                'B_w': float(row['Team B_width_m']), 'B_d': float(row['Team B_depth_m'])
            }
    return data


# ─── Drawing Components ───────────────────────────────────────────────────────
def draw_transparent_overlay(frame, x, y, w, h, color, alpha=0.6):
    sub_img = frame[y:y+h, x:x+w]
    rect = np.full(sub_img.shape, color, dtype=np.uint8)
    frame[y:y+h, x:x+w] = cv2.addWeighted(sub_img, 1 - alpha, rect, alpha, 0)

def render_dashboard(frame, f_num, p8, p10, p11, p12, p13, ball_trail, H):
    h, w = frame.shape[:2]
    out = frame.copy()
    
    # 1. Player Bboxes (Phase 8)
    for p in p8:
        color = TEAM_A_COLOR if p['team'] == "Team A" else (TEAM_B_COLOR if p['team'] == "Team B" else (REF_COLOR if p['team'] == "Referee" else STAFF_COLOR))
        x1, y1, x2, y2 = map(int, p['bbox'])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        
        # ID tag
        tag = f"#{p['track_id']}"
        cv2.rectangle(out, (x1, max(0, y1-20)), (x1+40, y1), color, cv2.FILLED)
        cv2.putText(out, tag, (x1+2, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1, cv2.LINE_AA)

    # 2. Possession Connection (Phase 11)
    if p10 and p11['closest_id'] != -1:
        closest = next((x for x in p8 if x['track_id'] == p11['closest_id']), None)
        if closest:
            c_color = (0, 255, 0) if p11['dist_m'] <= 2.5 else (0, 0, 255)
            thick = 2 if p11['dist_m'] <= 2.5 else 1
            bx, by = map(int, p10)
            px, py = map(int, closest['foot'])
            cv2.line(out, (bx, by), (px, py), c_color, thick, cv2.LINE_AA)
            cv2.putText(out, f"{p11['dist_m']:.1f}m", ((bx+px)//2, (by+py)//2 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c_color, 2, cv2.LINE_AA)

    # 3. Ball Tracking (Phase 10)
    if p10 is not None and len(ball_trail) > 1:
        for i in range(1, len(ball_trail)):
            f_prev, pt_prev = ball_trail[i-1]
            f_curr, pt_curr = ball_trail[i]
            
            # Only connect if consecutive frames and physical distance < 60px per frame
            if f_curr - f_prev == 1:
                dist_px = np.hypot(pt_curr[0] - pt_prev[0], pt_curr[1] - pt_prev[1])
                if dist_px < 60.0:
                    # Clean comet tail
                    thickness = max(1, int(3 * (i / len(ball_trail))))
                    cv2.line(out, tuple(map(int, pt_prev)), tuple(map(int, pt_curr)), (0, 140, 255), thickness, cv2.LINE_AA)
                    
    if p10:
        bx, by = map(int, p10)
        cv2.circle(out, (bx, by), 10, (0, 215, 255), 2, cv2.LINE_AA)
        cv2.circle(out, (bx, by), 2, (255, 255, 255), -1)

    # 4. Top HUD Banner (Phase 11 & 13)
    draw_transparent_overlay(out, 0, 0, w, 60, (0,0,0), alpha=0.7)
    
    cv2.putText(out, f"MATCH DASHBOARD  |  Frame {f_num}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    
    # Possession Stats
    p_text = f"POSSESSION: TEAM A {p11['pct_A']:.1f}%  |  TEAM B {p11['pct_B']:.1f}%"
    cv2.putText(out, p_text, (400, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    
    # Team Metrics
    if p13:
        m_text = f"TEAM A: Width {p13['A_w']:.1f}m, Depth {p13['A_d']:.1f}m   |   TEAM B: Width {p13['B_w']:.1f}m, Depth {p13['B_d']:.1f}m"
        cv2.putText(out, m_text, (950, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1, cv2.LINE_AA)

    # 5. Event Popups (Phase 12)
    if p12:
        evt = p12[0]
        text = f"COMPLETED PASS: {evt['from_team']} #{evt['from_player']} -> #{evt['to_player']}" if evt['event_type'] == "Pass" else f"TURNOVER! {evt['to_team']} #{evt['to_player']} intercepted"
        bg = TEAM_A_COLOR if (evt['event_type'] == "Pass" and evt['from_team'] == "Team A") else (TEAM_B_COLOR if (evt['event_type'] == "Pass") else (0,0,255))
        
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        ex = (w - tw) // 2
        ey = h - 80
        cv2.rectangle(out, (ex - 15, ey - th - 15), (ex + tw + 15, ey + 15), bg, cv2.FILLED)
        cv2.rectangle(out, (ex - 15, ey - th - 15), (ex + tw + 15, ey + 15), (255,255,255), 2)
        cv2.putText(out, text, (ex, ey), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

    # 7. Timeline Bar
    bar_h = 10
    cv2.rectangle(out, (0, h - bar_h), (w, h), (50, 50, 50), cv2.FILLED)
    pct_a = p11['pct_A'] / 100.0 if (p11['pct_A'] + p11['pct_B']) > 0 else 0.5
    sep_x = int(w * pct_a)
    if sep_x > 0:
        cv2.rectangle(out, (0, h - bar_h), (sep_x, h), TEAM_A_COLOR, cv2.FILLED)
    if sep_x < w:
        cv2.rectangle(out, (sep_x, h - bar_h), (w, h), TEAM_B_COLOR, cv2.FILLED)

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_dir", default="data/raw/SNMOT-062")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--no_viewer", action="store_true")
    args = parser.parse_args()

    seq = os.path.basename(args.seq_dir)
    img_dir = os.path.join(args.seq_dir, "img1")
    frames = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))])

    print("\n  Phase 14 -- Generating Ultimate Dashboard Video")
    
    # Load all data
    d8  = load_phase9(os.path.join(args.output_dir, f"{seq}_phase9_pitch_coords.csv"))
    d10 = load_phase10(os.path.join(args.output_dir, f"{seq}_phase10_ball.csv"))
    d11 = load_phase11(os.path.join(args.output_dir, f"{seq}_phase11_possession.csv"))
    d12 = load_phase12(os.path.join(args.output_dir, f"{seq}_phase12_events.csv"))
    d13 = load_phase13(os.path.join(args.output_dir, f"{seq}_phase13_team_metrics.csv"))
    
    h_file = os.path.join(args.output_dir, "homography_keyframes.json")
    if not os.path.exists(h_file):
        h_file = os.path.join(args.output_dir, "homography.npy")
    calib = load_homography(h_file)

    first = cv2.imread(os.path.join(img_dir, frames[0]))
    fh, fw = first.shape[:2]

    out_path = os.path.join(args.output_dir, f"{seq}_phase14_ultimate.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (fw, fh))

    ball_trail = deque(maxlen=12)
    t0 = time.time()

    for fi, fname in enumerate(frames):
        fnum = fi + 1
        frame = cv2.imread(os.path.join(img_dir, fname))
        
        H = get_homography_for_frame(calib, fi, fw, fh)
        
        p10 = d10.get(fnum, None)
        if p10 is not None:
            ball_trail.append((fnum, p10))
        else:
            ball_trail.clear()
            
        p8  = d8.get(fnum, [])
        p11 = d11.get(fnum, {'poss_team': 'None', 'closest_id': -1, 'dist_m': 99, 'pct_A': 50, 'pct_B': 50})
        p12 = d12.get(fnum, [])
        p13 = d13.get(fnum, None)

        final_frame = render_dashboard(frame, fnum, p8, p10, p11, p12, p13, ball_trail, H)
        writer.write(final_frame)
        
        if not args.no_viewer:
            show_frame(final_frame, "FootVision AI Ultimate Dashboard")
            
        if fnum % 50 == 0 or fnum == len(frames):
            el = time.time() - t0
            fps = fnum / el if el > 0 else 0
            eta = (len(frames) - fnum) / fps if fps > 0 else 0
            print(f"  Frame {fnum}/{len(frames)} | {fps:.1f} FPS | ETA {eta:.0f}s")

    writer.release()
    close_all_windows()
    print(f"\n  [Success] Ultimate Video saved to: {out_path}\n")

if __name__ == "__main__":
    main()
