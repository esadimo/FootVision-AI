"""
src.analytics.statistics — Calculates player physical metrics, team spatial metrics, and heatmaps.
"""

import csv
import math
from collections import defaultdict
import numpy as np
import cv2

from src.visualization.pitch_plots import draw_pitch

def _moving_average(points, window=5):
    """Applies a simple moving average to smooth (x, y) trajectories and reduce bounding-box jitter."""
    if len(points) < window:
        return points
    
    smoothed = []
    for i in range(len(points)):
        start = max(0, i - window // 2)
        end = min(len(points), i + window // 2 + 1)
        slice_pts = points[start:end]
        avg_x = sum(p[0] for p in slice_pts) / len(slice_pts)
        avg_y = sum(p[1] for p in slice_pts) / len(slice_pts)
        smoothed.append((avg_x, avg_y))
    return smoothed

def calculate_player_stats(coords_csv: str, pass_events_csv: str, fps: float = 25.0):
    """
    Calculates distance covered and top speed for each player.
    """
    trajectories = defaultdict(list)
    teams = {}
    
    # Read tracking data
    with open(coords_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = int(row['track_id'])
            team = row['team_label']
            x = float(row['pitch_x_m'])
            y = float(row['pitch_y_m'])
            
            if 0 <= x <= 105 and 0 <= y <= 68: # Filter out-of-bounds anomalies
                trajectories[tid].append((x, y))
                teams[tid] = team

    # Read pass data to count passes per player
    passes_made = defaultdict(int)
    if pass_events_csv:
        try:
            with open(pass_events_csv, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['event_type'] == "Pass":
                        passes_made[int(row['from_player'])] += 1
        except FileNotFoundError:
            pass

    stats = []
    for tid, pts in trajectories.items():
        if len(pts) < fps: # Ignore players visible for less than 1 second
            continue
            
        smoothed_pts = _moving_average(pts, window=11) # ~0.4s window
        
        total_dist = 0.0
        max_speed_ms = 0.0
        
        for i in range(1, len(smoothed_pts)):
            dx = smoothed_pts[i][0] - smoothed_pts[i-1][0]
            dy = smoothed_pts[i][1] - smoothed_pts[i-1][1]
            dist = math.hypot(dx, dy)
            total_dist += dist
            
            speed = dist * fps # m/s
            if speed > max_speed_ms and speed < 12.0: # Cap at ~43 km/h to reject remaining teleportation glitches
                max_speed_ms = speed
                
        time_visible_s = len(pts) / fps
        avg_speed_kmh = (total_dist / time_visible_s) * 3.6 if time_visible_s > 0 else 0
        max_speed_kmh = max_speed_ms * 3.6
        
        stats.append({
            'track_id': tid,
            'team_label': teams[tid],
            'time_visible_s': round(time_visible_s, 2),
            'distance_covered_m': round(total_dist, 2),
            'avg_speed_kmh': round(avg_speed_kmh, 2),
            'max_speed_kmh': round(max_speed_kmh, 2),
            'passes_made': passes_made[tid]
        })
        
    return sorted(stats, key=lambda x: x['distance_covered_m'], reverse=True)


def calculate_team_metrics(coords_csv: str):
    """
    Calculates frame-by-frame spatial width and depth for teams.
    Width = max(Y) - min(Y)
    Depth = max(X) - min(X)
    """
    frames = defaultdict(lambda: {'Team A': [], 'Team B': []})
    
    with open(coords_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            team = row['team_label']
            if team in ['Team A', 'Team B']:
                fnum = int(row['frame_number'])
                x = float(row['pitch_x_m'])
                y = float(row['pitch_y_m'])
                if 0 <= x <= 105 and 0 <= y <= 68:
                    frames[fnum][team].append((x, y))
                    
    metrics = []
    for fnum in sorted(frames.keys()):
        row_data = {'frame_number': fnum}
        for team in ['Team A', 'Team B']:
            pts = frames[fnum][team]
            if len(pts) >= 3:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                row_data[f'{team}_width_m'] = round(max(ys) - min(ys), 2)
                row_data[f'{team}_depth_m'] = round(max(xs) - min(xs), 2)
            else:
                row_data[f'{team}_width_m'] = 0.0
                row_data[f'{team}_depth_m'] = 0.0
        metrics.append(row_data)
        
    return metrics


def generate_team_heatmaps(coords_csv: str, output_path: str):
    """Generates and saves a side-by-side heatmap image for Team A and Team B."""
    pts_A = []
    pts_B = []
    
    with open(coords_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            team = row['team_label']
            x = float(row['pitch_x_m'])
            y = float(row['pitch_y_m'])
            if 0 <= x <= 105 and 0 <= y <= 68:
                if team == 'Team A':
                    pts_A.append((x, y))
                elif team == 'Team B':
                    pts_B.append((x, y))
                    
    PITCH_LENGTH = 105.0
    PITCH_WIDTH = 68.0
    SCALE = 8 # Pixels per meter
    
    w_px = int(PITCH_LENGTH * SCALE)
    h_px = int(PITCH_WIDTH * SCALE)
    
    def build_heatmap_canvas(pts, title):
        pitch_canvas = draw_pitch(w_px, h_px)
        
        hm = np.zeros((h_px, w_px), dtype=np.float32)
        for (x, y) in pts:
            px, py = int(x * SCALE), int(y * SCALE)
            if 0 <= px < w_px and 0 <= py < h_px:
                hm[py, px] += 1
                
        # Gaussian smoothing
        hm = cv2.GaussianBlur(hm, (0, 0), sigmaX=SCALE*2, sigmaY=SCALE*2)
        
        if np.max(hm) > 0:
            hm = hm / np.max(hm)
            
        hm_colored = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Blend only where there is heat
        mask = hm > 0.05
        out = pitch_canvas.copy()
        out[mask] = cv2.addWeighted(pitch_canvas[mask], 0.3, hm_colored[mask], 0.7, 0)
        
        # Add Title banner
        cv2.rectangle(out, (0, 0), (w_px, 40), (20, 20, 20), cv2.FILLED)
        cv2.putText(out, title, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        return out
        
    canvas_A = build_heatmap_canvas(pts_A, "Team A Heatmap")
    canvas_B = build_heatmap_canvas(pts_B, "Team B Heatmap")
    
    # White divider
    divider = np.full((h_px, 4, 3), 255, dtype=np.uint8)
    
    composite = np.hstack([canvas_A, divider, canvas_B])
    cv2.imwrite(output_path, composite)
