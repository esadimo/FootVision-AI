"""
FootVision AI — Phase 7
Multi-Object Player Tracking (ByteTrack)

Objective:
  Maintain persistent temporal identities (track_id) across consecutive frames:
    - Replace frame-isolated detection IDs with stable track IDs over time.
    - Leverage ByteTrack (two-stage high/low confidence association + Kalman Filter).
    - Draw distinct visual colors and trajectory trails for each tracked player.
    - Compute tracking continuity metrics: unique tracks created, average track lifetime,
      and track stability.
    - Export structured tracking dataset (.csv) and annotated tracking video (.mp4).

Output CSV Schema:
    frame_number, timestamp, track_id, class_name, confidence,
    x1, y1, x2, y2, center_x, center_y, bottom_center_x, bottom_center_y

Usage:
    python scripts/phase7_player_tracking.py [options]

    --seq_dir      MOT sequence root (default: data/raw/SNMOT-062)
    --threshold    Detection confidence cutoff (default: 0.20)
    --output_dir   Output directory (default: outputs/)
    --trail_len    Number of previous frames to draw in trajectory trail (default: 30)
    --no_viewer    Skip live OpenCV popup window
    --max_frames   Process only first N frames (0 = all)
"""

import os
import sys
import time
import csv
import argparse
from collections import defaultdict, deque
from typing import Dict, List, Tuple

# ── Project-root import fix ───────────────────────────────────────────────────
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
import numpy as np
from tqdm import tqdm


# ─── Color Palette Generator for Consistent Track IDs ─────────────────────────

def get_track_color(track_id: int) -> Tuple[int, int, int]:
    """
    Generates a distinct, deterministic BGR color for any track ID using HSV hashing.
    """
    # Golden ratio angle distribution for maximum color separation
    hue = int((track_id * 137.508) % 180)
    # Bright and saturated in HSV
    hsv_color = np.uint8([[[hue, 230, 255]]])
    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2])


# ─── Drawing Helpers ─────────────────────────────────────────────────────────

def draw_track(frame: np.ndarray,
               x1: int, y1: int, x2: int, y2: int,
               track_id: int,
               conf: float,
               color: Tuple[int, int, int],
               trail_points: List[Tuple[int, int]]) -> None:
    """
    Renders player bounding box, track ID banner, foot contact dot, and motion trail.
    """
    # 1. Draw trajectory trail (historical foot positions)
    if len(trail_points) > 1:
        for i in range(1, len(trail_points)):
            pt1 = trail_points[i - 1]
            pt2 = trail_points[i]
            # Trail fades slightly towards older points
            thickness = max(1, int(3 * (i / len(trail_points))))
            cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)

    # 2. Bounding Box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # 3. Track ID & Confidence Header
    label = f"ID:{track_id} ({conf:.2f})"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
    
    # Background banner above box
    banner_y1 = max(0, y1 - th - 8)
    banner_y2 = max(th + 8, y1)
    cv2.rectangle(frame, (x1, banner_y1), (x1 + tw + 8, banner_y2), color, cv2.FILLED)
    
    # White text label
    cv2.putText(frame, label, (x1 + 4, banner_y2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

    # 4. Foot position anchor (bottom-center)
    bx = int((x1 + x2) / 2)
    by = int(y2)
    cv2.circle(frame, (bx, by), 4, color, -1)
    cv2.circle(frame, (bx, by), 5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_tracking_hud(frame: np.ndarray,
                      frame_number: int,
                      total_frames: int,
                      active_tracks: int,
                      cumulative_tracks: int,
                      fps: float) -> None:
    """Draws tracking telemetry overlay strip at the top of the canvas."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 38), (15, 15, 15), cv2.FILLED)
    text = (f"Frame {frame_number:04d}/{total_frames}  |  "
            f"Active Players: {active_tracks:2d}  |  "
            f"Cumulative Unique Tracks: {cumulative_tracks:3d}  |  "
            f"Tracking Speed: {fps:.1f} fps")
    cv2.putText(frame, text, (14, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 230, 230), 1, cv2.LINE_AA)


# ─── Main Tracking Pipeline ───────────────────────────────────────────────────

def run_tracking_pipeline(seq_dir: str,
                          threshold: float,
                          output_dir: str,
                          trail_len: int,
                          show_viewer: bool,
                          max_frames: int) -> None:

    from ultralytics import YOLO
    from src.visualization.overlays import show_frame, close_all_windows

    # ── Locate Sequence Assets ────────────────────────────────────────────
    img_dir      = os.path.join(seq_dir, "img1")
    seqinfo_path = os.path.join(seq_dir, "seqinfo.ini")

    valid_ext = (".jpg", ".jpeg", ".png")
    all_files = sorted([f for f in os.listdir(img_dir)
                        if f.lower().endswith(valid_ext)])

    # Read FPS
    fps_src = 25.0
    if os.path.exists(seqinfo_path):
        with open(seqinfo_path) as f:
            for line in f:
                if line.startswith("frameRate="):
                    fps_src = float(line.split("=")[1])
                    break

    total_frames = len(all_files)
    if max_frames > 0:
        all_files = all_files[:max_frames]
        total_frames = len(all_files)

    # Read image dimensions from first frame
    first_frame = cv2.imread(os.path.join(img_dir, all_files[0]))
    frame_h, frame_w = first_frame.shape[:2]

    # ── Output Paths ──────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    seq_name    = os.path.basename(seq_dir.rstrip("/\\"))
    video_path  = os.path.join(output_dir, f"{seq_name}_phase7_tracking.mp4")
    csv_path    = os.path.join(output_dir, f"{seq_name}_phase7_tracks.csv")
    report_path = os.path.join(output_dir, f"{seq_name}_phase7_report.txt")

    # ── Video & CSV Writers ───────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps_src, (frame_w, frame_h))

    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame_number", "timestamp", "track_id", "class_name", "confidence",
        "x1", "y1", "x2", "y2",
        "center_x", "center_y",
        "bottom_center_x", "bottom_center_y"
    ])

    # ── Tracking State Management ─────────────────────────────────────────
    # History buffer for drawing motion paths: {track_id: deque([(bx, by), ...])}
    trail_buffers: Dict[int, deque] = defaultdict(lambda: deque(maxlen=trail_len))
    
    # Statistical track life tracking: {track_id: [frame_first_seen, frame_last_seen, total_detections]}
    track_stats: Dict[int, Dict] = {}

    # ── Load Model ────────────────────────────────────────────────────────
    print(f"\n{'=' * 75}")
    print(f"  FootVision AI — Phase 7: Multi-Object Player Tracking (ByteTrack)")
    print(f"{'=' * 75}")
    print(f"  Sequence       : {seq_dir}")
    print(f"  Total Frames   : {total_frames} ({total_frames / fps_src:.1f}s @ {fps_src} fps)")
    print(f"  Confidence Cut : {threshold:.2f}")
    print(f"  Output Video   : {video_path}")
    print(f"  Output CSV     : {csv_path}")
    print(f"\n  Initializing YOLOv8 + ByteTrack tracker...")
    model = YOLO("yolov8n.pt")
    print(f"  Tracker initialized. Starting tracking pipeline...\n")

    t_start = time.perf_counter()
    total_tracked_records = 0

    for frame_idx, fname in enumerate(tqdm(all_files, desc="  Tracking", unit="frame")):
        frame_number = frame_idx + 1
        timestamp    = frame_idx / fps_src

        frame = cv2.imread(os.path.join(img_dir, fname))
        if frame is None:
            continue
        vis = frame.copy()

        # ── ByteTrack Inference ───────────────────────────────────────────
        # persist=True ensures ByteTrack retains track continuity across sequential calls
        results = model.track(
            source=frame,
            conf=threshold,
            classes=[0],             # COCO index 0 is 'person'
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False
        )[0]

        active_this_frame = 0
        boxes = results.boxes

        # ── Parse Tracked Entities ────────────────────────────────────────
        if boxes is not None and boxes.id is not None:
            track_ids = boxes.id.int().cpu().tolist()
            confs     = boxes.conf.cpu().tolist()
            xyxys     = boxes.xyxy.int().cpu().tolist()

            for track_id, conf, (x1, y1, x2, y2) in zip(track_ids, confs, xyxys):
                active_this_frame += 1
                total_tracked_records += 1

                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                bx = float(cx)
                by = float(y2)

                # Record statistics
                if track_id not in track_stats:
                    track_stats[track_id] = {
                        "first_frame": frame_number,
                        "last_frame": frame_number,
                        "hits": 1
                    }
                else:
                    track_stats[track_id]["last_frame"] = frame_number
                    track_stats[track_id]["hits"] += 1

                # Update trajectory trail
                trail_buffers[track_id].append((int(bx), int(by)))

                # Get color & draw track
                color = get_track_color(track_id)
                draw_track(vis, x1, y1, x2, y2, track_id, conf, color, list(trail_buffers[track_id]))

                # Write CSV
                csv_writer.writerow([
                    frame_number,
                    f"{timestamp:.3f}",
                    track_id,
                    "person",
                    f"{conf:.4f}",
                    x1, y1, x2, y2,
                    f"{cx:.1f}", f"{cy:.1f}",
                    f"{bx:.1f}", f"{by:.1f}"
                ])

        # ── Draw HUD ──────────────────────────────────────────────────────
        elapsed = time.perf_counter() - t_start
        fps_proc = frame_number / elapsed if elapsed > 0 else 0.0
        cumulative_tracks = len(track_stats)
        draw_tracking_hud(vis, frame_number, total_frames, active_this_frame, cumulative_tracks, fps_proc)

        # ── Write Video ───────────────────────────────────────────────────
        writer.write(vis)

        # ── Live Viewer ───────────────────────────────────────────────────
        if show_viewer:
            keep = show_frame("FootVision AI — Phase 7 Tracking", vis, delay_ms=1)
            if not keep:
                print("\n  [Viewer] Quit requested. Ending pipeline early.")
                break

    # ── Finalize Resources ────────────────────────────────────────────────
    writer.release()
    csv_file.close()
    if show_viewer:
        close_all_windows()

    t_elapsed = time.perf_counter() - t_start

    # ── Compute Tracking Analytics ────────────────────────────────────────
    total_unique_tracks = len(track_stats)
    track_durations_frames = [s["last_frame"] - s["first_frame"] + 1 for s in track_stats.values()]
    track_hits = [s["hits"] for s in track_stats.values()]
    
    avg_duration_frames = np.mean(track_durations_frames) if track_durations_frames else 0.0
    avg_duration_sec    = avg_duration_frames / fps_src
    median_duration_sec = np.median(track_durations_frames) / fps_src if track_durations_frames else 0.0
    
    # Long-lived tracks (tracked for >= 5 continuous seconds)
    stable_tracks = sum(1 for d in track_durations_frames if (d / fps_src) >= 5.0)
    stable_pct = (stable_tracks / total_unique_tracks * 100.0) if total_unique_tracks > 0 else 0.0

    # ── Print & Save Report ───────────────────────────────────────────────
    report_lines = [
        "FootVision AI — Phase 7 Multi-Object Tracking Report",
        "=" * 65,
        f"Sequence               : {seq_dir}",
        f"Total Frames Processed : {total_frames} ({total_frames / fps_src:.1f} seconds)",
        f"Detector Confidence    : {threshold:.2f}",
        "",
        "TRACKING CONTINUITY METRICS",
        f"  Total Unique Tracks  : {total_unique_tracks}",
        f"  Total Track Detections : {total_tracked_records}",
        f"  Average Active Players : {total_tracked_records / total_frames:.2f} / frame",
        f"  Average Track Lifetime : {avg_duration_sec:.2f} seconds ({avg_duration_frames:.1f} frames)",
        f"  Median Track Lifetime  : {median_duration_sec:.2f} seconds",
        f"  Stable Tracks (>= 5s)  : {stable_tracks} of {total_unique_tracks} ({stable_pct:.1f}%)",
        "",
        "TIMING & SPEED",
        f"  Total Processing Time  : {t_elapsed:.2f} seconds",
        f"  Average Throughput     : {total_frames / t_elapsed:.2f} FPS",
        "",
        "OUTPUT DELIVERABLES",
        f"  Annotated Track Video  : {video_path}",
        f"  Track Data Table (CSV) : {csv_path}",
    ]

    print(f"\n{'=' * 75}")
    for line in report_lines:
        print(f"  {line}")
    print(f"{'=' * 75}\n")

    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"  Summary report written to: {report_path}\n")


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 7: Multi-Object Player Tracking using ByteTrack and YOLOv8."
    )
    parser.add_argument("--seq_dir",    default="data/raw/SNMOT-062",
                        help="Path to MOT sequence directory")
    parser.add_argument("--threshold",  type=float, default=0.20,
                        help="Confidence cutoff for detector (default: 0.20)")
    parser.add_argument("--output_dir", default="outputs",
                        help="Output directory (default: outputs/)")
    parser.add_argument("--trail_len",  type=int, default=30,
                        help="Length of historical motion trail points (default: 30)")
    parser.add_argument("--no_viewer",  action="store_true",
                        help="Disable live popup viewer for headless batch execution")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="Process only first N frames (0 = all)")
    args = parser.parse_args()

    try:
        run_tracking_pipeline(
            seq_dir     = args.seq_dir,
            threshold   = args.threshold,
            output_dir  = args.output_dir,
            trail_len   = args.trail_len,
            show_viewer = not args.no_viewer,
            max_frames  = args.max_frames
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
