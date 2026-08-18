"""
FootVision AI — Phase 8
Team, Referee, and Goalkeeper/Staff Classification Pipeline

Kits Modeled:
  - Team A: White / Navy Kit
  - Team B: Green / White Kit
  - Referee: Yellow / Gold Kit
  - Staff / GK: Black / Dark Clothing

Usage:
    python scripts/phase8_team_classification.py [options]

    --seq_dir      MOT sequence root (default: data/raw/SNMOT-062)
    --threshold    Detector confidence cutoff (default: 0.20)
    --output_dir   Output directory (default: outputs/)
    --no_viewer    Run headless without OpenCV popup
    --max_frames   Process only first N frames (0 = all)
"""

import os
import sys
import time
import csv
import argparse
from typing import Dict, List, Tuple

# ── Project-root import fix ───────────────────────────────────────────────────
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
import numpy as np
from tqdm import tqdm

from src.teams.crop_extractor import extract_torso_crop
from src.teams.colour_features import extract_chest_color_metrics
from src.teams.classifier import MatchKitClassifier


# ─── Visual Rendering Helpers ────────────────────────────────────────────────

def draw_team_player(frame: np.ndarray,
                     x1: int, y1: int, x2: int, y2: int,
                     track_id: int,
                     team_label: str,
                     conf: float,
                     team_color: Tuple[int, int, int]) -> None:
    """
    Renders player bounding box with distinct team badge.
    """
    # 1. Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), team_color, 2)

    # 2. Header banner with Team name and Track ID
    label = f"{team_label} #{track_id}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
    
    banner_y1 = max(0, y1 - th - 8)
    banner_y2 = max(th + 8, y1)
    cv2.rectangle(frame, (x1, banner_y1), (x1 + tw + 8, banner_y2), team_color, cv2.FILLED)
    
    # Text contrast (dark text on bright banners, white text on dark banners)
    brightness = team_color[0] * 0.114 + team_color[1] * 0.587 + team_color[2] * 0.299
    text_color = (0, 0, 0) if brightness > 150 else (255, 255, 255)
    
    cv2.putText(frame, label, (x1 + 4, banner_y2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, text_color, 1, cv2.LINE_AA)

    # 3. Feet position marker
    bx = int((x1 + x2) / 2)
    by = int(y2)
    cv2.circle(frame, (bx, by), 4, team_color, -1)
    cv2.circle(frame, (bx, by), 5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_kit_hud(frame: np.ndarray,
                 frame_number: int,
                 total_frames: int,
                 counts: Dict[str, int],
                 colors: Dict[str, Tuple[int, int, int]],
                 fps: float) -> None:
    """Draws top HUD bar showing live counts for Team A, Team B, Referees, and Staff/GKs."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 40), (18, 18, 18), cv2.FILLED)
    
    info_text = f"Frame {frame_number:04d}/{total_frames}  |  Speed: {fps:.1f} fps"
    cv2.putText(frame, info_text, (12, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1, cv2.LINE_AA)

    # Team A badge (White)
    bx_a = w - 550
    cv2.circle(frame, (bx_a, 20), 6, colors.get("Team A", (240, 240, 240)), -1)
    cv2.putText(frame, f"Team A (White): {counts.get('Team A', 0):2d}", (bx_a + 10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (240, 240, 240), 1, cv2.LINE_AA)

    # Team B badge (Green)
    bx_b = w - 380
    cv2.circle(frame, (bx_b, 20), 6, colors.get("Team B", (35, 180, 50)), -1)
    cv2.putText(frame, f"Team B (Green): {counts.get('Team B', 0):2d}", (bx_b + 10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (35, 210, 50), 1, cv2.LINE_AA)

    # Ref badge (Yellow)
    bx_r = w - 220
    cv2.circle(frame, (bx_r, 20), 6, colors.get("Referee", (0, 215, 255)), -1)
    cv2.putText(frame, f"Ref: {counts.get('Referee', 0):2d}", (bx_r + 10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 215, 255), 1, cv2.LINE_AA)

    # Staff / GK badge (Black)
    bx_g = w - 110
    cv2.circle(frame, (bx_g, 20), 6, colors.get("Staff/GK", (45, 45, 45)), -1)
    cv2.circle(frame, (bx_g, 20), 7, (200, 200, 200), 1)
    cv2.putText(frame, f"Staff/GK: {counts.get('Staff/GK', 0):2d}", (bx_g + 10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1, cv2.LINE_AA)


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def run_team_classification_pipeline(seq_dir: str,
                                     threshold: float,
                                     output_dir: str,
                                     show_viewer: bool,
                                     max_frames: int) -> None:

    from ultralytics import YOLO
    from src.visualization.overlays import show_frame, close_all_windows

    img_dir      = os.path.join(seq_dir, "img1")
    seqinfo_path = os.path.join(seq_dir, "seqinfo.ini")

    valid_ext = (".jpg", ".jpeg", ".png")
    all_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(valid_ext)])
    
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

    first_frame = cv2.imread(os.path.join(img_dir, all_files[0]))
    frame_h, frame_w = first_frame.shape[:2]

    os.makedirs(output_dir, exist_ok=True)
    seq_name    = os.path.basename(seq_dir.rstrip("/\\"))
    video_path  = os.path.join(output_dir, f"{seq_name}_phase8_teams.mp4")
    csv_path    = os.path.join(output_dir, f"{seq_name}_phase8_teams.csv")
    report_path = os.path.join(output_dir, f"{seq_name}_phase8_report.txt")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps_src, (frame_w, frame_h))

    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame_number", "timestamp", "track_id", "team_label", "class_name", "confidence",
        "x1", "y1", "x2", "y2",
        "center_x", "center_y",
        "bottom_center_x", "bottom_center_y"
    ])

    print(f"\n{'=' * 75}")
    print(f"  FootVision AI — Phase 8: Team, Referee & Staff/GK Classification")
    print(f"{'=' * 75}")
    print(f"  Sequence     : {seq_dir}")
    print(f"  Frames       : {total_frames} ({total_frames / fps_src:.1f}s @ {fps_src} fps)")
    print(f"  Kit Profiles : Team A (White/Navy), Team B (Green/White), Ref (Yellow), Staff/GK (Black)")
    print(f"  Output Video : {video_path}")
    print(f"  Output CSV   : {csv_path}")

    model = YOLO("yolov8n.pt")
    kit_classifier = MatchKitClassifier()

    print(f"\n  Starting full sequence tracking & kit classification...\n")
    t_start = time.perf_counter()

    cumulative_counts = {"Team A": 0, "Team B": 0, "Referee": 0, "Staff/GK": 0, "Unknown": 0}
    total_detections = 0

    for frame_idx, fname in enumerate(tqdm(all_files, desc="  Classifying", unit="frame")):
        frame_number = frame_idx + 1
        timestamp    = frame_idx / fps_src

        frame = cv2.imread(os.path.join(img_dir, fname))
        if frame is None:
            continue
        vis = frame.copy()

        # Track players with ByteTrack
        results = model.track(
            source=frame,
            conf=threshold,
            classes=[0],
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False
        )[0]

        frame_counts = {"Team A": 0, "Team B": 0, "Referee": 0, "Staff/GK": 0}
        boxes = results.boxes

        if boxes is not None and boxes.id is not None:
            track_ids = boxes.id.int().cpu().tolist()
            confs     = boxes.conf.cpu().tolist()
            xyxys     = boxes.xyxy.int().cpu().tolist()

            for track_id, conf, (x1, y1, x2, y2) in zip(track_ids, confs, xyxys):
                total_detections += 1
                
                # 1. Extract chest crop & calculate color metrics
                torso = extract_torso_crop(frame, (x1, y1, x2, y2))
                instant_label = "Unknown"
                
                if torso is not None:
                    metrics = extract_chest_color_metrics(torso)
                    instant_label = kit_classifier.predict_single(metrics)

                # 2. Temporal smoothing per track ID
                stable_label = kit_classifier.update_track(track_id, instant_label, min_votes=3)
                
                frame_counts[stable_label] = frame_counts.get(stable_label, 0) + 1
                cumulative_counts[stable_label] = cumulative_counts.get(stable_label, 0) + 1

                # 3. Canvas rendering
                color = kit_classifier.get_color(stable_label)
                draw_team_player(vis, x1, y1, x2, y2, track_id, stable_label, conf, color)

                # 4. Save CSV record
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                bx = float(cx)
                by = float(y2)

                csv_writer.writerow([
                    frame_number,
                    f"{timestamp:.3f}",
                    track_id,
                    stable_label,
                    "person",
                    f"{conf:.4f}",
                    x1, y1, x2, y2,
                    f"{cx:.1f}", f"{cy:.1f}",
                    f"{bx:.1f}", f"{by:.1f}"
                ])

        # ── Draw HUD ──────────────────────────────────────────────────────
        elapsed = time.perf_counter() - t_start
        fps_proc = frame_number / elapsed if elapsed > 0 else 0.0
        draw_kit_hud(vis, frame_number, total_frames, frame_counts, kit_classifier.team_colors_bgr, fps_proc)

        # ── Write Video ───────────────────────────────────────────────────
        writer.write(vis)

        # ── Live Viewer ───────────────────────────────────────────────────
        if show_viewer:
            keep = show_frame("FootVision AI — Phase 8 Team Classification", vis, delay_ms=1)
            if not keep:
                print("\n  [Viewer] Quit requested. Ending pipeline early.")
                break

    # ── Finalize ──────────────────────────────────────────────────────────
    writer.release()
    csv_file.close()
    if show_viewer:
        close_all_windows()

    t_elapsed = time.perf_counter() - t_start

    # ── Summary Report ────────────────────────────────────────────────────
    report_lines = [
        "FootVision AI — Phase 8 Kit Classification Report",
        "=" * 65,
        f"Sequence               : {seq_dir}",
        f"Total Frames Processed : {total_frames} ({total_frames / fps_src:.1f} seconds)",
        "",
        "KIT SEPARATION BREAKDOWN",
        f"  Team A (White / Navy)        : {cumulative_counts['Team A']} instances ({cumulative_counts['Team A']/max(1, total_detections)*100:.1f}%)",
        f"  Team B (Green / White)       : {cumulative_counts['Team B']} instances ({cumulative_counts['Team B']/max(1, total_detections)*100:.1f}%)",
        f"  Referees (Yellow)            : {cumulative_counts['Referee']} instances ({cumulative_counts['Referee']/max(1, total_detections)*100:.1f}%)",
        f"  Staff / Goalkeepers (Black)  : {cumulative_counts['Staff/GK']} instances ({cumulative_counts['Staff/GK']/max(1, total_detections)*100:.1f}%)",
        "",
        "TIMING & SPEED",
        f"  Total Processing Time        : {t_elapsed:.2f} seconds",
        f"  Pipeline Speed               : {total_frames / t_elapsed:.2f} FPS",
        "",
        "OUTPUT DELIVERABLES",
        f"  Annotated Video : {video_path}",
        f"  Dataset (CSV)   : {csv_path}",
    ]

    print(f"\n{'=' * 75}")
    for line in report_lines:
        print(f"  {line}")
    print(f"{'=' * 75}\n")

    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"  Summary report saved to: {report_path}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 8: Accurate Kit & Role Classification (White vs Green vs Ref vs Staff/GK)."
    )
    parser.add_argument("--seq_dir",    default="data/raw/SNMOT-062",
                        help="Path to MOT sequence directory")
    parser.add_argument("--threshold",  type=float, default=0.20,
                        help="Detector confidence threshold (default: 0.20)")
    parser.add_argument("--output_dir", default="outputs",
                        help="Output directory (default: outputs/)")
    parser.add_argument("--no_viewer",  action="store_true",
                        help="Disable live popup window for faster headless processing")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="Process only first N frames (0 = all)")
    args = parser.parse_args()

    try:
        run_team_classification_pipeline(
            seq_dir     = args.seq_dir,
            threshold   = args.threshold,
            output_dir  = args.output_dir,
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
