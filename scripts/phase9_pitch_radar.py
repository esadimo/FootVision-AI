"""
FootVision AI -- Phase 9: Pitch Calibration & 2D Tactical Radar Mapping

WORKFLOW:
    1. Run phase9_pick_landmarks.py FIRST to calibrate homography (one-time step).
    2. Run this script to process the full sequence and produce:
       - outputs/SNMOT-062_phase9_tactical_radar.mp4  (side-by-side composite video)
       - outputs/SNMOT-062_phase9_pitch_coords.csv    (per-frame pitch meter coordinates)

Usage:
    python scripts/phase9_pitch_radar.py [--seq_dir data/raw/SNMOT-062] [--no_viewer]
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

from src.teams.crop_extractor  import extract_torso_crop
from src.teams.colour_features import extract_chest_color_metrics
from src.teams.classifier      import MatchKitClassifier
from src.calibration.homography import load_homography, project_point
from src.visualization.pitch_plots import draw_pitch_radar, build_composite_frame
from src.visualization.overlays import show_frame, close_all_windows


# ─── Default Constants ────────────────────────────────────────────────────────
MODEL_PATH        = "yolov8n.pt"
CONF_THRESHOLD    = 0.20
IOU_THRESHOLD     = 0.45
FPS               = 25.0
TARGET_CLASS      = 0       # 'person' in COCO
COMPOSITE_HEIGHT  = 540     # pixel height for both panels of the composite frame

TEAM_A_COLOR  = (200, 130, 130)   # BGR for broadcast overlay (light navy-white)
TEAM_B_COLOR  = (80,  200,  80)   # BGR for broadcast overlay (green)
REF_COLOR     = (0,   215, 255)   # Yellow
STAFF_COLOR   = (128, 128, 128)   # Gray

LABEL_COLORS = {
    "Team A":   TEAM_A_COLOR,
    "Team B":   TEAM_B_COLOR,
    "Referee":  REF_COLOR,
    "Staff/GK": STAFF_COLOR,
    "Unknown":  (200, 200, 200),
}


# ─── Annotation Helper ────────────────────────────────────────────────────────

def annotate_broadcast_frame(frame: np.ndarray,
                             tracks: list,
                             labels: dict) -> np.ndarray:
    """
    Draws bounding boxes, track IDs, and team labels on the broadcast frame.

    Parameters
    ----------
    frame : np.ndarray
        Raw video frame.
    tracks : list
        List of track boxes from YOLO results.
    labels : dict
        {track_id: team_label} mapping.

    Returns
    -------
    np.ndarray
        Annotated copy of the frame.
    """
    out = frame.copy()
    for box in tracks:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        tid   = int(box.id[0]) if box.id is not None else -1
        label = labels.get(tid, "Unknown")
        color = LABEL_COLORS.get(label, (200, 200, 200))

        # Bounding box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Label background
        tag = f"#{tid} {label}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        y_tag = max(y1 - 4, th + 4)
        cv2.rectangle(out, (x1, y_tag - th - 4), (x1 + tw + 4, y_tag + 2), color, cv2.FILLED)
        cv2.putText(out, tag, (x1 + 2, y_tag - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    return out


# ─── HUD for Broadcast Panel ─────────────────────────────────────────────────

def draw_broadcast_hud(frame: np.ndarray,
                       frame_num: int,
                       timestamp: float,
                       label_counts: dict) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w, 42), (10, 10, 10), cv2.FILLED)
    info = (f"Frame {frame_num}  |  {timestamp:.2f}s  |  "
            f"A:{label_counts.get('Team A', 0)}  "
            f"B:{label_counts.get('Team B', 0)}  "
            f"Ref:{label_counts.get('Referee', 0)}  "
            f"GK/Staff:{label_counts.get('Staff/GK', 0)}")
    cv2.putText(out, info, (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    return out


from src.calibration.homography import load_homography, project_point, get_homography_for_frame


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 9: Tactical Radar Mapping")
    parser.add_argument("--seq_dir",     default="data/raw/SNMOT-062")
    parser.add_argument("--model",       default=MODEL_PATH)
    parser.add_argument("--conf",        type=float, default=CONF_THRESHOLD)
    parser.add_argument("--iou",         type=float, default=IOU_THRESHOLD)
    parser.add_argument("--output_dir",  default="outputs")
    parser.add_argument("--homography",  default=None,
                        help="Path to homography file (.json for multi-keyframe or .npy)")
    parser.add_argument("--no_viewer",   action="store_true",
                        help="Suppress live preview window")
    parser.add_argument("--skip_video",  action="store_true",
                        help="Skip writing output video (CSV only)")
    args = parser.parse_args()

    # ── Auto-Detect Homography File ───────────────────────────────────────────
    h_path = args.homography
    if h_path is None:
        json_candidate = os.path.join(args.output_dir, "homography_keyframes.json")
        npy_candidate  = os.path.join(args.output_dir, "homography.npy")
        if os.path.exists(json_candidate):
            h_path = json_candidate
        elif os.path.exists(npy_candidate):
            h_path = npy_candidate
        else:
            print(f"\n[ERROR] No homography calibration found in {args.output_dir}")
            print("  Please run 'python scripts/phase9_pick_landmarks.py' first to calibrate.\n")
            sys.exit(1)

    calib_data = load_homography(h_path)

    # ── Load Image Sequence ───────────────────────────────────────────────────
    img_dir = os.path.join(args.seq_dir, "img1")
    frames  = sorted([f for f in os.listdir(img_dir)
                      if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    if not frames:
        print(f"[ERROR] No images found in {img_dir}")
        sys.exit(1)

    seq_name   = os.path.basename(args.seq_dir)
    total_frames = len(frames)
    print(f"\n  Phase 9 -- Tactical Radar Mapping")
    print(f"  Sequence : {seq_name}  ({total_frames} frames)")
    print(f"  Model    : {args.model}  | Conf: {args.conf}")
    print(f"  Calib    : {h_path}")
    print()

    # ── Load Model & Classifier ───────────────────────────────────────────────
    model      = YOLO(args.model)
    classifier = MatchKitClassifier()

    # ── Peek at first frame to get resolution ────────────────────────────────
    first = cv2.imread(os.path.join(img_dir, frames[0]))
    fh, fw = first.shape[:2]

    # ── Video Writer Setup ────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    vid_path = os.path.join(args.output_dir, f"{seq_name}_phase9_tactical_radar.mp4")
    csv_path = os.path.join(args.output_dir, f"{seq_name}_phase9_pitch_coords.csv")

    # Determine composite frame size from a dummy run
    dummy_radar   = draw_pitch_radar([], frame_number=0)
    dummy_comp    = build_composite_frame(first, dummy_radar, target_height=COMPOSITE_HEIGHT)
    comp_h, comp_w = dummy_comp.shape[:2]

    writer = None
    if not args.skip_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(vid_path, fourcc, FPS, (comp_w, comp_h))
        print(f"  Output video : {vid_path}")
    print(f"  Output CSV   : {csv_path}")
    print()

    # ── CSV Writer ────────────────────────────────────────────────────────────
    csv_file   = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame_number", "timestamp_s", "track_id", "team_label",
                         "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
                         "foot_x_px", "foot_y_px",
                         "pitch_x_m", "pitch_y_m"])

    # ── Tracking & Smoothing States ───────────────────────────────────────────
    smoothed_positions = {}  # {track_id: (smoothed_x, smoothed_y)}
    player_trails      = {}  # {track_id: [(x_m, y_m), ...]}
    SMOOTHING_ALPHA    = 0.60
    MAX_TRAIL_LEN      = 10

    # ── Main Loop ─────────────────────────────────────────────────────────────
    t0 = time.time()
    for fi, fname in enumerate(frames):
        frame_path = os.path.join(img_dir, fname)
        frame      = cv2.imread(frame_path)
        if frame is None:
            continue

        frame_num = fi + 1
        timestamp = fi / FPS

        # Dynamic homography matrix for current camera position
        H_current = get_homography_for_frame(calib_data, fi, fw, fh)

        # Run YOLO + ByteTrack
        results = model.track(
            frame,
            persist=True,
            conf=args.conf,
            iou=args.iou,
            classes=[TARGET_CLASS],
            tracker="bytetrack.yaml",
            verbose=False
        )

        result  = results[0]
        tracks  = result.boxes if result.boxes is not None else []

        # ── Per-Track: Classify + Project ────────────────────────────────────
        current_labels = {}
        player_positions = []
        label_counts = {"Team A": 0, "Team B": 0, "Referee": 0, "Staff/GK": 0}

        for box in tracks:
            if box.id is None:
                continue
            tid   = int(box.id[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Kit classification (sliding window majority vote)
            crop = extract_torso_crop(frame, [x1, y1, x2, y2])
            if crop is not None and crop.size > 0:
                metrics = extract_chest_color_metrics(crop)
                label   = classifier.update_track(
                    tid,
                    classifier.predict_single(metrics),
                    min_votes=5
                )
            else:
                label = "Unknown"

            current_labels[tid] = label
            if label in label_counts:
                label_counts[label] += 1

            # Ground contact foot point refinement (removes ground shadow bias)
            h_box = y2 - y1
            foot_x_px = (x1 + x2) / 2.0
            foot_y_px = y2 - 0.02 * h_box

            # Project using current frame's dynamic homography
            raw_x_m, raw_y_m = project_point(H_current, float(foot_x_px), float(foot_y_px))

            # Temporal EMA trajectory smoothing (eliminates stride/bounding box jitter)
            if tid in smoothed_positions:
                prev_x, prev_y = smoothed_positions[tid]
                smooth_x = SMOOTHING_ALPHA * raw_x_m + (1.0 - SMOOTHING_ALPHA) * prev_x
                smooth_y = SMOOTHING_ALPHA * raw_y_m + (1.0 - SMOOTHING_ALPHA) * prev_y
            else:
                smooth_x = raw_x_m
                smooth_y = raw_y_m

            smoothed_positions[tid] = (smooth_x, smooth_y)

            # Update historical movement trails
            if tid not in player_trails:
                player_trails[tid] = []
            player_trails[tid].append((smooth_x, smooth_y))
            if len(player_trails[tid]) > MAX_TRAIL_LEN:
                player_trails[tid].pop(0)

            # Write to CSV
            csv_writer.writerow([frame_num, f"{timestamp:.4f}", tid, label,
                                  x1, y1, x2, y2,
                                  f"{foot_x_px:.1f}", f"{foot_y_px:.1f}",
                                  f"{smooth_x:.4f}", f"{smooth_y:.4f}"])

            player_positions.append({
                "track_id":   tid,
                "team_label": label,
                "pitch_x":    smooth_x,
                "pitch_y":    smooth_y,
            })

        # ── Build Composite Frame ─────────────────────────────────────────────
        broadcast_ann  = annotate_broadcast_frame(frame, tracks, current_labels)
        broadcast_hud  = draw_broadcast_hud(broadcast_ann, frame_num, timestamp, label_counts)
        radar          = draw_pitch_radar(player_positions,
                                          frame_number=frame_num,
                                          timestamp=timestamp,
                                          player_trails=player_trails)
        composite      = build_composite_frame(broadcast_hud, radar, target_height=COMPOSITE_HEIGHT)

        if writer is not None:
            writer.write(composite)

        if not args.no_viewer:
            show_frame(composite, "Phase 9 -- Tactical Radar")

        # Progress log
        if frame_num % 50 == 0 or frame_num == total_frames:
            elapsed = time.time() - t0
            fps_now = frame_num / elapsed if elapsed > 0 else 0
            eta     = (total_frames - frame_num) / fps_now if fps_now > 0 else 0
            print(f"  Frame {frame_num:4d}/{total_frames}  |  "
                  f"{fps_now:.2f} FPS  |  ETA: {eta:.0f}s  |  "
                  f"A:{label_counts['Team A']}  B:{label_counts['Team B']}  "
                  f"Ref:{label_counts['Referee']}")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    csv_file.close()
    if writer is not None:
        writer.release()
    close_all_windows()

    elapsed = time.time() - t0
    fps_avg = total_frames / elapsed if elapsed > 0 else 0

    print(f"\n  ===================================================")
    print(f"  Phase 9 Tactical Radar -- Complete")
    print(f"  ===================================================")
    print(f"  Frames processed : {total_frames}")
    print(f"  Total elapsed    : {elapsed:.1f}s  ({fps_avg:.2f} FPS avg)")
    if not args.skip_video:
        print(f"  Composite video  : {vid_path}")
    print(f"  Pitch coords CSV : {csv_path}")
    print()


if __name__ == "__main__":
    main()
