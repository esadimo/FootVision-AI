"""
FootVision AI — Phase 6
Full-Sequence Detection Pipeline

Objective:
  Wire together all previous phase components into a single end-to-end pipeline:
    - Phase 2: sequential frame reading + video writing
    - Phase 3: bounding-box coordinate math (center, bottom-center)
    - Phase 4: YOLOv8 person detection
    - Phase 5: confidence threshold (0.20) established as optimal

  For every frame in the image sequence:
    1. Read the frame.
    2. Run the detector (threshold 0.20).
    3. Compute all bounding-box properties.
    4. Draw boxes on the frame and write to output video.
    5. Append every detection as a row in the output CSV.
  At the end, print a processing-time report.

Output CSV schema:
    frame_number, timestamp, detection_id, class_name, confidence,
    x1, y1, x2, y2, center_x, center_y, bottom_center_x, bottom_center_y

Usage:
    python scripts/phase6_full_detection_pipeline.py [options]

    --seq_dir      MOT sequence root  (default: data/raw/SNMOT-062)
    --threshold    Confidence cutoff   (default: 0.20)
    --output_dir   Where to save video and CSV (default: outputs/)
    --no_viewer    Skip live OpenCV window
    --max_frames   Process only the first N frames (0 = all)
"""

import os
import sys
import time
import csv
import argparse

# ── Project-root import fix ───────────────────────────────────────────────────
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
import numpy as np
from tqdm import tqdm


# ─── Bounding-box math (Phase 3 logic, now in a shared helper) ───────────────

def box_properties(x1: int, y1: int, x2: int, y2: int) -> dict:
    """
    Compute geometric properties of a bounding box.

    Returns
    -------
    dict with keys:
        center_x, center_y          — centroid of the box
        bottom_center_x, bottom_center_y  — feet-touch point (used for pitch projection)
    """
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bx = cx            # bottom-center x = center x
    by = float(y2)     # bottom-center y = bottom edge
    return dict(center_x=cx, center_y=cy,
                bottom_center_x=bx, bottom_center_y=by)


# ─── Drawing helpers ─────────────────────────────────────────────────────────

def draw_detection(frame: np.ndarray,
                   x1: int, y1: int, x2: int, y2: int,
                   conf: float,
                   det_id: int) -> None:
    """Draw one detection: box, label banner, and bottom-center feet marker."""
    colour = (255, 127, 0)   # cyan-orange

    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

    # Label text with background banner
    label = f"#{det_id} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    cv2.rectangle(frame,
                  (x1, max(y1 - th - 6, 0)),
                  (x1 + tw, y1),
                  colour, cv2.FILLED)
    cv2.putText(frame, label,
                (x1, max(y1 - 3, th)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (255, 255, 255), 1, cv2.LINE_AA)

    # Bottom-center (feet) marker
    bx = int((x1 + x2) / 2)
    cv2.circle(frame, (bx, y2), 4, (0, 255, 0), -1)


def draw_hud(frame: np.ndarray,
             frame_number: int, total_frames: int,
             n_detections: int, fps: float) -> None:
    """Draw a heads-up display strip at the top of the frame."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 36), (20, 20, 20), cv2.FILLED)
    text = (f"Frame {frame_number:04d}/{total_frames}  |  "
            f"Detections: {n_detections:2d}  |  "
            f"Processing: {fps:.1f} fps")
    cv2.putText(frame, text, (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)


# ─── Main pipeline ────────────────────────────────────────────────────────────

def run_pipeline(seq_dir: str,
                 threshold: float,
                 output_dir: str,
                 show_viewer: bool,
                 max_frames: int) -> None:

    from ultralytics import YOLO
    from src.visualization.overlays import show_frame, close_all_windows

    # ── Locate assets ─────────────────────────────────────────────────────
    img_dir      = os.path.join(seq_dir, "img1")
    seqinfo_path = os.path.join(seq_dir, "seqinfo.ini")

    valid_ext = (".jpg", ".jpeg", ".png")
    all_files = sorted([f for f in os.listdir(img_dir)
                        if f.lower().endswith(valid_ext)])

    # Read FPS from seqinfo.ini
    fps_src = 25.0
    if os.path.exists(seqinfo_path):
        with open(seqinfo_path) as f:
            for line in f:
                if line.startswith("frameRate="):
                    fps_src = float(line.split("=")[1])
                    break

    # Determine frame count to process
    total_frames = len(all_files)
    if max_frames > 0:
        all_files = all_files[:max_frames]
        total_frames = len(all_files)

    # Read frame dimensions from the first image
    first_frame = cv2.imread(os.path.join(img_dir, all_files[0]))
    frame_h, frame_w = first_frame.shape[:2]

    # ── Output paths ──────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    seq_name    = os.path.basename(seq_dir.rstrip("/\\"))
    video_path  = os.path.join(output_dir, f"{seq_name}_phase6_detections.mp4")
    csv_path    = os.path.join(output_dir, f"{seq_name}_phase6_detections.csv")
    report_path = os.path.join(output_dir, f"{seq_name}_phase6_report.txt")

    # ── Video writer ──────────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps_src, (frame_w, frame_h))

    # ── CSV writer ────────────────────────────────────────────────────────
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame_number", "timestamp",
        "detection_id", "class_name", "confidence",
        "x1", "y1", "x2", "y2",
        "center_x", "center_y",
        "bottom_center_x", "bottom_center_y"
    ])

    # ── Load model ────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  FootVision AI — Phase 6: Full Detection Pipeline")
    print(f"{'=' * 70}")
    print(f"  Sequence     : {seq_dir}")
    print(f"  Frames       : {total_frames}  ({total_frames / fps_src:.1f} s @ {fps_src} fps)")
    print(f"  Threshold    : {threshold}")
    print(f"  Output video : {video_path}")
    print(f"  Output CSV   : {csv_path}")
    print(f"\n  Loading YOLOv8 model...")
    model = YOLO("yolov8n.pt")
    print(f"  Model ready. Starting pipeline...\n")

    # ── Timing accumulators ───────────────────────────────────────────────
    t_detect_total = 0.0
    t_draw_total   = 0.0
    total_dets     = 0
    t_pipeline_start = time.perf_counter()

    # ── Main loop ─────────────────────────────────────────────────────────
    processing_fps = 0.0
    for frame_idx, fname in enumerate(tqdm(all_files, desc="  Processing", unit="frame")):

        frame_number = frame_idx + 1                          # 1-indexed (matches GT)
        timestamp    = frame_idx / fps_src                    # seconds

        # Read frame
        frame = cv2.imread(os.path.join(img_dir, fname))
        if frame is None:
            continue
        vis = frame.copy()                                     # separate canvas for drawing

        # ── Detection ─────────────────────────────────────────────────────
        t0 = time.perf_counter()
        results = model(frame, verbose=False)[0]
        t_detect_total += time.perf_counter() - t0

        # ── Parse + filter detections ─────────────────────────────────────
        t0 = time.perf_counter()
        det_id = 0
        frame_dets = 0
        for box in results.boxes:
            conf   = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())

            if model.names[cls_id] != "person" or conf < threshold:
                continue

            det_id    += 1
            frame_dets += 1

            coords = box.xyxy[0].cpu().numpy().tolist()
            x1, y1, x2, y2 = [int(v) for v in coords]

            # Phase 3 math
            props = box_properties(x1, y1, x2, y2)

            # Draw on canvas
            draw_detection(vis, x1, y1, x2, y2, conf, det_id)

            # Write CSV row
            csv_writer.writerow([
                frame_number,
                f"{timestamp:.3f}",
                det_id,
                "person",
                f"{conf:.4f}",
                x1, y1, x2, y2,
                f"{props['center_x']:.1f}",
                f"{props['center_y']:.1f}",
                f"{props['bottom_center_x']:.1f}",
                f"{props['bottom_center_y']:.1f}",
            ])

        total_dets += frame_dets

        # ── HUD overlay ───────────────────────────────────────────────────
        elapsed = time.perf_counter() - t_pipeline_start
        processing_fps = frame_number / elapsed if elapsed > 0 else 0.0
        draw_hud(vis, frame_number, total_frames, frame_dets, processing_fps)
        t_draw_total += time.perf_counter() - t0

        # ── Write to video ────────────────────────────────────────────────
        writer.write(vis)

        # ── Live viewer ───────────────────────────────────────────────────
        if show_viewer:
            keep = show_frame("FootVision AI — Phase 6", vis, delay_ms=1)
            if not keep:
                print("\n  [Viewer] Quit — stopping pipeline early.")
                break

    # ── Cleanup ───────────────────────────────────────────────────────────
    writer.release()
    csv_file.close()
    if show_viewer:
        close_all_windows()

    t_total = time.perf_counter() - t_pipeline_start

    # ── Processing report ─────────────────────────────────────────────────
    avg_fps         = total_frames / t_total        if t_total   > 0 else 0
    avg_det_ms      = t_detect_total / total_frames * 1000
    avg_draw_ms     = t_draw_total   / total_frames * 1000
    avg_dets_frame  = total_dets / total_frames     if total_frames > 0 else 0

    report_lines = [
        "FootVision AI — Phase 6 Processing Report",
        "=" * 60,
        f"Sequence     : {seq_dir}",
        f"Frames proc. : {total_frames}",
        f"Threshold    : {threshold}",
        "",
        "TIMING",
        f"  Total wall time      : {t_total:.2f} s",
        f"  Average pipeline FPS : {avg_fps:.2f}",
        f"  Avg detection time   : {avg_det_ms:.2f} ms/frame",
        f"  Avg drawing time     : {avg_draw_ms:.2f} ms/frame",
        "",
        "DETECTIONS",
        f"  Total detections     : {total_dets}",
        f"  Avg per frame        : {avg_dets_frame:.2f}",
        "",
        "OUTPUT",
        f"  Annotated video      : {video_path}",
        f"  Detection CSV        : {csv_path}",
    ]

    print(f"\n{'=' * 70}")
    for line in report_lines:
        print(f"  {line}")
    print(f"{'=' * 70}\n")

    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"  Report saved: {report_path}\n")


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 6: End-to-end detection pipeline — video in, annotated video + CSV out."
    )
    parser.add_argument("--seq_dir",    default="data/raw/SNMOT-062")
    parser.add_argument("--threshold",  type=float, default=0.20)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--no_viewer",  action="store_true")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="Process only first N frames (0 = all 750)")
    args = parser.parse_args()

    try:
        run_pipeline(
            seq_dir    = args.seq_dir,
            threshold  = args.threshold,
            output_dir = args.output_dir,
            show_viewer= not args.no_viewer,
            max_frames = args.max_frames,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
