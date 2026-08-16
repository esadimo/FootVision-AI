"""
FootVision AI — Phase 5
Detection Evaluation Against Ground Truth

Objective:
  1. Load ground-truth bounding boxes from the MOT-format gt.txt file.
  2. Sample a set of representative frames spread across the sequence.
  3. Run the YOLOv8 person detector (threshold 0.20) on each frame.
  4. Match detections to GT boxes using Intersection-over-Union (IoU >= 0.5).
  5. Compute per-frame and overall: TP, FP, FN, Precision, Recall.
  6. Display each frame in the live viewer (GT in green, detections in cyan).
  7. Save a per-frame CSV and a text summary report to outputs/.

Ground-Truth File Format (MOT Challenge):
  frame_id, track_id, x(left), y(top), width, height, conf, class_id, visibility, _

Matching Logic:
  A detection is a True Positive (TP) if IoU >= IOU_THRESHOLD with any GT box.
  Unmatched detections are False Positives (FP).
  Unmatched GT boxes are False Negatives (FN).

Usage:
    python scripts/phase5_detection_evaluation.py [options]

    --seq_dir       Path to MOT sequence root (default: data/raw/SNMOT-062)
    --threshold     Confidence threshold to evaluate (default: 0.20)
    --n_frames      Number of frames to evaluate (default: 30)
    --iou_thresh    IoU threshold for TP matching (default: 0.50)
    --no_viewer     Skip interactive viewer (useful for batch runs)
"""

import os
import sys
import argparse
import csv

# Add project root directory to sys.path to allow importing src module directly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
import numpy as np
from typing import List, Dict, Tuple


# ─── Constants ────────────────────────────────────────────────────────────────

WINDOW_NAME = "FootVision AI — Phase 5: Detection Evaluation"


# ─── Ground-Truth Loader ──────────────────────────────────────────────────────

def load_ground_truth(gt_path: str) -> Dict[int, List[List[int]]]:
    """
    Parse the MOT-format gt.txt file into a dictionary keyed by frame index.

    MOT format: frame_id, track_id, x, y, w, h, conf, class_id, visibility, _
    Coordinates [x, y, w, h] are converted to [x1, y1, x2, y2].

    Parameters
    ----------
    gt_path : str
        Absolute or relative path to gt.txt

    Returns
    -------
    dict[int, list[list[int]]]
        { frame_id: [[x1, y1, x2, y2], ...], ... }
    """
    gt: Dict[int, List[List[int]]] = {}
    with open(gt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            frame_id = int(parts[0])
            x  = float(parts[2])
            y  = float(parts[3])
            w  = float(parts[4])
            h  = float(parts[5])
            # Convert from [x, y, w, h] to [x1, y1, x2, y2]
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            gt.setdefault(frame_id, []).append([x1, y1, x2, y2])
    return gt


# ─── IoU Calculation ──────────────────────────────────────────────────────────

def compute_iou(boxA: List[int], boxB: List[int]) -> float:
    """
    Compute the Intersection over Union (IoU) between two bounding boxes.

    Both boxes are in [x1, y1, x2, y2] format.

    IoU = intersection_area / union_area
    """
    # Intersection rectangle
    ix1 = max(boxA[0], boxB[0])
    iy1 = max(boxA[1], boxB[1])
    ix2 = min(boxA[2], boxB[2])
    iy2 = min(boxA[3], boxB[3])

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    intersection = inter_w * inter_h

    if intersection == 0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - intersection

    return intersection / union if union > 0 else 0.0


# ─── Greedy IoU Matching ──────────────────────────────────────────────────────

def match_detections_to_gt(
    det_boxes: List[List[int]],
    gt_boxes:  List[List[int]],
    iou_thresh: float = 0.50
) -> Tuple[int, int, int]:
    """
    Greedily match detector output to GT boxes using IoU.

    Each GT box can be matched at most once (greedy: highest IoU first).

    Returns
    -------
    tp : int  — True Positives  (detected boxes that match a GT box)
    fp : int  — False Positives (detected boxes with no GT match)
    fn : int  — False Negatives (GT boxes that were not detected)
    """
    if not gt_boxes:
        return 0, len(det_boxes), 0
    if not det_boxes:
        return 0, 0, len(gt_boxes)

    matched_gt = set()
    tp = 0

    # Build IoU matrix [n_det x n_gt]
    iou_matrix = np.zeros((len(det_boxes), len(gt_boxes)), dtype=np.float32)
    for i, det in enumerate(det_boxes):
        for j, gt in enumerate(gt_boxes):
            iou_matrix[i, j] = compute_iou(det, gt)

    # Greedily assign by highest IoU
    while True:
        max_iou = iou_matrix.max()
        if max_iou < iou_thresh:
            break
        i, j = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
        tp += 1
        matched_gt.add(j)
        iou_matrix[i, :] = -1  # Remove this detection from further matching
        iou_matrix[:, j] = -1  # Remove this GT from further matching

    fp = len(det_boxes) - tp
    fn = len(gt_boxes)  - tp
    return tp, fp, fn


# ─── Drawing Helpers ──────────────────────────────────────────────────────────

def draw_gt_boxes(frame: np.ndarray, gt_boxes: List[List[int]]) -> None:
    """Draw ground-truth boxes in solid green."""
    for (x1, y1, x2, y2) in gt_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
    # Legend
    cv2.putText(frame, "GT", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2, cv2.LINE_AA)


def draw_det_boxes(frame: np.ndarray,
                   det_boxes: List[List[int]],
                   confs: List[float]) -> None:
    """Draw detector predictions in cyan with confidence label."""
    for (x1, y1, x2, y2), conf in zip(det_boxes, confs):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 220, 0), 2)
        label = f"{conf:.2f}"
        cv2.putText(frame, label, (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 220, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, "DET", (80, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 220, 0), 2, cv2.LINE_AA)


def draw_metrics_overlay(frame: np.ndarray,
                          frame_id: int,
                          tp: int, fp: int, fn: int,
                          precision: float, recall: float) -> None:
    """Draw per-frame metric summary in the top-right corner."""
    h, w = frame.shape[:2]
    lines = [
        f"Frame {frame_id:04d}",
        f"GT: {tp + fn}   DET: {tp + fp}",
        f"TP:{tp}  FP:{fp}  FN:{fn}",
        f"Prec: {precision:.2f}  Rec: {recall:.2f}",
    ]
    x_start = w - 280
    y_start = 20
    # Dark background strip
    cv2.rectangle(frame, (x_start - 6, 4), (w - 4, y_start + len(lines) * 22 + 6),
                  (20, 20, 20), cv2.FILLED)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x_start, y_start + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)


# ─── Main Evaluation Loop ─────────────────────────────────────────────────────

def evaluate(seq_dir: str, threshold: float, n_frames: int,
             iou_thresh: float, show_viewer: bool) -> None:
    """
    Full Phase 5 evaluation pipeline.
    """
    from ultralytics import YOLO
    from src.visualization.overlays import show_frame, close_all_windows

    # ── Locate sequence assets ────────────────────────────────────────────
    img_dir = os.path.join(seq_dir, "img1")
    gt_path = os.path.join(seq_dir, "gt", "gt.txt")

    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"GT file not found: {gt_path}")

    # ── Read sequence info ────────────────────────────────────────────────
    fps = 25.0
    seqinfo_path = os.path.join(seq_dir, "seqinfo.ini")
    if os.path.exists(seqinfo_path):
        with open(seqinfo_path) as f:
            for line in f:
                if line.startswith("frameRate="):
                    fps = float(line.split("=")[1])
                    break

    valid_ext = (".jpg", ".jpeg", ".png")
    all_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(valid_ext)])
    total_frames = len(all_files)

    # ── Sample n_frames evenly across the sequence ────────────────────────
    indices = np.linspace(0, total_frames - 1, n_frames, dtype=int).tolist()
    # MOT frame IDs are 1-indexed
    sampled_frame_ids = [i + 1 for i in indices]

    print(f"\n{'=' * 70}")
    print(f"  FootVision AI — Phase 5: Detection Evaluation")
    print(f"{'=' * 70}")
    print(f"  Sequence  : {seq_dir}")
    print(f"  Total frames in sequence : {total_frames}")
    print(f"  Frames to evaluate       : {n_frames}")
    print(f"  Confidence threshold     : {threshold:.2f}")
    print(f"  IoU match threshold      : {iou_thresh:.2f}")
    print(f"  GT annotations loaded    : Loading...")

    # ── Load ground truth ─────────────────────────────────────────────────
    gt_all = load_ground_truth(gt_path)
    print(f"  GT annotations loaded    : {sum(len(v) for v in gt_all.values())} boxes "
          f"across {len(gt_all)} frames")

    # ── Load model ────────────────────────────────────────────────────────
    print(f"\n  Loading YOLOv8 model (yolov8n.pt)...")
    model = YOLO("yolov8n.pt")

    # ── Per-frame CSV writer ──────────────────────────────────────────────
    os.makedirs("outputs", exist_ok=True)
    csv_path = f"outputs/phase5_evaluation_thresh{int(threshold*100):02d}.csv"
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["frame_id", "n_gt", "n_detected", "tp", "fp", "fn",
                     "precision", "recall"])

    # ── Aggregate counters ────────────────────────────────────────────────
    total_tp = total_fp = total_fn = 0
    failure_frames = []   # frames with recall < 0.7
    success_frames = []   # frames with recall == 1.0

    print(f"\n  {'Frame':>6} | {'GT':>4} | {'Det':>4} | {'TP':>4} | {'FP':>4} | "
          f"{'FN':>4} | {'Prec':>6} | {'Rec':>6}")
    print("  " + "-" * 60)

    for frame_id in sampled_frame_ids:
        img_path = os.path.join(img_dir, all_files[frame_id - 1])
        frame    = cv2.imread(img_path)
        if frame is None:
            continue

        # ── GT boxes for this frame ───────────────────────────────────────
        gt_boxes = gt_all.get(frame_id, [])

        # ── Run detector ──────────────────────────────────────────────────
        results = model(img_path, verbose=False)[0]
        det_boxes = []
        det_confs = []
        for box in results.boxes:
            conf   = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            if model.names[cls_id] == "person" and conf >= threshold:
                coords = box.xyxy[0].cpu().numpy().tolist()
                x1, y1, x2, y2 = [int(v) for v in coords]
                det_boxes.append([x1, y1, x2, y2])
                det_confs.append(conf)

        # ── Match detections to GT ────────────────────────────────────────
        tp, fp, fn = match_detections_to_gt(det_boxes, gt_boxes, iou_thresh)
        precision  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall     = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # ── Aggregate ─────────────────────────────────────────────────────
        total_tp += tp
        total_fp += fp
        total_fn += fn

        if recall < 0.70:
            failure_frames.append((frame_id, recall, len(gt_boxes), len(det_boxes)))
        if recall == 1.0 and precision >= 0.85:
            success_frames.append((frame_id, precision, recall))

        # ── CSV row ───────────────────────────────────────────────────────
        writer.writerow([frame_id, len(gt_boxes), len(det_boxes),
                         tp, fp, fn, f"{precision:.3f}", f"{recall:.3f}"])

        print(f"  {frame_id:>6} | {len(gt_boxes):>4} | {len(det_boxes):>4} | "
              f"{tp:>4} | {fp:>4} | {fn:>4} | {precision:>6.2f} | {recall:>6.2f}")

        # ── Live Viewer ───────────────────────────────────────────────────
        if show_viewer:
            vis = frame.copy()
            draw_gt_boxes(vis, gt_boxes)
            draw_det_boxes(vis, det_boxes, det_confs)
            draw_metrics_overlay(vis, frame_id, tp, fp, fn, precision, recall)
            keep_going = show_frame(WINDOW_NAME, vis, delay_ms=1)
            if not keep_going:
                print("  [Viewer] User quit. Stopping evaluation early.")
                break

    csv_file.close()
    if show_viewer:
        close_all_windows()

    # ── Overall metrics ───────────────────────────────────────────────────
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    print("\n" + "=" * 70)
    print(f"  OVERALL RESULTS  (threshold={threshold:.2f}, IoU>={iou_thresh:.2f})")
    print("=" * 70)
    print(f"  Total TP : {total_tp}")
    print(f"  Total FP : {total_fp}")
    print(f"  Total FN : {total_fn}")
    print(f"  Precision: {overall_precision:.4f}  ({overall_precision*100:.1f}%)")
    print(f"  Recall   : {overall_recall:.4f}  ({overall_recall*100:.1f}%)")
    print()

    # ── Text summary report ───────────────────────────────────────────────
    report_path = f"outputs/phase5_report_thresh{int(threshold*100):02d}.txt"
    with open(report_path, "w") as rpt:
        rpt.write("FootVision AI — Phase 5: Detection Evaluation Report\n")
        rpt.write("=" * 70 + "\n")
        rpt.write(f"Sequence     : {seq_dir}\n")
        rpt.write(f"Threshold    : {threshold:.2f}\n")
        rpt.write(f"IoU Threshold: {iou_thresh:.2f}\n")
        rpt.write(f"Frames eval. : {n_frames}\n\n")
        rpt.write(f"OVERALL METRICS\n")
        rpt.write(f"  TP        : {total_tp}\n")
        rpt.write(f"  FP        : {total_fp}\n")
        rpt.write(f"  FN        : {total_fn}\n")
        rpt.write(f"  Precision : {overall_precision:.4f}  ({overall_precision*100:.1f}%)\n")
        rpt.write(f"  Recall    : {overall_recall:.4f}  ({overall_recall*100:.1f}%)\n\n")
        rpt.write(f"RECOMMENDED THRESHOLD: {threshold:.2f}\n")
        rpt.write(f"  Rationale: Captures small/distant players missed by higher thresholds.\n\n")
        rpt.write(f"SUCCESSFUL FRAMES (recall=1.0, prec>=0.85): {len(success_frames)}\n")
        for (fid, prec, rec) in success_frames[:5]:
            rpt.write(f"  Frame {fid:04d}: Prec={prec:.2f}, Rec={rec:.2f}\n")
        rpt.write(f"\nFAILURE FRAMES (recall < 0.70): {len(failure_frames)}\n")
        for (fid, rec, n_gt, n_det) in failure_frames:
            rpt.write(f"  Frame {fid:04d}: Recall={rec:.2f}, GT={n_gt}, Det={n_det}\n")
        rpt.write(f"\nKNOWN LIMITATIONS\n")
        rpt.write(f"  - Distant or very small players are harder to detect.\n")
        rpt.write(f"  - Motion blur in fast-moving frames reduces box quality.\n")
        rpt.write(f"  - Overlapping/occluded players may only produce one detection.\n")
        rpt.write(f"  - Non-maximum suppression (NMS) may merge close players into one box.\n")
        rpt.write(f"  - General COCO model has no football-specific fine-tuning.\n")

    print(f"  Per-frame CSV saved : {csv_path}")
    print(f"  Summary report saved: {report_path}")
    print("=" * 70 + "\n")


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 5: Evaluate YOLOv8 person detector against MOT ground truth."
    )
    parser.add_argument("--seq_dir",    default="data/raw/SNMOT-062",
                        help="Path to MOT sequence root directory")
    parser.add_argument("--threshold",  type=float, default=0.20,
                        help="YOLO confidence threshold (default: 0.20)")
    parser.add_argument("--n_frames",   type=int,   default=30,
                        help="Number of frames to evaluate (default: 30)")
    parser.add_argument("--iou_thresh", type=float, default=0.50,
                        help="IoU threshold for TP matching (default: 0.50)")
    parser.add_argument("--no_viewer",  action="store_true",
                        help="Skip the live OpenCV viewer (run headless)")
    args = parser.parse_args()

    try:
        evaluate(
            seq_dir=args.seq_dir,
            threshold=args.threshold,
            n_frames=args.n_frames,
            iou_thresh=args.iou_thresh,
            show_viewer=not args.no_viewer,
        )
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
