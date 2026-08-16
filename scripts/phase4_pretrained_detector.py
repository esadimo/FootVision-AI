"""
FootVision AI — Phase 4
Pretrained Detector Script

Objective:
  1. Load a pretrained detector model (YOLOv8 nano - yolov8n.pt).
  2. Run inference on a reference frame.
  3. Filter detections to keep only the 'person' class (COCO index 0).
  4. Manually extract bounding boxes, class names, and confidence scores.
  5. Apply confidence thresholds (0.2, 0.4, 0.6, 0.8).
  6. Manually draw the detections and export threshold-specific validation images.

Usage:
    python scripts/phase4_pretrained_detector.py [--image data/frames/frame_0000_first.jpg] [--threshold 0.40]
"""

import os
import sys
import argparse

# Add project root directory to sys.path to allow importing src module directly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
import numpy as np


def run_detector(image_path: str, threshold: float, output_path: str) -> None:
    """
    Loads YOLOv8 nano, runs inference, filters for the 'person' class,
    and draws bounding boxes manually at the specified confidence threshold.
    """
    # Import inside function to allow running script metadata checks without loading torch
    from ultralytics import YOLO
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Selected reference frame does not exist: {image_path}. "
                                "Make sure you completed Phase 2 first!")

    # 1. Load model (automatically downloads weight file to root directory if missing)
    print(f"\n  [YOLOv8] Loading pretrained 'yolov8n.pt' weights...")
    model = YOLO("yolov8n.pt")
    
    # 2. Run inference
    # verbose=False reduces terminal clutter; device='cuda' or 'cpu' will be chosen automatically
    print(f"  [Inference] Running model on: {image_path}")
    results = model(image_path, verbose=False)[0]
    
    # Read the original image array to draw on
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Could not read frame image: {image_path}")
        
    person_count = 0
    raw_detections = []

    # 3. Parse boxes, confidence scores, and class labels
    # YOLOv8 returns coordinates in [x1, y1, x2, y2] format
    boxes = results.boxes
    for box in boxes:
        # Get coordinates as integers
        coords = box.xyxy[0].cpu().numpy().tolist()
        x1, y1, x2, y2 = [int(val) for val in coords]
        
        # Get confidence score and class ID
        conf = float(box.conf[0].cpu().numpy())
        cls_id = int(box.cls[0].cpu().numpy())
        cls_name = model.names[cls_id]
        
        raw_detections.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": conf,
            "class_id": cls_id,
            "class_name": cls_name
        })

    print(f"\n  Raw detection parser metrics (no thresholding):")
    print(f"    Total detected objects (any class): {len(raw_detections)}")
    
    # 4. Filter only 'person' class (index 0 in COCO dataset) and apply confidence threshold
    print(f"\n  Applying filter class='person' and confidence >= {threshold:.2f}:")
    print("  " + "-" * 75)
    print("  Idx | Class  | Confidence | Bounding Box Coordinate [x1, y1, x2, y2]")
    print("  " + "-" * 75)

    idx = 0
    for det in raw_detections:
        # Check class and threshold
        if det["class_name"] == "person" and det["confidence"] >= threshold:
            person_count += 1
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            
            print(f"  {person_count:3d} | person | {conf:10.2f} | [{x1:4d}, {y1:4d}, {x2:4d}, {y2:4d}]")
            
            # 5. Draw bounding box manually
            # Orange/cyan box border color
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 127, 0), 2)
            
            # Label background banner
            label = f"person {conf:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
            )
            cv2.rectangle(
                frame, 
                (x1, y1 - text_h - 6), 
                (x1 + text_w, y1), 
                (255, 127, 0), 
                thickness=cv2.FILLED
            )
            
            # Label text
            cv2.putText(
                frame, label,
                org=(x1, y1 - 4),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.4,
                color=(255, 255, 255),
                thickness=1,
                lineType=cv2.LINE_AA
            )

    print("  " + "-" * 75)
    print(f"  Total players ('person') detected above {threshold:.2f} threshold: {person_count}")
    print("  " + "-" * 75)

    # Save output image
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, frame)
    print(f"  Annotated visual export saved to: {output_path}")
    
    # ── Live Viewer Pop-up ────────────────────────────────────────────────
    # Wait indefinitely (delay_ms=0) since it's a single frame comparison
    # Press space or Q/Esc to close it
    from src.visualization.overlays import show_frame, close_all_windows
    print("  [Viewer] Opening pop-up viewer. Press Spacebar or 'q'/ESC to close and proceed.")
    show_frame(f"FootVision AI - YOLO Detection Threshold {threshold}", frame, delay_ms=0)
    close_all_windows()
    print()


def run_threshold_experiments(image_path: str) -> None:
    """Runs detector across four standard thresholds to compare counts."""
    thresholds = [0.20, 0.40, 0.60, 0.80]
    print("\n" + "=" * 80)
    print("  RUNNING MULTI-THRESHOLD CONFIDENCE STUDY")
    print("=" * 80)
    
    for t in thresholds:
        out_name = f"outputs/phase4_yolo_thresh_{int(t*100):02d}.jpg"
        run_detector(image_path, t, out_name)
    
    print("  Threshold experiments complete. Compare files in /outputs.")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Run object detector on a single frame using COCO weights (Phase 4)."
    )
    parser.add_argument(
        "--image",
        default="data/frames/frame_0000_first.jpg",
        help="Path to frame image to run detector on"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.40,
        help="Model confidence threshold (0.0 to 1.0)"
    )
    parser.add_argument(
        "--experiment",
        action="store_true",
        help="Run multi-threshold experiments (0.2, 0.4, 0.6, 0.8)"
    )
    args = parser.parse_args()

    try:
        if args.experiment:
            run_threshold_experiments(args.image)
        else:
            base_filename = os.path.splitext(os.path.basename(args.image))[0]
            out_path = f"outputs/phase4_detector_{base_filename}_conf_{int(args.threshold*100):02d}.jpg"
            run_detector(args.image, args.threshold, out_path)
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
