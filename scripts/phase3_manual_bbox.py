"""
FootVision AI — Phase 3
Manual Bounding Box Exercise

Objective:
  1. Load a reference frame (e.g. data/frames/frame_0000_first.jpg).
  2. Define 5 player bounding boxes using manual [x1, y1, x2, y2] coordinates.
  3. Calculate width, height, center (xc, yc), and bottom-center (xb, yb) for each box.
  4. Draw these boxes, center indicators, and bottom-center feet markers.
  5. Print the calculations and export the annotated visual frame.

Mathematical equations:
  width = x2 - x1
  height = y2 - y1
  center_x (xc) = (x1 + x2) / 2
  center_y (yc) = (y1 + y2) / 2
  bottom_center_x (xb) = (x1 + x2) / 2
  bottom_center_y (yb) = y2

Usage:
    python scripts/phase3_manual_bbox.py [--image data/frames/frame_0000_first.jpg]
"""

import os
import sys
import argparse
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Core math conversions
# ---------------------------------------------------------------------------

def calculate_box_properties(x1: int, y1: int, x2: int, y2: int) -> dict:
    """
    Calculate bounding box dimensions, center, and bottom-center anchor points.
    """
    width = x2 - x1
    height = y2 - y1
    
    # Coordinates are calculated as floats to preserve precision, 
    # but drawn as rounded integers.
    xc = (x1 + x2) / 2.0
    yc = (y1 + y2) / 2.0
    
    xb = (x1 + x2) / 2.0
    yb = float(y2)
    
    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "width": width,
        "height": height,
        "center": (xc, yc),
        "bottom_center": (xb, yb)
    }


# ---------------------------------------------------------------------------
# Bounding boxes selection (Default manual definitions based on SNMOT-062 Frame 0)
# ---------------------------------------------------------------------------

# Note: These bounding boxes are selected mock values targeting visible players.
# We will inspect the actual frame and let the user modify them or use these default labels.
DEFAULT_PLAYERS = [
    {"id": 1, "bbox": [1085, 396, 1121, 490]},  # Player 1
    {"id": 2, "bbox": [890, 420, 922, 510]},   # Player 2
    {"id": 3, "bbox": [655, 435, 686, 528]},   # Player 3
    {"id": 4, "bbox": [1330, 480, 1370, 580]},  # Player 4
    {"id": 5, "bbox": [410, 525, 448, 620]}    # Player 5
]


# ---------------------------------------------------------------------------
# Drawing and Visual Export
# ---------------------------------------------------------------------------

def process_manual_boxes(image_path: str, output_path: str) -> None:
    """
    Annotates manual bounding boxes on a reference image frame and exports results.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Selected reference frame does not exist: {image_path}. "
                                "Make sure you completed Phase 2 first!")

    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Could not read frame: {image_path}")

    print(f"\n  Processing frame: {image_path}")
    print("  " + "-" * 80)
    print("  ID | Bounding Box [x1, y1, x2, y2] | Width | Height | Center (x, y)  | Feet Point (x, y)")
    print("  " + "-" * 80)

    for p in DEFAULT_PLAYERS:
        pid = p["id"]
        x1, y1, x2, y2 = p["bbox"]
        
        # Calculate mathematical metrics
        props = calculate_box_properties(x1, y1, x2, y2)
        xc, yc = props["center"]
        xb, yb = props["bottom_center"]
        
        print(f"  {pid:2d} | [{x1:4d}, {y1:4d}, {x2:4d}, {y2:4d}]       "
              f"| {props['width']:5d} | {props['height']:6d} | ({xc:6.1f}, {yc:6.1f}) | ({xb:6.1f}, {yb:6.1f})")

        # ── Draw Bounding Box ──────────────────────────────────────────────
        # Yellow bounding box outline
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # ── Draw Center Point ──────────────────────────────────────────────
        # Red circle at the exact center (xc, yc)
        cv2.circle(frame, (int(round(xc)), int(round(yc))), 4, (0, 0, 255), -1)

        # ── Draw Bottom-Center Point (Feet Marker) ─────────────────────────
        # Bright green circle where player touches the pitch (xb, yb)
        cv2.circle(frame, (int(round(xb)), int(round(yb))), 5, (0, 255, 0), -1)

        # ── Draw Label Text ───────────────────────────────────────────────
        label = f"P{pid} ({props['width']}x{props['height']})"
        cv2.putText(
            frame, label,
            org=(x1, y1 - 8),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.45,
            color=(0, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA
        )

    # Export annotated visual copy
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, frame)
    print("  " + "-" * 80)
    print(f"  Annotated frame successfully written to: {output_path}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Run manual bounding-box coordinates exercise (Phase 3)."
    )
    parser.add_argument(
        "--image",
        default="data/frames/frame_0000_first.jpg",
        help="Path to reference frame image"
    )
    parser.add_argument(
        "--output",
        default="outputs/phase3_manual_bbox.jpg",
        help="Path to save annotated output image"
    )
    args = parser.parse_args()

    try:
        process_manual_boxes(args.image, args.output)
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
