"""
FootVision AI — Phase 2, Tasks 2.5 / 2.6
Annotated Video Writer

Demonstrates:
  - Drawing shapes on frames (Task 2.5): rectangle, point, line, text, counter
  - Writing an output video (Task 2.6)

Input : Path to any video file
Output: Annotated video saved to outputs/  with a visible frame counter on every frame

Usage:
    python scripts/phase2_write_annotated_video.py <path_to_video> [--output outputs/annotated.mp4]

Example:
    python scripts/phase2_write_annotated_video.py data/raw/clip.mp4
"""

import sys
import argparse
import os
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Drawing helpers — each function shows one OpenCV drawing concept
# ---------------------------------------------------------------------------

def draw_frame_counter(frame: np.ndarray, frame_idx: int, total_frames: int) -> None:
    """
    Draw a frame counter in the top-left corner.

    cv2.putText arguments:
        img       — the frame to draw on (modified in-place)
        text      — the string to render
        org       — bottom-left corner of the text (x, y)
        fontFace  — font type (FONT_HERSHEY_SIMPLEX is the most readable)
        fontScale — scale factor; 1.0 = roughly 30px tall
        color     — BGR tuple (not RGB!)
        thickness — stroke width in pixels
        lineType  — cv2.LINE_AA gives anti-aliased (smooth) edges
    """
    text = f"Frame: {frame_idx}  /  {total_frames}"
    cv2.putText(
        frame, text,
        org=(20, 45),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1.2,
        color=(255, 255, 255),   # white
        thickness=2,
        lineType=cv2.LINE_AA,
    )


def draw_demo_shapes(frame: np.ndarray) -> None:
    """
    Draw a small set of demonstration shapes in the bottom-right corner.

    These shapes illustrate the main OpenCV drawing primitives.
    They are placed in the bottom-right corner to avoid covering player areas.
    """
    h, w = frame.shape[:2]

    # Anchor point for the demo panel
    px = w - 220
    py = h - 130

    # ── Rectangle ──────────────────────────────────────────────────────────
    # cv2.rectangle(img, pt1_top_left, pt2_bottom_right, color_BGR, thickness)
    # thickness=-1 fills the shape; positive values draw an outline.
    cv2.rectangle(frame, (px, py), (px + 80, py + 40), (0, 200, 255), 2)
    cv2.putText(frame, "rect", (px + 5, py + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)

    # ── Circle (used as a "point") ──────────────────────────────────────────
    # cv2.circle(img, center, radius, color_BGR, thickness)
    cx, cy = px + 140, py + 20
    cv2.circle(frame, (cx, cy), 8, (0, 255, 100), -1)   # filled green dot
    cv2.putText(frame, "pt", (cx - 8, cy + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 100), 1, cv2.LINE_AA)

    # ── Line ────────────────────────────────────────────────────────────────
    # cv2.line(img, pt1, pt2, color_BGR, thickness)
    lx = px
    cv2.line(frame, (lx, py + 60), (lx + 190, py + 60), (255, 80, 80), 2)
    cv2.putText(frame, "line", (lx + 70, py + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 80, 80), 1, cv2.LINE_AA)

    # ── Text label ─────────────────────────────────────────────────────────
    cv2.putText(frame, "FootVision AI", (px, py + 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main writer loop
# ---------------------------------------------------------------------------

def write_annotated_video(video_path: str, output_path: str) -> None:
    """
    Read every frame of the input video or directory, draw annotations, and write to output.

    OpenCV VideoWriter requires:
        filename   — output path (extension determines container, e.g. .mp4)
        fourcc     — codec code created with cv2.VideoWriter_fourcc(*'mp4v')
        fps        — must match the source to avoid playback speed issues
        frameSize  — (width, height) tuple — must exactly match the frames
    """
    import os

    is_dir = os.path.isdir(video_path)
    
    if is_dir:
        valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
        files = sorted([f for f in os.listdir(video_path) if f.lower().endswith(valid_exts)])
        if not files:
            raise FileNotFoundError(f"No valid image frames found in directory: {video_path}")
        total_frames = len(files)
        # Attempt to read frameRate from parent's seqinfo.ini, else default to 25
        fps = 25.0
        parent_dir = os.path.dirname(video_path)
        seqinfo_path = os.path.join(parent_dir, "seqinfo.ini")
        if os.path.exists(seqinfo_path):
            try:
                with open(seqinfo_path, "r") as f:
                    for line in f:
                        if line.startswith("frameRate="):
                            fps = float(line.split("=")[1].strip())
                            break
            except Exception:
                pass
        
        first_frame_path = os.path.join(video_path, files[0])
        first_frame = cv2.imread(first_frame_path)
        if first_frame is None:
            raise ValueError(f"Could not read first frame: {first_frame_path}")
        height, width = first_frame.shape[:2]
        cap = None
    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")
        fps          = cap.get(cv2.CAP_PROP_FPS)
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # VideoWriter setup
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"\n  Input  : {video_path}  ({width}x{height} @ {fps:.2f} fps, {total_frames} frames)")
    print(f"  Output : {output_path}")
    print()

    for frame_idx in range(total_frames):
        if is_dir:
            frame_file_path = os.path.join(video_path, files[frame_idx])
            frame = cv2.imread(frame_file_path)
            ret = frame is not None
        else:
            ret, frame = cap.read()
            
        if not ret:
            break

        # ----- Task 2.5 — draw shapes -----
        draw_demo_shapes(frame)

        # ----- Task 2.6 — draw frame counter -----
        draw_frame_counter(frame, frame_idx, total_frames - 1)

        # Write the annotated frame to the output video
        out.write(frame)

        # Progress display
        if frame_idx % 25 == 0 or frame_idx == total_frames - 1:
            pct = 100 * frame_idx / max(total_frames - 1, 1)
            print(f"  Writing frame {frame_idx:5d} / {total_frames - 1}  ({pct:.1f}%)", end="\r")

    if cap is not None:
        cap.release()
    out.release()   # ← CRITICAL: flush and close the video file

    print(f"\n\n  Done. Annotated video written to: {output_path}")
    print(f"  Total frames written: {total_frames}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Write an annotated output video with a frame counter on every frame.\n"
            "Phase 2, Tasks 2.5 / 2.6."
        )
    )
    parser.add_argument("video", help="Path to the input video file")
    parser.add_argument(
        "--output",
        default=None,
        help="Path for the output video (default: outputs/<input_name>_annotated.mp4)",
    )
    args = parser.parse_args()

    # Default output path based on input filename
    if args.output is None:
        base = os.path.splitext(os.path.basename(args.video))[0]
        output_path = os.path.join("outputs", f"{base}_annotated.mp4")
    else:
        output_path = args.output

    try:
        write_annotated_video(args.video, output_path)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
