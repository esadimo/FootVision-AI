"""
FootVision AI — Phase 2, Tasks 2.2 / 2.3 / 2.4
Frame Extractor and Array Inspector

Demonstrates:
  - Reading frames sequentially (Task 2.2)
  - Saving selected frames: first, frame 100, every second, last (Task 2.3)
  - Printing frame array properties: shape, dtype, min, max (Task 2.4)

Input : Path to any video file
Output: Selected frames saved to data/frames/  and  array info printed to stdout

Usage:
    python scripts/phase2_frame_extractor.py <path_to_video> [--output_dir data/frames]

Example:
    python scripts/phase2_frame_extractor.py data/raw/clip.mp4
"""

import sys
import argparse
import os
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Frame saving helpers
# ---------------------------------------------------------------------------

def save_frame(frame: np.ndarray, output_dir: str, filename: str) -> str:
    """Save a single frame image to disk and return the full path."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    cv2.imwrite(path, frame)
    return path


def inspect_array(frame: np.ndarray, label: str) -> None:
    """
    Print the key NumPy properties of a frame array.

    A typical OpenCV frame has shape (height, width, channels).
    Channel order is BGR (Blue, Green, Red), not RGB.
    dtype is uint8, meaning pixel values range from 0 to 255.
    """
    print(f"\n  [{label}] Array properties:")
    print(f"    frame.shape  = {frame.shape}   # (height, width, channels)")
    print(f"    frame.dtype  = {frame.dtype}   # uint8 → values 0–255")
    print(f"    frame.min()  = {frame.min()}")
    print(f"    frame.max()  = {frame.max()}")


# ---------------------------------------------------------------------------
# Main extraction logic
# ---------------------------------------------------------------------------

def extract_frames(video_path: str, output_dir: str) -> None:
    """
    Read a video and save selected frames while printing array information.

    Saved frames
    ------------
    frame_0000_first.jpg    — the very first frame
    frame_0100.jpg          — frame index 100 (if it exists)
    frame_every_sec_*.jpg   — one frame per second
    frame_XXXX_last.jpg     — the final frame
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step         = max(1, int(round(fps)))   # number of frames between "every second" saves

    print(f"\n  Video : {video_path}")
    print(f"  FPS   : {fps:.2f}  |  Total frames: {total_frames}")
    print(f"  Saving frames to: {output_dir}")
    print()

    saved = []
    frame_idx = 0

    # ----- Task 2.2 — sequential read loop -----
    while True:
        ret, frame = cap.read()
        if not ret:
            break   # end of video

        print(f"  Reading frame {frame_idx:5d} / {total_frames - 1}", end="\r")

        # ----- Task 2.3 — save selected frames -----

        # First frame
        if frame_idx == 0:
            path = save_frame(frame, output_dir, "frame_0000_first.jpg")
            saved.append(path)
            inspect_array(frame, f"First frame (index 0)")

        # Frame 100
        if frame_idx == 100:
            path = save_frame(frame, output_dir, "frame_0100.jpg")
            saved.append(path)

        # One frame every second
        if frame_idx % step == 0:
            fname = f"frame_every_sec_{frame_idx:05d}.jpg"
            path = save_frame(frame, output_dir, fname)
            saved.append(path)

        frame_idx += 1

    # ----- Save the final frame -----
    # Re-seek to the last frame
    last_idx = total_frames - 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, last_idx)
    ret, last_frame = cap.read()
    if ret:
        fname = f"frame_{last_idx:05d}_last.jpg"
        path = save_frame(last_frame, output_dir, fname)
        saved.append(path)
        inspect_array(last_frame, f"Last frame (index {last_idx})")

    cap.release()

    print(f"\n\n  Done. {len(saved)} frames saved to: {output_dir}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract and save selected frames from a video file.\n"
            "Phase 2, Tasks 2.2 / 2.3 / 2.4."
        )
    )
    parser.add_argument("video", help="Path to the input video file")
    parser.add_argument(
        "--output_dir",
        default="data/frames",
        help="Directory where extracted frames will be saved (default: data/frames)",
    )
    args = parser.parse_args()

    try:
        extract_frames(args.video, args.output_dir)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
