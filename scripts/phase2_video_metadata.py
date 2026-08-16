"""
FootVision AI — Phase 2, Task 2.1
Video Metadata Inspector

Input : Path to any video file
Output: Printed metadata — resolution, FPS, frame count, duration, codec

Usage:
    python scripts/phase2_video_metadata.py <path_to_video>

Example:
    python scripts/phase2_video_metadata.py data/raw/clip.mp4
"""

import sys
import argparse
import cv2


def get_video_metadata(video_path: str) -> dict:
    """
    Open a video file or an image sequence directory and return its metadata as a dictionary.

    Parameters
    ----------
    video_path : str
        Absolute or relative path to the video file or directory of image frames.

    Returns
    -------
    dict
        Keys: width, height, fps, frame_count, duration_sec, codec
    """
    import os
    
    if os.path.isdir(video_path):
        # We are dealing with an image sequence directory
        valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
        files = sorted([f for f in os.listdir(video_path) if f.lower().endswith(valid_exts)])
        if not files:
            raise FileNotFoundError(f"No valid image frames found in directory: {video_path}")
            
        first_frame_path = os.path.join(video_path, files[0])
        first_frame = cv2.imread(first_frame_path)
        if first_frame is None:
            raise ValueError(f"Could not read first frame: {first_frame_path}")
            
        height, width = first_frame.shape[:2]
        frame_count = len(files)
        # Attempt to read frame rate from a parent directory's seqinfo.ini, or default to 25
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
                
        duration_sec = frame_count / fps
        return {
            "width":        width,
            "height":       height,
            "fps":          fps,
            "frame_count":  frame_count,
            "duration_sec": duration_sec,
            "codec":        "IMAGE_SEQ",
        }

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    # Read properties using OpenCV constants
    width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps         = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Duration in seconds — calculated from frame count and FPS
    duration_sec = frame_count / fps if fps > 0 else 0.0

    # Codec — stored as a float that encodes four ASCII characters (FourCC)
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "UNKNOWN"
    if fourcc_int != 0:
        codec = (
            chr(fourcc_int & 0xFF)
            + chr((fourcc_int >> 8) & 0xFF)
            + chr((fourcc_int >> 16) & 0xFF)
            + chr((fourcc_int >> 24) & 0xFF)
        )

    cap.release()

    return {
        "width":        width,
        "height":       height,
        "fps":          fps,
        "frame_count":  frame_count,
        "duration_sec": duration_sec,
        "codec":        codec.strip(),
    }


def print_metadata(meta: dict, video_path: str) -> None:
    """Pretty-print video metadata to stdout."""
    print()
    print("=" * 50)
    print("  FootVision AI — Video Metadata")
    print("=" * 50)
    print(f"  File       : {video_path}")
    print(f"  Resolution : {meta['width']} x {meta['height']} px")
    print(f"  FPS        : {meta['fps']:.3f}")
    print(f"  Frames     : {meta['frame_count']}")
    print(f"  Duration   : {meta['duration_sec']:.2f} seconds")
    print(f"  Codec      : {meta['codec']}")
    print("=" * 50)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Inspect metadata of a video file (Phase 2, Task 2.1)."
    )
    parser.add_argument("video", help="Path to the video file")
    args = parser.parse_args()

    try:
        meta = get_video_metadata(args.video)
        print_metadata(meta, args.video)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
