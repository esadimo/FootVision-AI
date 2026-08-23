"""
FootVision AI - Auto Homography Tracker (Dynamic Camera Calibration)

This script takes a single homography keyframe (usually frame 0) and uses
Enhanced Correlation Coefficient (ECC) Maximization to automatically track
the camera pans, tilts, and zooms over the entire sequence.

It outputs a dense homography_keyframes.json where every frame has its own
matrix, making Phase 9 100% dynamic and robust to camera movement.
"""

import os
import sys
import argparse
import json
import time
import glob

import cv2
import numpy as np
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.calibration.homography import load_homography

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_dir", type=str, required=True, help="Path to sequence directory")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--downscale", type=float, default=0.25, help="Scale for ECC speedup")
    parser.add_argument("--no_viewer", action="store_true", help="Run headlessly")
    return parser.parse_args()

def main():
    args = parse_args()
    seq_name = os.path.basename(os.path.normpath(args.seq_dir))
    
    # Locate frames
    img_dir = os.path.join(args.seq_dir, "img1")
    frame_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    if not frame_files:
        print(f"Error: No frames found in {img_dir}")
        sys.exit(1)
        
    print(f"Tracking homography for {seq_name} ({len(frame_files)} frames)...")
    
    # Check for initial keyframe
    calib_file = os.path.join(args.output_dir, "homography_keyframes.json")
    if not os.path.exists(calib_file):
        print(f"Error: {calib_file} not found. Please run phase9_pick_landmarks.py to set Frame 0 first!")
        sys.exit(1)
        
    keyframes = load_homography(calib_file)
    if not isinstance(keyframes, list) or len(keyframes) == 0:
        print("Error: Invalid initial keyframes.")
        sys.exit(1)
        
    # Start tracking from the first available keyframe
    start_idx = keyframes[0]["frame_idx"]
    H_base = keyframes[0]["H"]
    
    # Prepare ECC criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-4)
    
    dense_keyframes = []
    
    # Store previous frame for tracking
    prev_frame = cv2.imread(frame_files[start_idx])
    h_orig, w_orig = prev_frame.shape[:2]
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Downscale for performance
    s = args.downscale
    prev_gray_small = cv2.resize(prev_gray, (int(w_orig*s), int(h_orig*s)))
    
    # Scaling matrix for correcting H
    S_mat = np.array([
        [s, 0, 0],
        [0, s, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    S_inv = np.linalg.inv(S_mat)
    
    H_cumulative_small = np.eye(3, dtype=np.float32)
    
    print(f"Starting ECC tracking from frame {start_idx} (Scale={s})")
    
    t0 = time.time()
    
    for i in tqdm(range(start_idx, len(frame_files))):
        if i == start_idx:
            # First frame uses base matrix
            dense_keyframes.append({"frame_idx": i, "H": H_base.tolist()})
            continue
            
        curr_frame = cv2.imread(frame_files[i])
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        curr_gray_small = cv2.resize(curr_gray, (int(w_orig*s), int(h_orig*s)))
        
        # Estimate warp step (prev -> curr)
        warp_step = np.eye(3, dtype=np.float32)
        try:
            cc, warp_step = cv2.findTransformECC(
                prev_gray_small, curr_gray_small, warp_step, 
                cv2.MOTION_HOMOGRAPHY, criteria, None, 1
            )
            # Accumulate warp: H_cum_new = warp_step * H_cum_old
            H_cumulative_small = warp_step @ H_cumulative_small
            prev_gray_small = curr_gray_small
        except cv2.error as e:
            # If ECC fails (e.g., massive motion blur), fallback to previous matrix
            pass
            
        # Map cumulative small tracking matrix back to full resolution
        H_cumulative_full = S_inv @ H_cumulative_small @ S_mat
        
        # The tracked matrix H_cumulative_full maps (Frame 0 pixels -> Frame i pixels).
        # We need (Frame i pixels -> Pitch meters).
        # We know: 
        #   p_pitch = H_base * p_frame0
        #   p_frame_i = H_cumulative_full * p_frame0  =>  p_frame0 = H_cumulative_full_inv * p_frame_i
        # Therefore:
        #   p_pitch = H_base * H_cumulative_full_inv * p_frame_i
        # So, H_i = H_base * H_cumulative_full_inv
        
        try:
            H_cum_inv = np.linalg.inv(H_cumulative_full)
            H_i = H_base @ H_cum_inv
        except np.linalg.LinAlgError:
            H_i = H_base
            
        dense_keyframes.append({"frame_idx": i, "H": H_i.tolist()})
        
        # Optional Visualizer
        if not args.no_viewer:
            preview = cv2.resize(curr_frame, (640, 360))
            cv2.putText(preview, f"Tracking: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow("Auto Homography", preview)
            if cv2.waitKey(1) == 27:
                print("Aborted by user.")
                break

    if not args.no_viewer:
        cv2.destroyAllWindows()
        
    print(f"Tracking complete in {time.time()-t0:.2f}s")
    
    # Save the dense keyframes
    out_file = os.path.join(args.output_dir, "homography_dense.json")
    with open(out_file, "w") as f:
        json.dump({"keyframes": dense_keyframes}, f, indent=2)
        
    # Overwrite the original keyframes with the dense one so phase9 automatically uses it
    os.replace(out_file, calib_file)
    print(f"Saved {len(dense_keyframes)} auto-calibrated frames to {calib_file}")
    
if __name__ == "__main__":
    main()
