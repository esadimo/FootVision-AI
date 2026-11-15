"""
FootVision AI — Master Pipeline Runner

Chains Phases 6-14 on any SoccerNet MOT sequence with a single command.

Usage:
    python scripts/run_pipeline.py --seq_dir data/raw/SNMOT-062 --no_viewer

    # Run on a different sequence:
    python scripts/run_pipeline.py --seq_dir data/raw/SNMOT-116 --no_viewer

Notes:
    - Phase 9 (homography calibration) requires an interactive one-time landmark-
      clicking session unless a keyframe file already exists for the sequence.
    - Kit colors are discovered automatically from the first ~120 frames via
      K-Means clustering — no hardcoding required.
    - All outputs are saved to <output_dir>/<seq_name>_phase*.* pattern.

Flags:
    --seq_dir        Path to SoccerNet MOT sequence (must contain img1/)
    --output_dir     Output directory (default: outputs/)
    --no_viewer      Run headless (no OpenCV popup windows)
    --start_phase    Skip phases before this number (useful for resuming)
    --end_phase      Stop after this phase (default: 14)
    --sample_frames  Frames used for kit color discovery in Phase 8 (default: 120)
    --ball_conf      YOLO confidence for ball detection (default: 0.05)
    --player_conf    YOLO confidence for player detection (default: 0.20)
"""

import os
import sys
import subprocess
import argparse
import time

# ── Project-root import fix ───────────────────────────────────────────────────
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def run_phase(phase_num: int, script: str, extra_args: list, label: str) -> bool:
    """
    Run a single phase script as a subprocess, streaming its output.
    Returns True on success, False on failure.
    """
    cmd = [sys.executable, script] + extra_args
    print(f"\n{'=' * 70}")
    print(f"  ▶  Phase {phase_num}: {label}")
    print(f"{'=' * 70}")
    print(f"  CMD: {' '.join(cmd)}\n")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=project_root)
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"\n  ✓  Phase {phase_num} completed in {elapsed:.1f}s")
        return True
    else:
        print(f"\n  ✗  Phase {phase_num} FAILED (exit code {result.returncode})")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="FootVision AI — Master Pipeline Runner (Generalized, any SoccerNet sequence)"
    )
    parser.add_argument("--seq_dir",       default="data/raw/SNMOT-062",
                        help="Path to SoccerNet MOT sequence directory")
    parser.add_argument("--output_dir",    default="outputs",
                        help="Output directory (default: outputs/)")
    parser.add_argument("--no_viewer",     action="store_true",
                        help="Run headless without any popup windows")
    parser.add_argument("--start_phase",   type=int, default=6,
                        help="Start pipeline from this phase (default: 6)")
    parser.add_argument("--end_phase",     type=int, default=14,
                        help="Stop pipeline after this phase (default: 14)")
    parser.add_argument("--sample_frames", type=int, default=120,
                        help="Frames for kit color discovery in Phase 8 (default: 120)")
    parser.add_argument("--ball_conf",     type=float, default=0.05,
                        help="YOLO confidence threshold for ball (default: 0.05)")
    parser.add_argument("--player_conf",   type=float, default=0.20,
                        help="YOLO confidence threshold for players (default: 0.20)")
    args = parser.parse_args()

    seq_name = os.path.basename(args.seq_dir.rstrip("/\\"))

    # Build common flag list
    no_viewer_flag = ["--no_viewer"] if args.no_viewer else []
    seq_flag       = ["--seq_dir", args.seq_dir]
    outdir_flag    = ["--output_dir", args.output_dir]
    common         = seq_flag + outdir_flag + no_viewer_flag

    print(f"\n{'#' * 70}")
    print(f"  FootVision AI — Full Pipeline")
    print(f"  Sequence  : {seq_name}")
    print(f"  Output    : {args.output_dir}/")
    print(f"  Phases    : {args.start_phase} -> {args.end_phase}")
    print(f"{'#' * 70}")

    # ── Phase definitions ─────────────────────────────────────────────────────
    phases = [
        # (phase_num, script, extra_args, label)
        (6,  "scripts/phase6_full_detection_pipeline.py",
              common + ["--threshold", str(args.player_conf)],
              "Full Person Detection (YOLO + ByteTrack)"),

        (7,  "scripts/phase7_player_tracking.py",
              common + ["--threshold", str(args.player_conf)],
              "Player Tracking (ByteTrack persistent)"),

        (8,  "scripts/phase8_team_classification.py",
              common + ["--threshold", str(args.player_conf),
                        "--sample_frames", str(args.sample_frames)],
              "Auto Kit Color Discovery + Team Classification (K-Means)"),

        (9,  "scripts/phase9_pitch_radar.py",
              common,
              "Pitch Radar / Homography Calibration"),

        (10, "scripts/phase10_ball_tracking.py",
              common + ["--conf", str(args.ball_conf)],
              "Ball Detection & Tracking (YOLOv8, dampened extrapolation)"),

        (11, "scripts/phase11_possession.py",
              common,
              "Possession Estimation (2.5m proximity, 3-frame state machine)"),

        (12, "scripts/phase12_pass_detection.py",
              common,
              "Pass & Turnover Detection"),

        (13, "scripts/phase13_statistics.py",
              common,
              "Player Statistics (distance, speed, heatmaps, team spatial metrics)"),

        (14, "scripts/phase14_ultimate_video.py",
              common,
              "Ultimate Dashboard Video (broadcast output, all phases combined)"),
    ]

    # ── Run selected phases ───────────────────────────────────────────────────
    failed_phases = []
    total_start = time.time()

    for phase_num, script, extra_args, label in phases:
        if phase_num < args.start_phase or phase_num > args.end_phase:
            print(f"  [Skip] Phase {phase_num}: {label}")
            continue

        ok = run_phase(phase_num, script, extra_args, label)
        if not ok:
            failed_phases.append(phase_num)
            print(f"\n  Pipeline halted at Phase {phase_num}.")
            print(f"  Fix the error and re-run with --start_phase {phase_num}")
            break

    total_elapsed = time.time() - total_start

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'#' * 70}")
    if not failed_phases:
        print(f"  ✓  Pipeline complete for '{seq_name}' in {total_elapsed:.0f}s")
        print(f"\n  Key outputs:")
        print(f"    {args.output_dir}/{seq_name}_kit_profile.json       ← discovered kit colors")
        print(f"    {args.output_dir}/{seq_name}_phase8_teams.mp4       ← team classification video")
        print(f"    {args.output_dir}/{seq_name}_phase10_ball.csv       ← ball tracking data")
        print(f"    {args.output_dir}/{seq_name}_phase11_possession.csv ← possession timeline")
        print(f"    {args.output_dir}/{seq_name}_phase12_events.csv     ← passes & turnovers")
        print(f"    {args.output_dir}/{seq_name}_phase13_stats.csv      ← player stats")
        print(f"    {args.output_dir}/{seq_name}_phase14_ultimate.mp4   ← broadcast video")
    else:
        print(f"  ✗  Pipeline failed at phase(s): {failed_phases}")
    print(f"{'#' * 70}\n")


if __name__ == "__main__":
    main()
