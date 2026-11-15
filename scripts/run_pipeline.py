"""
FootVision AI — pipeline runner.

Default engine (accurate, single pass — see src/pipeline/run.py):
    python scripts/run_pipeline.py --seq_dir data/SoccerNet/tracking/test/SNMOT-116
    python scripts/run_pipeline.py --seq_dir data/raw/SNMOT-062 --video     # + broadcast video (phase 14)

    Calibration is fully automatic. Optionally pass a hand-picked first-frame homography
    (from scripts/phase9_pick_landmarks.py) with --homography.

Outputs: outputs/<seq>/{tracks,ball,possession,events,player_stats,team_metrics}.csv,
         calibration.json, summary.json  + legacy outputs/<seq>_phase*.csv used by app.py.

The original phase-by-phase runner is kept as scripts/run_pipeline_legacy.py.
"""
import argparse
import os
import subprocess
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)


def main():
    ap = argparse.ArgumentParser(description="FootVision AI — accurate single-pass pipeline")
    ap.add_argument("--seq_dir", default="data/raw/SNMOT-062")
    ap.add_argument("--output_dir", default="outputs")
    ap.add_argument("--model", default="models/detection/fv_yolo26s_1280.pt")
    ap.add_argument("--tag", default="ft1280", help="detection-cache tag (change when the model changes)")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--homography", default=None, help="optional manual first-frame homography json")
    ap.add_argument("--video", action="store_true", help="also render the phase-14 broadcast video")
    ap.add_argument("--no_viewer", action="store_true", help="accepted for compatibility")
    a = ap.parse_args()

    from src.pipeline.run import PipelineConfig, run_sequence
    cfg = PipelineConfig(model=a.model, tag=a.tag, device=a.device, H_init=a.homography)
    run_sequence(a.seq_dir, out_root=a.output_dir, cfg=cfg)
    if a.video:
        subprocess.run([sys.executable, "scripts/phase14_ultimate_video.py", "--seq_dir", a.seq_dir,
                        "--output_dir", a.output_dir, "--no_viewer"], cwd=project_root, check=True)


if __name__ == "__main__":
    main()
