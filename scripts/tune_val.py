"""
Re-run the pipeline on the game-4 validation clips with config overrides, reusing the
cached detections and the calibration from a previous run, then score against GT.

    python scripts/tune_val.py --out runs/tune/split --set split_teams=True
    python scripts/tune_val.py --out runs/tune/nosplit --set split_teams=False

--set accepts dotted keys into PipelineConfig (e.g. events.min_ctrl=5, stitch.max_cost=1.2,
tracker.match_thresh=0.7).  Only tune here (game 4); the test games are for the final report.
"""
import argparse
import ast
import glob
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "scripts"))

VAL = [f"data/SoccerNet/tracking/train/SNMOT-{i:03d}" for i in range(60, 78)]


def make_cfg(sets):
    from src.pipeline.run import PipelineConfig
    cfg = PipelineConfig()
    for s in sets:
        k, v = s.split("=", 1)
        try:
            v = ast.literal_eval(v)
        except Exception:
            pass
        obj, parts = cfg, k.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        if isinstance(obj, dict):
            obj[parts[-1]] = v
        else:
            setattr(obj, parts[-1], v)
    return cfg


def run_one(args):
    seq_dir, out, sets, calib_from = args
    import warnings
    warnings.filterwarnings("ignore")
    from src.pipeline.run import run_sequence
    name = os.path.basename(seq_dir)
    os.makedirs(os.path.join(out, name), exist_ok=True)
    src = os.path.join(calib_from, name, "calibration.json")
    if os.path.exists(src):
        shutil.copy(src, os.path.join(out, name, "calibration.json"))
    logf = open(os.path.join(out, name + ".log"), "w")
    run_sequence(seq_dir, out_root=out, cfg=make_cfg(sets), legacy=False,
                 log=lambda m: (logf.write(m + "\n"), logf.flush()))
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--calib_from", default="outputs_val")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seqs", nargs="*", default=None)
    a = ap.parse_args()
    seqs = a.seqs or VAL
    with ProcessPoolExecutor(a.workers) as ex:
        for n in ex.map(run_one, [(s, a.out, a.set, a.calib_from) for s in seqs]):
            print(n, "done", flush=True)
    import pandas as pd
    import warnings
    warnings.filterwarnings("ignore")
    from evaluate_pipeline import evaluate
    r = pd.DataFrame([evaluate(s, a.out) for s in seqs])
    r.to_csv(os.path.join(a.out, "evaluation.csv"), index=False)
    print(r.drop(columns=["seq"]).mean(numeric_only=True).round(4).to_string())


if __name__ == "__main__":
    main()
