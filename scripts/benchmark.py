"""
Benchmark detection / tracking / ball against SoccerNet ground truth on cached detections.

Usage:
    python scripts/benchmark.py --tag ft1280 --seqs "data/SoccerNet/tracking/test/SNMOT-*" \
        --trackers bytetrack botsort ocsort --out runs/bench_ft1280.csv
Trackers are Ultralytics implementations configured from their default YAML plus overrides
given as name:key=val,key=val (e.g. "botsort:track_buffer=60,match_thresh=0.9").
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.data.soccernet import Sequence  # noqa: E402
from src.evaluation.metrics import evaluate_ball, evaluate_detection, evaluate_tracking  # noqa: E402
from src.pipeline.detect_track import BALL_CLASS, PERSON_CLASSES, cache_path, dets_to_frame, track_cached  # noqa: E402
from src.tracking.ball_trajectory import BallTrajectoryParams, solve_ball_trajectory  # noqa: E402


def tracker_cfg(spec: str) -> dict:
    import ultralytics
    name, _, ov = spec.partition(":")
    base = os.path.join(os.path.dirname(ultralytics.__file__), "cfg", "trackers", f"{name}.yaml")
    cfg = yaml.safe_load(open(base))
    for kv in filter(None, ov.split(",")):
        k, v = kv.split("=")
        cfg[k] = yaml.safe_load(v)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--cache_dir", default="data/cache")
    ap.add_argument("--trackers", nargs="*", default=["bytetrack"])
    ap.add_argument("--det_conf", type=float, nargs="*", default=[0.25])
    ap.add_argument("--ball", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = []
    paths = sorted({p for s in a.seqs for p in glob.glob(s)})
    for p in paths:
        seq = Sequence(p)
        cp = cache_path(a.cache_dir, seq, a.tag)
        if not os.path.exists(cp):
            print("missing cache", cp); continue
        z = np.load(cp); dets = {k: z[k] for k in z.files}
        df = dets_to_frame(dets)
        people = df[df.cls.isin(PERSON_CLASSES)]
        for c in a.det_conf:
            r = evaluate_detection(seq, people, conf_thr=c)
            rows.append({"seq": seq.name, "kind": "det", "cfg": f"conf{c}", **r})
        for spec in a.trackers:
            t = time.time()
            tr = track_cached(seq, dets, tracker_cfg(spec))
            r = evaluate_tracking(seq, tr)
            rows.append({"seq": seq.name, "kind": "track", "cfg": spec, **r, "secs": time.time() - t})
        if a.ball:
            b = df[df.cls == BALL_CLASS].copy()
            b["x"] = (b.x1 + b.x2) / 2; b["y"] = (b.y1 + b.y2) / 2
            # raw: most confident detection per frame (old behaviour)
            raw = b.sort_values("conf", ascending=False).groupby("frame").head(1)
            for tol in (10, 20):
                rows.append({"seq": seq.name, "kind": "ball", "cfg": f"raw_top1@{tol}", **evaluate_ball(seq, raw, tol)})
                traj = solve_ball_trajectory(b, seq.n_frames)
                rows.append({"seq": seq.name, "kind": "ball", "cfg": f"dp@{tol}", **evaluate_ball(seq, traj, tol)})
        print(f"{seq.name} done", flush=True)

    res = pd.DataFrame(rows)
    num = res.select_dtypes("number").columns
    summary = res.groupby(["kind", "cfg"])[list(num)].mean().round(4)
    pd.set_option("display.width", 220)
    print(summary.to_string())
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        res.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
