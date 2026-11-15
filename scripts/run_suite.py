"""
Run the full pipeline over many SoccerNet sequences (detections must be cached first, or the
model is loaded per sequence) with N parallel CPU workers, then evaluate against GT.

Usage:
    python scripts/run_suite.py --seqs "data/SoccerNet/tracking/test/SNMOT-*" --out outputs_test --workers 3
"""
import argparse
import glob
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--out", default="outputs_test")
    ap.add_argument("--model", default="models/detection/fv_yolo26s_1280.pt")
    ap.add_argument("--tag", default="ft1280")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--eval_csv", default=None)
    a = ap.parse_args()
    paths = sorted({p for s in a.seqs for p in glob.glob(s)})
    os.makedirs(a.out, exist_ok=True)
    py = sys.executable

    def one(p):
        name = os.path.basename(p)
        if os.path.exists(os.path.join(a.out, name, "summary.json")):
            return name, 0, 0.0
        t = time.time()
        with open(os.path.join(a.out, f"{name}.log"), "w") as log:
            rc = subprocess.call([py, "scripts/run_pipeline.py", "--seq_dir", p, "--output_dir", a.out,
                                  "--model", a.model, "--tag", a.tag], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        print(f"{name}: rc={rc} {time.time() - t:.0f}s", flush=True)
        return name, rc, time.time() - t

    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(one, paths))
    failed = [n for n, rc, _ in res if rc]
    print("failed:", failed, flush=True)
    subprocess.call([py, "scripts/evaluate_pipeline.py", "--runs", a.out, "--seqs", *paths,
                     "--out", a.eval_csv or os.path.join(a.out, "evaluation.csv")], cwd=ROOT)


if __name__ == "__main__":
    main()
