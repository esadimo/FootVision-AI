"""
Stress-test the automatic PTZ calibration on SoccerNet sequences (GT person boxes are used only
to mask players out of the line mask, exactly like detections would be in the pipeline).

Usage:
    python scripts/check_calibration.py --out runs/calib_check data/SoccerNet/tracking/test/SNMOT-116 ...
Writes one JSON line per sequence to <out>/results.jsonl, the per-frame homographies to
<out>/<seq>.json and a 4-frame overlay strip (projected pitch template in red) to <out>/<seq>.jpg.
"""
import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
import pandas as pd

from src.calibration.camera_model import _cues_for, calibrate_sequence, get_template, pitch_error_m
from src.calibration.line_tracker import _project, line_mask
from src.data.soccernet import Sequence
from src.evaluation.metrics import calibration_height_check


def check(seq_dir: str, out: str) -> dict:
    seq = Sequence(seq_dir)
    g = pd.DataFrame(seq.gt)
    g = g[g.role != "ball"]
    boxes = {int(k): v[["x1", "y1", "x2", "y2"]].to_numpy(float) for k, v in g.groupby("frame")}
    t = time.time()
    cams, infos = calibrate_sequence(seq.frame_paths, None, boxes, verbose=True)
    errs = [pitch_error_m(cams[i], *_cues_for(cv2.imread(seq.frame_paths[i]), boxes.get(i + 1)))
            for i in range(0, seq.n_frames, 25)]
    e = pd.DataFrame(errs)
    res = {"seq": seq.name, "secs": round(time.time() - t), "C": np.round(cams[0].C, 1).tolist(),
           "in3": round(float(np.nanmean([x.get("in3", np.nan) for x in infos])), 3),
           "quality": round(float(np.mean([x.get("quality", 0.0) for x in infos])), 3),
           "err_med_m": round(float(e.err_med_m.median()), 3),
           "within_0.5m": round(float(e["within_0.5m"].median()), 3),
           "err_p90_m": round(float(e.err_med_m.quantile(0.9)), 3),
           "frames_bad": int(((e.err_med_m > 1.0) | (e.clutter > 0.5)).sum()), "n_sampled": len(e)}
    # ground-truth check independent of the line detector (see calibration_height_check)
    res.update({k: round(v, 3) for k, v in calibration_height_check([c.H() for c in cams], g).items()})
    json.dump([c.H().tolist() for c in cams], open(os.path.join(out, seq.name + ".json"), "w"))
    strip = []
    for i in np.linspace(0, seq.n_frames - 1, 4).astype(int):
        f = cv2.imread(seq.frame_paths[i]); v = f.copy(); v[line_mask(f) > 0] = (0, 255, 255)
        for x, y in _project(cams[i].G(), get_template(cams[i].L, cams[i].W)).astype(int):
            if 0 <= x < f.shape[1] and 0 <= y < f.shape[0]:
                cv2.circle(v, (x, y), 3, (0, 0, 255), -1)
        strip.append(cv2.resize(v, (480, 270)))
    cv2.imwrite(os.path.join(out, seq.name + ".jpg"), np.hstack(strip))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seqs", nargs="+")
    ap.add_argument("--out", default="runs/calib_check")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for s in a.seqs:
        try:
            r = check(s, a.out)
        except Exception as ex:  # keep going; record the failure
            r = {"seq": os.path.basename(s), "error": repr(ex), "trace": traceback.format_exc()[-1500:]}
        print(json.dumps(r), flush=True)
        with open(os.path.join(a.out, "results.jsonl"), "a") as fh:
            fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
