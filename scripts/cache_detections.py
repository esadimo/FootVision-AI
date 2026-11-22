"""
Run a detector over SoccerNet sequences once and cache every box (conf ≥ --conf).

Usage:
    python scripts/cache_detections.py --model models/detection/fv_yolo26s_1280.pt --tag ft1280 \
        --seqs data/SoccerNet/tracking/test/SNMOT-1*  --device mps
    # stock COCO model: map person→player(0), sports ball→ball(3)
    python scripts/cache_detections.py --model models/detection/yolo26s.pt --coco --tag coco1280 ...
"""
import argparse
import glob
import os
import sys
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.data.soccernet import Sequence  # noqa: E402
from src.pipeline.detect_track import cache_path, detect_sequence  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--cache_dir", default="data/cache")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--coco", action="store_true")
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.model)
    model.to(a.device)
    class_map = {0: 0, 32: 3} if a.coco else None
    paths = sorted({p for s in a.seqs for p in glob.glob(s)})
    for p in paths:
        seq = Sequence(p)
        out = cache_path(a.cache_dir, seq, a.tag)
        if os.path.exists(out):
            continue
        t = time.time()
        d = detect_sequence(seq, model, imgsz=a.imgsz, conf=a.conf, class_map=class_map, batch=a.batch, device=a.device)
        os.makedirs(a.cache_dir, exist_ok=True)
        import numpy as np
        np.savez_compressed(out, **d)
        print(f"{seq.name}: {len(d['conf'])} boxes in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
