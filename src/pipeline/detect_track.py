"""
src.pipeline.detect_track — run the detector once per sequence, cache all boxes,
and run any Ultralytics tracker on the cached boxes.

Why cache: detection is ~95 % of the compute.  With cached boxes, tracker
variants / thresholds can be benchmarked against ground truth in seconds.

Cache format (npz, one per sequence & model):
    frame (int, 1-indexed), xyxy (N,4 float32), conf (N,), cls (N,)
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Dict, Iterable, Optional

import cv2
import numpy as np
import pandas as pd

from src.data.soccernet import Sequence

PERSON_CLASSES = (0, 1, 2)   # player, goalkeeper, referee (fine-tuned model)
BALL_CLASS = 3


# ── detection cache ───────────────────────────────────────────────────────────

def cache_path(cache_dir: str, seq: Sequence, tag: str) -> str:
    return os.path.join(cache_dir, f"{seq.name}__{tag}.npz")


def detect_sequence(seq: Sequence, model, imgsz: int = 1280, conf: float = 0.05,
                    iou: float = 0.6, class_map: Optional[Dict[int, int]] = None,
                    batch: int = 8, augment: bool = False, half: bool = False, device=None) -> Dict[str, np.ndarray]:
    """
    Run `model` (ultralytics YOLO) on every frame.  `class_map` remaps model class ids
    to our ids (e.g. COCO {0: 0, 32: 3}); classes not in the map are dropped.
    """
    frames, boxes, confs, clss = [], [], [], []
    paths = seq.frame_paths
    for s in range(0, len(paths), batch):
        chunk = paths[s:s + batch]
        res = model.predict(chunk, imgsz=imgsz, conf=conf, iou=iou, verbose=False, device=device,
                            augment=augment, half=half)
        for k, r in enumerate(res):
            b = r.boxes
            if b is None or len(b) == 0:
                continue
            c = b.cls.cpu().numpy().astype(int)
            xyxy = b.xyxy.cpu().numpy().astype(np.float32)
            cf = b.conf.cpu().numpy().astype(np.float32)
            if class_map is not None:
                keep = np.array([ci in class_map for ci in c], bool)
                c = np.array([class_map[ci] for ci in c[keep]], int)
                xyxy, cf = xyxy[keep], cf[keep]
            frames.append(np.full(len(c), s + k + 1, np.int32))
            boxes.append(xyxy); confs.append(cf); clss.append(c)
    if not frames:
        return {"frame": np.zeros(0, np.int32), "xyxy": np.zeros((0, 4), np.float32),
                "conf": np.zeros(0, np.float32), "cls": np.zeros(0, np.int32)}
    return {"frame": np.concatenate(frames), "xyxy": np.concatenate(boxes),
            "conf": np.concatenate(confs), "cls": np.concatenate(clss).astype(np.int32)}


def load_or_detect(seq: Sequence, model_path: str, cache_dir: str, tag: str, **kw) -> Dict[str, np.ndarray]:
    p = cache_path(cache_dir, seq, tag)
    if os.path.exists(p):
        z = np.load(p)
        return {k: z[k] for k in z.files}
    from ultralytics import YOLO
    model = YOLO(model_path)
    d = detect_sequence(seq, model, **kw)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(p, **d)
    return d


def dets_to_frame(d: Dict[str, np.ndarray]) -> pd.DataFrame:
    df = pd.DataFrame(d["xyxy"], columns=["x1", "y1", "x2", "y2"])
    df["frame"] = d["frame"]; df["conf"] = d["conf"]; df["cls"] = d["cls"]
    return df


# ── tracker adapter ───────────────────────────────────────────────────────────

class _Boxes:
    """Minimal numpy stand-in for ultralytics Boxes, as expected by tracker.update()."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xyxy = xyxy.astype(np.float32)
        self.conf = conf.astype(np.float32)
        self.cls = cls.astype(np.float32)

    @property
    def xywh(self):
        x1, y1, x2, y2 = self.xyxy.T
        return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], 1)

    @property
    def data(self):
        return np.c_[self.xyxy, self.conf, self.cls]

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, idx):
        return _Boxes(self.xyxy[idx], self.conf[idx], self.cls[idx])


def make_tracker(cfg: Dict):
    from ultralytics.trackers.track import TRACKER_MAP
    args = SimpleNamespace(**cfg)
    args.device = "cpu"
    return TRACKER_MAP[cfg["tracker_type"]](args=args)


def track_cached(seq: Sequence, dets: Dict[str, np.ndarray], cfg: Dict,
                 classes: Iterable[int] = PERSON_CLASSES, need_frames: Optional[bool] = None,
                 class_agnostic: bool = True) -> pd.DataFrame:
    """
    Run a tracker (Ultralytics implementation) over cached detections.

    class_agnostic: feed all person classes as one class so a player predicted as
    'referee' for one frame does not break its track.
    """
    tracker = make_tracker(cfg)
    if need_frames is None:
        need_frames = str(cfg.get("gmc_method", "none")).lower() not in ("none", "")
    sel = np.isin(dets["cls"], list(classes))
    fr, xyxy, conf, cls = dets["frame"][sel], dets["xyxy"][sel], dets["conf"][sel], dets["cls"][sel]
    order = np.argsort(fr, kind="stable")
    fr, xyxy, conf, cls = fr[order], xyxy[order], conf[order], cls[order]
    starts = np.searchsorted(fr, np.arange(1, seq.n_frames + 2))

    rows = []
    for f in range(1, seq.n_frames + 1):
        a, b = starts[f - 1], starts[f]
        c_in = np.zeros(b - a) if class_agnostic else cls[a:b]
        boxes = _Boxes(xyxy[a:b], conf[a:b], c_in)
        img = cv2.imread(seq.frame_paths[f - 1]) if need_frames else np.zeros((8, 8, 3), np.uint8)
        out = tracker.update(boxes, img)
        for t in out:
            x1, y1, x2, y2, tid, sc, _c, idx = t[:8]
            idx = int(idx)
            rows.append((f, int(tid), x1, y1, x2, y2, float(sc), int(cls[a + idx])))
    return pd.DataFrame(rows, columns=["frame", "track_id", "x1", "y1", "x2", "y2", "conf", "cls"])
