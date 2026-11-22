"""
src.tracking.ball_zoom — second, zoomed detection pass for the ball.

At imgsz 1280 a 1920-px broadcast frame is shrunk to two thirds, and the ball
(often 6–12 px) falls below what the detector can see: on SoccerNet game 4 there
was no ball candidate within 10 px of the true ball in ~25 % of frames.

Here, for every frame, a square crop is taken around where the ball is expected —
the first-pass trajectory position, else a linear interpolation between the nearest
known positions — and the detector runs on the crop at a higher scale (crop 640 px
at imgsz 960 ⇒ the ball is 2.25× larger than in the full-frame pass).  The crop
detections are mapped back to frame coordinates and added to the candidate set;
the trajectory solver then picks a consistent path through all candidates.
"""

from __future__ import annotations

import os
from typing import Dict

import cv2
import numpy as np
import pandas as pd

BALL_CLASS = 3


def expected_positions(ball: pd.DataFrame, n_frames: int, w: int, h: int) -> np.ndarray:
    """(n_frames, 2) expected ball centre: known positions, linear fill between them, held at the ends."""
    x = pd.Series(ball.set_index("frame").x.reindex(range(1, n_frames + 1)).to_numpy(float))
    y = pd.Series(ball.set_index("frame").y.reindex(range(1, n_frames + 1)).to_numpy(float))
    if x.notna().sum() == 0:
        return np.tile([w / 2, h / 2], (n_frames, 1))
    x = x.interpolate(limit_direction="both"); y = y.interpolate(limit_direction="both")
    return np.c_[x.to_numpy(), y.to_numpy()]


def zoom_detect(frame_paths, centres: np.ndarray, model, crop: int = 640, imgsz: int = 960,
                conf: float = 0.05, batch: int = 8, device=None) -> Dict[str, np.ndarray]:
    """Run `model` on a crop around centres[i] of every frame; returns ball boxes in frame coordinates."""
    frames, boxes, confs = [], [], []
    n = len(frame_paths)
    for s in range(0, n, batch):
        imgs, offs = [], []
        for i in range(s, min(n, s + batch)):
            img = cv2.imread(frame_paths[i])
            H, W = img.shape[:2]
            cx, cy = centres[i]
            x0 = int(np.clip(cx - crop / 2, 0, max(W - crop, 0))); y0 = int(np.clip(cy - crop / 2, 0, max(H - crop, 0)))
            imgs.append(img[y0:y0 + crop, x0:x0 + crop]); offs.append((x0, y0))
        res = model.predict(imgs, imgsz=imgsz, conf=conf, verbose=False, device=device, classes=[BALL_CLASS])
        for k, r in enumerate(res):
            b = r.boxes
            if b is None or len(b) == 0:
                continue
            xyxy = b.xyxy.cpu().numpy().astype(np.float32)
            xyxy[:, [0, 2]] += offs[k][0]; xyxy[:, [1, 3]] += offs[k][1]
            frames.append(np.full(len(xyxy), s + k + 1, np.int32)); boxes.append(xyxy)
            confs.append(b.conf.cpu().numpy().astype(np.float32))
    if not frames:
        return {"frame": np.zeros(0, np.int32), "xyxy": np.zeros((0, 4), np.float32), "conf": np.zeros(0, np.float32)}
    return {"frame": np.concatenate(frames), "xyxy": np.concatenate(boxes), "conf": np.concatenate(confs)}


def load_or_zoom(seq, ball: pd.DataFrame, model_path: str, cache_dir: str, tag: str, **kw) -> pd.DataFrame:
    """Cached zoom pass → DataFrame(frame, x1..y2, conf, x, y)."""
    p = os.path.join(cache_dir, f"{seq.name}__{tag}_ballzoom.npz")
    if os.path.exists(p):
        z = np.load(p); d = {k: z[k] for k in z.files}
    else:
        from ultralytics import YOLO
        img = cv2.imread(seq.frame_paths[0])
        centres = expected_positions(ball, seq.n_frames, img.shape[1], img.shape[0])
        d = zoom_detect(seq.frame_paths, centres, YOLO(model_path), **kw)
        os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(p, **d)
    df = pd.DataFrame(d["xyxy"], columns=["x1", "y1", "x2", "y2"])
    df["frame"] = d["frame"]; df["conf"] = d["conf"]
    df["x"] = (df.x1 + df.x2) / 2; df["y"] = (df.y1 + df.y2) / 2
    return df
