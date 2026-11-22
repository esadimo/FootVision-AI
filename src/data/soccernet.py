"""
src.data.soccernet — Loader for SoccerNet-Tracking (MOT-format) sequences.

A sequence folder looks like::

    SNMOT-062/
      img1/000001.jpg ...
      gt/gt.txt            frame,id,x,y,w,h,conf,-1,-1,-1   (1-indexed frames)
      seqinfo.ini          frameRate, seqLength, imWidth, imHeight
      gameinfo.ini         trackletID_<id>= <role description>;<jersey>

`gameinfo.ini` maps every GT track id to a role such as
"player team left", "goalkeeper team right", "referee", "ball" or "other".
We normalise those into a small vocabulary used everywhere in evaluation and
training:

    role  : player | goalkeeper | referee | ball | other
    team  : left | right | None
"""

from __future__ import annotations

import configparser
import os
from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, List, Optional

import numpy as np

ROLES = ("player", "goalkeeper", "referee", "ball", "other")

# Detector class ids used for the fine-tuned model (see scripts/prepare_yolo_dataset.py)
DET_CLASSES = {"player": 0, "goalkeeper": 1, "referee": 2, "ball": 3}
DET_NAMES = {v: k for k, v in DET_CLASSES.items()}


def parse_role(desc: str) -> Dict[str, Optional[str]]:
    """Normalise a gameinfo role description into {'role', 'team', 'jersey'}."""
    raw = desc.strip()
    jersey = None
    if ";" in raw:
        raw, jersey = raw.split(";", 1)
        jersey = jersey.strip() or None
    text = raw.strip().lower()

    if "ball" in text:
        role = "ball"
    elif "goalkeeper" in text or "goalie" in text:
        role = "goalkeeper"
    elif "referee" in text:
        role = "referee"
    elif "player" in text:
        role = "player"
    else:
        role = "other"

    team = None
    if "left" in text:
        team = "left"
    elif "right" in text:
        team = "right"
    return {"role": role, "team": team, "jersey": jersey, "raw": raw.strip()}


@dataclass
class Sequence:
    path: str
    name: str = field(init=False)

    def __post_init__(self):
        self.path = os.path.abspath(self.path.rstrip("/\\"))
        self.name = os.path.basename(self.path)

    # ── metadata ────────────────────────────────────────────────────────────
    @cached_property
    def seqinfo(self) -> Dict[str, str]:
        p = os.path.join(self.path, "seqinfo.ini")
        cfg = configparser.ConfigParser()
        if os.path.exists(p):
            cfg.read(p)
            return dict(cfg["Sequence"]) if "Sequence" in cfg else {}
        return {}

    @property
    def fps(self) -> float:
        return float(self.seqinfo.get("framerate", 25))

    @cached_property
    def frame_paths(self) -> List[str]:
        d = os.path.join(self.path, "img1")
        files = sorted(f for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        return [os.path.join(d, f) for f in files]

    @property
    def n_frames(self) -> int:
        return len(self.frame_paths)

    @cached_property
    def image_size(self):
        w = int(self.seqinfo.get("imwidth", 1920))
        h = int(self.seqinfo.get("imheight", 1080))
        return w, h

    # ── ground truth ────────────────────────────────────────────────────────
    @property
    def has_gt(self) -> bool:
        return os.path.exists(os.path.join(self.path, "gt", "gt.txt"))

    @cached_property
    def roles(self) -> Dict[int, Dict[str, Optional[str]]]:
        """GT track id → parsed role dict (from gameinfo.ini)."""
        p = os.path.join(self.path, "gameinfo.ini")
        out: Dict[int, Dict[str, Optional[str]]] = {}
        if not os.path.exists(p):
            return out
        with open(p, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line.lower().startswith("trackletid_") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                try:
                    tid = int(key.split("_", 1)[1])
                except ValueError:
                    continue
                out[tid] = parse_role(val)
        return out

    @cached_property
    def gt(self) -> np.ndarray:
        """
        Structured GT array with fields:
            frame (1-indexed), id, x1, y1, x2, y2, role, team
        role / team are strings resolved through gameinfo.ini.
        """
        p = os.path.join(self.path, "gt", "gt.txt")
        raw = np.loadtxt(p, delimiter=",", ndmin=2)
        roles = self.roles
        dtype = [("frame", "i4"), ("id", "i4"), ("x1", "f4"), ("y1", "f4"),
                 ("x2", "f4"), ("y2", "f4"), ("role", "U12"), ("team", "U6")]
        arr = np.zeros(len(raw), dtype=dtype)
        arr["frame"] = raw[:, 0].astype(int)
        arr["id"] = raw[:, 1].astype(int)
        arr["x1"] = raw[:, 2]
        arr["y1"] = raw[:, 3]
        arr["x2"] = raw[:, 2] + raw[:, 4]
        arr["y2"] = raw[:, 3] + raw[:, 5]
        for i, tid in enumerate(arr["id"]):
            r = roles.get(int(tid), {"role": "other", "team": None})
            arr["role"][i] = r["role"]
            arr["team"][i] = r["team"] or ""
        return arr

    def gt_frame(self, frame: int) -> np.ndarray:
        g = self.gt
        return g[g["frame"] == frame]


def list_sequences(root: str) -> List[Sequence]:
    """All sequence folders (containing img1/) directly under `root`."""
    out = []
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if os.path.isdir(os.path.join(p, "img1")):
            out.append(Sequence(p))
    return out
