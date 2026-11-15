"""
src.tracking.stitching — offline re-linking of fragmented tracks (tracklets).

Online trackers break a player's identity whenever he is occluded or leaves the
frame for more than `track_buffer` frames, and on fast pans.  Because we
process recorded video we can look at the whole clip and join fragments:

  A ends at frame a_end, B starts at b_start > a_end (gap g frames):
    • physically reachable: on the *pitch* (metres, camera-independent) the
      distance from A's last position (extrapolated with its velocity, capped)
      to B's first position must be ≤ base_m + vmax · g / fps
    • same role (player / goalkeeper / referee) and, for players, same team
    • similar appearance (kit colour histogram), when features are given
  Cost = normalised distance + w_app · appearance distance + gap penalty.
  Links are accepted greedily by increasing cost (each tracklet has at most one
  successor and one predecessor), which is near-optimal for this sparse problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd


@dataclass
class StitchParams:
    max_gap: int = 125          # frames (5 s at 25 fps)
    vmax_mps: float = 8.0       # sprint speed bound for reachability
    base_m: float = 2.5         # slack for projection / box noise
    w_app: float = 2.0          # weight of appearance distance
    gap_weight: float = 0.002   # cost per frame of gap
    max_cost: float = 1.5
    fps: float = 25.0
    min_len: int = 3


def foot_points_pitch(tracks: pd.DataFrame, H_by_frame: Dict[int, np.ndarray]) -> pd.DataFrame:
    """Add pitch_x / pitch_y (metres) from the bottom-centre of each box."""
    t = tracks.copy()
    px = np.full(len(t), np.nan); py = np.full(len(t), np.nan)
    for f, idx in t.groupby("frame").indices.items():
        H = H_by_frame.get(int(f))
        if H is None:
            continue
        g = t.iloc[idx]
        pts = np.stack([(g.x1 + g.x2).to_numpy() / 2, g.y2.to_numpy()], 1)[None].astype(np.float64)
        q = cv2.perspectiveTransform(pts, np.asarray(H, np.float64))[0]
        px[idx] = q[:, 0]; py[idx] = q[:, 1]
    t["pitch_x"] = px; t["pitch_y"] = py
    return t


def _summaries(t: pd.DataFrame, fps: float) -> pd.DataFrame:
    rows = []
    for tid, g in t.sort_values("frame").groupby("track_id"):
        g = g.dropna(subset=["pitch_x"])
        if g.empty:
            continue
        head, tail = g.head(5), g.tail(5)
        def vel(s):
            if len(s) < 2:
                return np.zeros(2)
            dt = (s.frame.iloc[-1] - s.frame.iloc[0]) / fps
            return np.array([s.pitch_x.iloc[-1] - s.pitch_x.iloc[0], s.pitch_y.iloc[-1] - s.pitch_y.iloc[0]]) / max(dt, 1e-3)
        rows.append({"track_id": tid, "start": int(g.frame.iloc[0]), "end": int(g.frame.iloc[-1]), "n": len(g),
                     "p0": head[["pitch_x", "pitch_y"]].median().to_numpy(),
                     "p1": tail[["pitch_x", "pitch_y"]].median().to_numpy(),
                     "v0": vel(g.head(10)), "v1": vel(g.tail(10))})
    return pd.DataFrame(rows).set_index("track_id")


def stitch_tracklets(tracks: pd.DataFrame, H_by_frame: Dict[int, np.ndarray],
                     labels: Optional[Dict[int, str]] = None,
                     appearance: Optional[Dict[int, np.ndarray]] = None,
                     p: StitchParams = StitchParams()) -> pd.DataFrame:
    """
    tracks     : frame, track_id, x1..y2 (one tracker run)
    labels     : track_id → compatibility label (e.g. 'player:A', 'referee'); links require equality
    appearance : track_id → L2-normalised appearance vector
    Returns tracks with track_id rewritten so stitched fragments share the first fragment's id.
    """
    t = tracks if "pitch_x" in tracks else foot_points_pitch(tracks, H_by_frame)
    s = _summaries(t, p.fps)
    s = s[s.n >= p.min_len]
    ids = list(s.index)
    links = []
    for a in ids:
        A = s.loc[a]
        for b in ids:
            if a == b:
                continue
            B = s.loc[b]
            gap = B.start - A.end
            if gap <= 0 or gap > p.max_gap:
                continue
            if labels is not None and labels.get(a) != labels.get(b):
                continue
            dt = gap / p.fps
            # extrapolate A forward (velocity capped, decays after ~1 s)
            v = A.v1
            sp = np.linalg.norm(v)
            if sp > p.vmax_mps:
                v = v * p.vmax_mps / sp
            pred = A.p1 + v * min(dt, 1.0)
            d = float(np.linalg.norm(B.p0 - pred))
            reach = p.base_m + p.vmax_mps * dt
            if d > reach:
                continue
            cost = d / reach + p.gap_weight * gap
            if appearance is not None and a in appearance and b in appearance:
                cost += p.w_app * float(np.linalg.norm(appearance[a] - appearance[b])) / 2.0
            if cost <= p.max_cost:
                links.append((cost, a, b))
    links.sort()
    succ, pred_of = {}, {}
    for cost, a, b in links:
        if a in succ or b in pred_of:
            continue
        # no cycles possible (time-ordered); also forbid joining if b's chain overlaps a's chain
        succ[a] = b; pred_of[b] = a
    root = {}
    for tid in t.track_id.unique():
        r = tid
        while r in pred_of:
            r = pred_of[r]
        root[tid] = r
    out = tracks.copy()
    out["track_id"] = out.track_id.map(root)
    return out
