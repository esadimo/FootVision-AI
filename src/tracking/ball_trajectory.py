"""
src.tracking.ball_trajectory — offline, globally consistent ball trajectory.

The old tracker (src/tracking/ball_tracker.py) trusted the single most
confident detection in each frame and extrapolated forward with a damped
constant velocity.  One false positive (a white sock, a penalty spot, a
spectator) teleported the ball, and every occlusion produced a guessed
position that drifted away from the real ball.

Here the whole clip is solved at once (we process recorded video, so the
future is available):

  * every detection with conf ≥ min_conf is a candidate node
  * candidates in frames t' < t may link if t - t' ≤ max_gap and the implied
    image speed is plausible; the link cost grows with speed and gap length
  * each node earns a score from its confidence (logit), so a low-confidence
    candidate is only used when it continues a consistent trajectory
  * the best-scoring path through this DAG (longest-path DP) is the trajectory;
    gaps between linked nodes are filled by linear interpolation (both ends known)
  * several disjoint segments are allowed (ball out of view / long occlusion):
    starting a new segment costs `restart_cost`

Output: one row per frame with x, y (NaN when unknown), conf, source
('det' | 'interp').
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class BallTrajectoryParams:
    min_conf: float = 0.05
    conf_bias: float = 1.5          # added to logit(conf): >0 favours using detections
    max_gap: int = 25               # frames a link may skip (≈1 s)
    vmax_px: float = 70.0           # max plausible image displacement per frame
    speed_weight: float = 1.0       # cost per (d / (vmax·dt))²
    gap_cost: float = 0.08          # cost per skipped frame
    restart_cost: float = 3.0       # cost to start a new segment
    max_interp_gap: int = 25        # never interpolate across longer gaps


def solve_ball_trajectory(cands: pd.DataFrame, n_frames: int,
                          p: BallTrajectoryParams = BallTrajectoryParams()) -> pd.DataFrame:
    """
    cands: DataFrame with frame (1-indexed), x, y (centre px), conf.
    Returns DataFrame(frame, x, y, conf, source) for frames 1..n_frames.
    """
    c = cands[cands.conf >= p.min_conf].sort_values(["frame", "conf"], ascending=[True, False]).reset_index(drop=True)
    out = pd.DataFrame({"frame": np.arange(1, n_frames + 1), "x": np.nan, "y": np.nan,
                        "conf": np.nan, "source": ""})
    if c.empty:
        return out

    fr = c.frame.to_numpy(); xs = c.x.to_numpy(float); ys = c.y.to_numpy(float)
    conf = np.clip(c.conf.to_numpy(float), 1e-4, 1 - 1e-4)
    emit = np.log(conf / (1 - conf)) + p.conf_bias
    n = len(c)
    score = np.full(n, -np.inf); back = np.full(n, -1)
    starts = np.searchsorted(fr, fr - p.max_gap)          # first candidate within the gap window

    for j in range(n):
        best, arg = -p.restart_cost, -1                   # start a new segment here
        lo = starts[j]
        # candidates strictly before frame fr[j]
        hi = np.searchsorted(fr, fr[j])
        if hi > lo:
            i = np.arange(lo, hi)
            dt = (fr[j] - fr[i]).astype(float)
            d = np.hypot(xs[j] - xs[i], ys[j] - ys[i])
            ratio = d / (p.vmax_px * dt)
            ok = ratio <= 1.5
            if ok.any():
                tot = score[i] - p.speed_weight * ratio ** 2 - p.gap_cost * (dt - 1)
                tot[~ok] = -np.inf
                k = int(np.argmax(tot))
                if tot[k] > best:
                    best, arg = tot[k], i[k]
        score[j] = best + emit[j]
        back[j] = arg

    # best path end (allow multiple segments: walk back through restarts)
    j = int(np.argmax(score))
    chain = []
    while j >= 0:
        chain.append(j)
        j = back[j]
    chain = chain[::-1]

    # Only the single best chain is returned; segments are joined by restarts which the
    # DP already scored, so long gaps (restarts) simply are not interpolated.
    pos = {}
    for a, b in zip(chain[:-1], chain[1:]):
        pos[fr[a]] = (xs[a], ys[a], conf[a], "det")
        gap = fr[b] - fr[a]
        linked = back[b] == a
        if linked and 1 < gap <= p.max_interp_gap:
            for t in range(1, gap):
                w = t / gap
                pos[fr[a] + t] = (xs[a] + w * (xs[b] - xs[a]), ys[a] + w * (ys[b] - ys[a]), np.nan, "interp")
    last = chain[-1]
    pos[fr[last]] = (xs[last], ys[last], conf[last], "det")

    for f, (x, y, cf, src) in pos.items():
        out.loc[f - 1, ["x", "y", "conf", "source"]] = [x, y, cf, src]
    return out
