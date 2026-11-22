"""
src.teams.track_splitter — cut tracks where the kit colour changes team.

The tracker sometimes swaps identities between two players in a duel.  When the two
players are opponents, the track's kit colour flips from one team to the other
part-way through.  The whole track then carries a single team label, so the
ball carrier gets the wrong team for the part after the swap (this was the
largest single source of possession errors on SoccerNet game 4: ~6 % of
player frames sat in such cross-team tracks).

For every outfield-player track we sample the jersey colour every `step` frames,
score each sample by which team centroid it is closer to, smooth the score, and
cut the track at the boundary between sustained runs of opposite teams.  The
pieces get new track ids; tracklet stitching then re-joins correct pieces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from src.teams.team_clusterer import colour_feature, jersey_crop


@dataclass
class SplitParams:
    step: int = 3              # sample every n-th frame of a track
    smooth: int = 5            # rolling-median window (samples)
    min_run: int = 8           # a team run must last this many samples (~1 s at step 3)
    min_score: float = 0.12    # median |score| a run needs to count as a confident team
    min_track: int = 40        # frames; shorter tracks are left alone


def team_scores(read_frame: Callable[[int], np.ndarray], rows: pd.DataFrame, model: Dict) -> pd.DataFrame:
    """Signed score per row: (dA − dB)/(dA + dB) < 0 ⇒ looks like team A."""
    out = []
    for f, g in rows.groupby("frame"):
        img = read_frame(int(f))
        for idx, r in g.iterrows():
            c = jersey_crop(img, r.x1, r.y1, r.x2, r.y2)
            fe = colour_feature(c) if c is not None else None
            if fe is None:
                continue
            da, db = np.linalg.norm(fe - model["A"]), np.linalg.norm(fe - model["B"])
            out.append((idx, (da - db) / max(da + db, 1e-9)))
    if not out:
        return pd.DataFrame(columns=["frame", "track_id", "score"])
    i, sc = zip(*out)
    res = rows.loc[list(i), ["frame", "track_id"]].copy()
    res["score"] = sc
    return res


def _runs(sign: np.ndarray):
    runs, s = [], 0
    for k in range(1, len(sign) + 1):
        if k == len(sign) or sign[k] != sign[s]:
            runs.append([sign[s], s, k - 1]); s = k
    return runs


def split_points(frames: np.ndarray, score: np.ndarray, p: SplitParams) -> list:
    """Frames at which to cut, from per-sample frames and smoothed scores."""
    sm = pd.Series(score).rolling(p.smooth, center=True, min_periods=1).median().to_numpy()
    sign = np.sign(sm).astype(int)
    sign[sign == 0] = 1
    runs = _runs(sign)
    # absorb short or weak runs into their neighbours until stable
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for k, (sg, a, b) in enumerate(runs):
            weak = (b - a + 1) < p.min_run or np.median(np.abs(sm[a:b + 1])) < p.min_score
            if weak:
                nb = k - 1 if k > 0 else k + 1
                if k > 0 and k + 1 < len(runs):  # pick the longer neighbour
                    nb = k - 1 if (runs[k - 1][2] - runs[k - 1][1]) >= (runs[k + 1][2] - runs[k + 1][1]) else k + 1
                runs[nb][1], runs[nb][2] = min(runs[nb][1], a), max(runs[nb][2], b)
                del runs[k]
                # merge equal-sign neighbours
                merged = []
                for r in sorted(runs, key=lambda r: r[1]):
                    if merged and merged[-1][0] == r[0]:
                        merged[-1][2] = r[2]
                    else:
                        merged.append(r)
                runs = merged
                changed = True
                break
    return [int((frames[runs[k][2]] + frames[runs[k + 1][1]]) // 2 + 1) for k in range(len(runs) - 1)]


def split_team_switches(tracks: pd.DataFrame, read_frame: Callable[[int], np.ndarray], model: Optional[Dict],
                        p: SplitParams = SplitParams()) -> pd.DataFrame:
    """Return tracks with cross-team identity swaps cut into separate track ids."""
    if model is None or "role" not in tracks:
        return tracks
    t = tracks.copy()
    players = t[(t.role == "player")]
    lens = players.groupby("track_id").size()
    cand = lens[lens >= p.min_track].index
    rows = players[players.track_id.isin(cand)].sort_values(["track_id", "frame"])
    rows = rows[rows.groupby("track_id").cumcount() % p.step == 0]
    sc = team_scores(read_frame, rows, model)
    next_id = int(t.track_id.max()) + 1
    for tid, g in sc.sort_values("frame").groupby("track_id"):
        if len(g) < 2 * p.min_run:
            continue
        cuts = split_points(g.frame.to_numpy(), g.score.to_numpy(), p)
        for c in cuts:
            sel = (t.track_id == tid) & (t.frame >= c)
            t.loc[sel, "track_id"] = next_id
            tid = next_id; next_id += 1
    return t
