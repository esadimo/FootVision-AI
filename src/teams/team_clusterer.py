"""
src.teams.team_clusterer — unsupervised two-team assignment at the *track* level.

Why the old classifier failed (SNMOT-062: 62 % of white-kit detections → wrong team)
-------------------------------------------------------------------------------------
It compared the *median LAB colour* of a torso crop with 4 K-Means centroids.
Lightness dominated the distance, so a white shirt in shadow (or with grass
leaking into the crop) landed closer to the mid-tone green centroid.  It also
forced 4 clusters (A, B, referee, GK) on every match and labelled them with
brittle rules (darkest = GK, yellowest = referee).

What this module does instead
-----------------------------
1. Roles (player / goalkeeper / referee) come from the fine-tuned detector, so
   only *outfield players* are clustered into two teams.
2. Per detection, a jersey crop (upper body, side-trimmed) is converted to HSV
   and grass pixels are removed.  Features are a normalised 2-D Hue×Saturation
   histogram plus a value(brightness) histogram — colour *distribution* rather
   than one median colour, and robust to shading.
3. Features are averaged per track (many frames → one robust descriptor) and
   2-means clustering is run on the tracks (weighted by track length).
4. Every detection gets its track's team.  Goalkeepers are assigned to the
   team whose players' mean pitch-x / image-x side they stand on.

Optionally `embedder` can supply CNN embeddings that are concatenated to the
colour features.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import cv2
import numpy as np
import pandas as pd

H_BINS, S_BINS, V_BINS = 18, 6, 6


def jersey_crop(frame: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> Optional[np.ndarray]:
    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(w_img, x2)), int(min(h_img, y2))
    w, h = x2 - x1, y2 - y1
    if w < 6 or h < 14:
        return None
    c = frame[y1 + int(0.12 * h): y1 + int(0.55 * h), x1 + int(0.15 * w): x2 - int(0.15 * w)]
    return c if c.size else None


def colour_feature(crop: np.ndarray) -> Optional[np.ndarray]:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    grass = (h >= 32) & (h <= 85) & (s >= 60) & (v >= 40)   # pitch green
    keep = ~grass
    if keep.sum() < 15:
        return None
    h, s, v = h[keep], s[keep], v[keep]
    hs, _, _ = np.histogram2d(h, s, bins=[H_BINS, S_BINS], range=[[0, 180], [0, 256]])
    vh, _ = np.histogram(v, bins=V_BINS, range=(0, 256))
    f = np.concatenate([hs.ravel(), vh.astype(np.float64)])
    f = f / max(f.sum(), 1)
    return np.sqrt(f)  # Hellinger mapping: Euclidean distance ≈ Hellinger distance between histograms


def extract_features(frames_iter, rows: pd.DataFrame, max_per_track: int = 60,
                     embedder: Optional[Callable] = None) -> pd.DataFrame:
    """
    frames_iter : callable(frame_number) -> BGR image
    rows        : DataFrame with frame, track_id, x1..y2 (people to describe)
    Samples up to `max_per_track` detections per track (evenly spaced, largest boxes preferred).
    """
    rows = rows.copy()
    rows["h"] = rows.y2 - rows.y1
    pick = []
    for tid, g in rows.groupby("track_id"):
        if len(g) > max_per_track:
            idx = np.linspace(0, len(g) - 1, max_per_track).astype(int)
            g = g.sort_values("frame").iloc[idx]
        pick.append(g)
    pick = pd.concat(pick).sort_values("frame")
    feats, keep_rows, crops = [], [], []
    for f, g in pick.groupby("frame"):
        img = frames_iter(int(f))
        for _, r in g.iterrows():
            c = jersey_crop(img, r.x1, r.y1, r.x2, r.y2)
            if c is None:
                continue
            fe = colour_feature(c)
            if fe is None:
                continue
            feats.append(fe); keep_rows.append(r); crops.append(c)
    out = pd.DataFrame(keep_rows).reset_index(drop=True)
    F = np.array(feats)
    if embedder is not None and len(crops):
        E = embedder(crops)
        E = E / np.linalg.norm(E, axis=1, keepdims=True)
        F = np.concatenate([F, E], axis=1)
    out["feat"] = list(F)
    return out


def _kmeans2(X: np.ndarray, w: np.ndarray, n_init: int = 10, seed: int = 0, k: int = 2):
    rng = np.random.default_rng(seed)
    best = None
    k = min(k, len(X))
    for _ in range(n_init):
        c = X[rng.choice(len(X), k, replace=False, p=w / w.sum())]
        for _ in range(100):
            d = ((X[:, None, :] - c[None]) ** 2).sum(-1)
            lab = d.argmin(1)
            newc = np.stack([np.average(X[lab == j], axis=0, weights=w[lab == j]) if (lab == j).any() else c[j]
                             for j in range(k)])
            if np.allclose(newc, c):
                break
            c = newc
        inertia = float((w * d[np.arange(len(X)), lab]).sum())
        if best is None or inertia < best[0]:
            best = (inertia, lab, c)
    return best[1], best[2]


def assign_teams(feat_rows: pd.DataFrame, player_tracks: Optional[set] = None,
                 min_samples: int = 3, merge_dist: float = 0.2, return_model: bool = False):
    """
    Cluster tracks into two teams.  Returns {track_id: {'team': 'A'|'B', 'margin': float}}
    (and, with return_model, also {'A': centroid, 'B': centroid, 'spread': float}).
    `player_tracks` restricts which tracks define the clusters (outfield players);
    every track in feat_rows is then assigned to the nearest team centroid.
    """
    agg = feat_rows.groupby("track_id").agg(n=("feat", "size"), feat=("feat", lambda s: np.mean(np.stack(s.values), 0)))
    fit = agg[agg.n >= min_samples]
    if player_tracks is not None:
        fit = fit[fit.index.isin(player_tracks)]
    if len(fit) < 2:
        return ({}, None) if return_model else {}
    X = np.stack(fit.feat.values)
    wts = fit.n.to_numpy(dtype=float)
    # Over-cluster (up to 4 kits: two teams + referees + goalkeepers/staff that slipped in as
    # "players"), merge near-identical clusters, and take the two heaviest clusters as the teams.
    # A plain 2-way split fails when officials are present: their kits are farther from both
    # teams than the teams are from each other, so 2-means separates officials vs players.
    k = min(4, len(X))
    lab, cent = _kmeans2(X, wts, k=k)
    groups = [[j] for j in range(k)]
    merged = True
    while merged and len(groups) > 2:
        merged = False
        gc = [np.average(X[np.isin(lab, g)], axis=0, weights=wts[np.isin(lab, g)]) if np.isin(lab, g).any() else None
              for g in groups]
        best_pair, best_d = None, merge_dist
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if gc[i] is None or gc[j] is None:
                    continue
                dd = float(np.linalg.norm(gc[i] - gc[j]))
                if dd < best_d:
                    best_pair, best_d = (i, j), dd
        if best_pair:
            i, j = best_pair
            groups[i] = groups[i] + groups[j]; del groups[j]; merged = True
    weight = [wts[np.isin(lab, g)].sum() for g in groups]
    order = np.argsort(weight)[::-1]
    team_groups = [groups[order[0]], groups[order[1]]] if len(groups) >= 2 else [groups[0], groups[0]]
    cmean = lambda g: np.average(X[np.isin(lab, g)], axis=0, weights=wts[np.isin(lab, g)])
    cent = np.stack([cmean(g) for g in team_groups])
    # remaining clusters: a split-off part of a team (closer to it than the two teams are to
    # each other) is merged back; a genuinely different kit (officials, staff) stays out
    sep = float(np.linalg.norm(cent[0] - cent[1]))
    for gi in order[2:]:
        g = groups[gi]
        if not np.isin(lab, g).any():
            continue
        dd = np.linalg.norm(cent - cmean(g), axis=1)
        if dd.min() < sep:
            team_groups[int(dd.argmin())] = team_groups[int(dd.argmin())] + g
    cent = np.stack([cmean(g) for g in team_groups])
    member = {tid: (0 if lab[i] in team_groups[0] else 1 if lab[i] in team_groups[1] else -1)
              for i, tid in enumerate(fit.index)}
    # outliers = fitted tracks in a non-team cluster; tracks too short to fit use nearest centroid
    # unless clearly far from both
    spread = np.median([np.linalg.norm(X[i] - cent[member[t]]) for i, t in enumerate(fit.index) if member[t] >= 0])
    outliers = {t for t, m in member.items() if m < 0}
    # canonical naming: team A = the cluster whose centroid is brighter (stable across runs)
    vb = slice(H_BINS * S_BINS, H_BINS * S_BINS + V_BINS)
    bright = [float((cent[k][vb] ** 2 * np.arange(V_BINS)).sum()) for k in range(2)]
    a_idx = int(np.argmax(bright))
    out = {}
    for tid, r in agg.iterrows():
        d = ((cent - r.feat) ** 2).sum(1)
        k = int(d.argmin())
        far = tid not in member and float(np.sqrt(d.min())) > 3 * spread
        out[int(tid)] = {"team": "" if (tid in outliers or far) else ("A" if k == a_idx else "B"),
                         "margin": float(abs(d[0] - d[1]) / max(d.sum(), 1e-9))}
    if return_model:
        return out, {"A": cent[a_idx], "B": cent[1 - a_idx], "spread": float(spread)}
    return out
