"""
src.pipeline.roles_teams — role (player / goalkeeper / referee) and team per track.

Roles: confidence-weighted vote of the fine-tuned detector's class over the whole
track (a referee mis-detected as a player for a few frames does not flip).

Teams: outfield-player tracks are clustered into two kits (src.teams.team_clusterer).
Goalkeepers wear a third kit, so they cannot be clustered with their team; each
goalkeeper is assigned to the team whose players' median pitch-x is on the same
side of the halfway line as him (the team he defends for).

Goalkeepers: the detector rarely predicts the goalkeeper class (≈7 % recall on
SoccerNet game 4), but a keeper is easy to recognise from context — his kit
matches neither team (team clustering leaves him unassigned) and he stays near
his own goal.  `refine_goalkeepers` promotes such tracks, at most one keeper at a
time per goal.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from src.teams.team_clusterer import assign_teams, extract_features

ROLE_NAMES = {0: "player", 1: "goalkeeper", 2: "referee"}


def track_roles(tracks: pd.DataFrame) -> Dict[int, str]:
    out = {}
    for tid, g in tracks.groupby("track_id"):
        w = g.groupby("cls").conf.sum()
        out[int(tid)] = ROLE_NAMES.get(int(w.idxmax()), "player")
    return out


def refine_goalkeepers(tracks: pd.DataFrame, roles: Dict[int, str], team_of: Dict[int, str],
                      pitch_len: float = 105.0, pitch_wid: float = 68.0, goal_zone_m: float = 25.0,
                      min_frames: int = 10, max_overlap: int = 5, min_gk_share: float = 0.3,
                      kit_dist: Optional[Dict[int, float]] = None, kit_outlier: float = 2.0) -> Dict[int, str]:
    """
    Promote 'player' tracks to goalkeeper when they (a) wear neither team's kit (no team
    assigned, or kit_dist — distance to the nearest team kit centroid in units of the teams'
    typical spread — ≥ kit_outlier) or were often detected as goalkeeper, and (b) stay within
    `goal_zone_m` of a goal line, inside the pitch.  On game 4, keepers sit at ≥ 2.8 spreads
    and outfield players near goal at ≤ 1.0, while team clustering only leaves a track
    unassigned beyond 3 spreads.  Per goal, candidates are taken greedily by evidence and
    must not overlap in time with an already chosen keeper (one keeper per goal at a time).
    Returns the updated roles dict.
    """
    if "pitch_x" not in tracks or tracks.pitch_x.isna().all():
        return roles
    roles = dict(roles)
    cand = {0: [], 1: []}   # side 0 = goal at x=0, side 1 = goal at x=L
    for tid, g in tracks.groupby("track_id"):
        tid = int(tid)
        if roles.get(tid) != "player" or len(g) < min_frames:
            continue
        mx, my = float(np.nanmedian(g.pitch_x)), float(np.nanmedian(g.pitch_y)) if "pitch_y" in g else pitch_wid / 2
        if np.isnan(mx) or not (-2 <= mx <= pitch_len + 2) or abs(my - pitch_wid / 2) > 25:
            continue
        side = 0 if mx < pitch_len / 2 else 1
        if min(mx, pitch_len - mx) > goal_zone_m:
            continue
        w = g.groupby("cls").conf.sum()
        gk_share = float(w.get(1, 0.0) / max(w.sum(), 1e-9))
        no_team = team_of.get(tid, "") == "" or (kit_dist or {}).get(tid, 0.0) >= kit_outlier
        if not (no_team or gk_share >= min_gk_share):
            continue
        score = len(g) * (no_team + gk_share)
        cand[side].append((score, tid, int(g.frame.min()), int(g.frame.max())))
    # Detector-voted goalkeeper tracks are kept but do not block a slot: they are often short
    # fragments of the same keeper whose main track was voted 'player'.
    taken = {0: [], 1: []}
    for side, cs in cand.items():
        for _, tid, a, b in sorted(cs, reverse=True):
            if all(min(b, e) - max(a, s) + 1 <= max_overlap for s, e in taken[side]):
                roles[tid] = "goalkeeper"
                taken[side].append((a, b))
    return roles


def goalkeeper_teams(tracks: pd.DataFrame, roles: Dict[int, str], team_of: Dict[int, str],
                     pitch_len: float = 105.0) -> Dict[int, str]:
    """
    Each goalkeeper gets the team that defends his goal: the team whose outfield players'
    positions (all detections, so long tracks weigh more than fragments) have the lower
    median pitch-x defends the goal at x = 0.
    """
    team_of = dict(team_of)
    if "pitch_x" not in tracks:
        return team_of
    t = tracks[["track_id", "pitch_x"]].dropna()
    tm = t.track_id.map(lambda x: team_of.get(int(x), "") if roles.get(int(x)) == "player" else "")
    side = {k: float(t.pitch_x[tm == k].median()) for k in ("A", "B") if (tm == k).any()}
    if len(side) < 2:
        return team_of
    left, right = min(side, key=side.get), max(side, key=side.get)
    for tid, g in t.groupby("track_id"):
        if roles.get(int(tid)) == "goalkeeper":
            team_of[int(tid)] = left if g.pitch_x.median() < pitch_len / 2 else right
    return team_of


def assign_roles_and_teams(tracks: pd.DataFrame, read_frame: Callable[[int], np.ndarray],
                           pitch_len: float = 105.0, max_per_track: int = 40,
                           pitch_wid: float = 68.0, return_model: bool = False):
    """
    tracks: frame, track_id, x1..y2, conf, cls (+ pitch_x for goalkeeper side assignment).
    Returns tracks with added columns role, team ('A'/'B'/'' ) and team_margin.
    """
    roles = track_roles(tracks)
    feats = extract_features(read_frame, tracks, max_per_track=max_per_track)
    players = {t for t, r in roles.items() if r == "player"}
    teams, model = assign_teams(feats, player_tracks=players, return_model=True)
    team_of = {t: v["team"] for t, v in teams.items() if roles.get(t) == "player"}
    margin = {t: v["margin"] for t, v in teams.items()}
    kit_dist = None
    if model is not None and len(feats):
        mf = feats.groupby("track_id").feat.apply(lambda v: np.mean(np.stack(v.values), 0))
        kit_dist = {int(t): float(min(np.linalg.norm(f - model["A"]), np.linalg.norm(f - model["B"]))
                                  / max(model["spread"], 1e-9)) for t, f in mf.items()}
    roles = refine_goalkeepers(tracks, roles, team_of, pitch_len, pitch_wid, kit_dist=kit_dist)
    team_of = {t: v for t, v in team_of.items() if roles.get(t) == "player"}

    team_of = goalkeeper_teams(tracks, roles, team_of, pitch_len)
    out = tracks.copy()
    out["role"] = out.track_id.map(roles).fillna("player")
    out["team"] = out.track_id.map(team_of).fillna("")
    out["team_margin"] = out.track_id.map(margin)
    return (out, model) if return_model else out
