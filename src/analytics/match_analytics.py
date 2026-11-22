"""
src.analytics.match_analytics — position smoothing, ball control, possession,
passes and physical statistics built on calibrated pitch coordinates.

Fixes relative to the phase 11–13 analytics:
  * positions are smoothed with a constant-velocity Kalman filter + RTS smoother
    per *contiguous* track segment (gaps are never bridged by a straight line),
    so distance / speed are not inflated by jitter or teleports
  * ball control uses the image (ball centre inside the player's lower body box,
    robust to ball height) OR pitch distance ≤ control_m — a lofted ball
    projected to the ground is far from its true position
  * possession changes only after a player controls the ball for `min_ctrl`
    frames; frames before the first control are 'none' and not counted
  * a pass needs two consecutive control episodes by different players of the same
    team AND the ball to travel ≥ min_pass_m; a change of team is a turnover
  * top speed is a robust high percentile of the smoothed speed, not the max
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ── smoothing ─────────────────────────────────────────────────────────────────

def rts_smooth(z: np.ndarray, dt: float, q_acc: float = 4.0, r_pos: float = 0.35) -> np.ndarray:
    """Constant-velocity Kalman + Rauch–Tung–Striebel smoother for (N, 2) positions; NaNs allowed."""
    n = len(z)
    F = np.eye(4); F[0, 2] = F[1, 3] = dt
    G = np.array([[dt ** 2 / 2, 0], [0, dt ** 2 / 2], [dt, 0], [0, dt]])
    Q = G @ G.T * q_acc ** 2
    Hm = np.zeros((2, 4)); Hm[0, 0] = Hm[1, 1] = 1
    R = np.eye(2) * r_pos ** 2
    first = np.flatnonzero(~np.isnan(z[:, 0]))[0]
    x = np.r_[z[first], 0, 0]; P = np.diag([1, 1, 25, 25.0])
    xs_f, Ps_f, xs_p, Ps_p = np.zeros((n, 4)), np.zeros((n, 4, 4)), np.zeros((n, 4)), np.zeros((n, 4, 4))
    for k in range(n):
        if k > 0:
            x = F @ x; P = F @ P @ F.T + Q
        xs_p[k], Ps_p[k] = x, P
        if not np.isnan(z[k, 0]):
            S = Hm @ P @ Hm.T + R
            K = P @ Hm.T @ np.linalg.inv(S)
            x = x + K @ (z[k] - Hm @ x); P = (np.eye(4) - K @ Hm) @ P
        xs_f[k], Ps_f[k] = x, P
    xs = xs_f.copy()
    for k in range(n - 2, -1, -1):
        C = Ps_f[k] @ F.T @ np.linalg.inv(Ps_p[k + 1])
        xs[k] = xs_f[k] + C @ (xs[k + 1] - xs_p[k + 1])
    return xs  # x, y, vx, vy


def smooth_tracks(tracks: pd.DataFrame, fps: float = 25.0, max_gap: int = 5) -> pd.DataFrame:
    """Add sx, sy (smoothed metres) and vx, vy (m/s); segments split at gaps > max_gap frames."""
    out = []
    for tid, g in tracks.sort_values("frame").groupby("track_id"):
        g = g.copy()
        seg = (g.frame.diff().fillna(1) > max_gap).cumsum()
        for _, s in g.groupby(seg):
            full = pd.DataFrame({"frame": np.arange(s.frame.min(), s.frame.max() + 1)})
            m = full.merge(s[["frame", "pitch_x", "pitch_y"]], on="frame", how="left")
            z = m[["pitch_x", "pitch_y"]].to_numpy(float)
            if np.isnan(z[:, 0]).all():
                continue
            xs = rts_smooth(z, 1.0 / fps)
            m["sx"], m["sy"], m["vx"], m["vy"] = xs[:, 0], xs[:, 1], xs[:, 2], xs[:, 3]
            s = s.merge(m[["frame", "sx", "sy", "vx", "vy"]], on="frame", how="left")
            s["segment"] = f"{tid}_{int(s.frame.iloc[0])}"
            out.append(s)
    return pd.concat(out, ignore_index=True) if out else tracks.assign(sx=np.nan, sy=np.nan, vx=np.nan, vy=np.nan)


# ── ball control & possession ─────────────────────────────────────────────────

@dataclass
class EventParams:
    control_m: float = 1.5      # pitch distance ball(ground proj) ↔ feet
    box_margin: float = 0.25    # widen the box by this fraction of its width (image test)
    min_ctrl: int = 3           # frames of control before possession is credited
    min_pass_m: float = 3.0
    max_pass_frames: int = 125  # 5 s — longer gaps are not one pass
    max_hold: int = 50          # >0: possession lapses to none this many frames after the last control
                                # (loose / dead ball); 0 keeps it until the other team controls
    fps: float = 25.0


def control_per_frame(people: pd.DataFrame, ball: pd.DataFrame, p: EventParams = EventParams()) -> pd.DataFrame:
    """
    people: frame, track_id, team, role, x1..y2, sx, sy
    ball  : frame, x, y (px), bx, by (pitch metres, may be NaN)
    Returns per frame: controller track_id (or -1), team, method.
    """
    b = ball.set_index("frame")
    rows = []
    for f, g in people[people.role.isin(["player", "goalkeeper"]) & (people.team != "")].groupby("frame"):
        if f not in b.index or np.isnan(b.at[f, "x"]):
            continue
        bx, by = b.at[f, "x"], b.at[f, "y"]
        w = (g.x2 - g.x1).to_numpy()
        inside = ((bx >= g.x1 - p.box_margin * w) & (bx <= g.x2 + p.box_margin * w) &
                  (by >= g.y1 + 0.5 * (g.y2 - g.y1)) & (by <= g.y2 + 0.15 * (g.y2 - g.y1))).to_numpy()
        dm = np.full(len(g), np.inf)
        if "bx" in b.columns and not np.isnan(b.at[f, "bx"]):
            dm = np.hypot(g.sx.to_numpy() - b.at[f, "bx"], g.sy.to_numpy() - b.at[f, "by"])
        # image distance from ball to feet, normalised by box height (tie-breaker)
        dimg = np.hypot(((g.x1 + g.x2) / 2 - bx).to_numpy(), (g.y2 - by).to_numpy()) / np.maximum((g.y2 - g.y1).to_numpy(), 1)
        cand = inside | (dm <= p.control_m)
        if not cand.any():
            rows.append((f, -1, "", "")); continue
        k = int(np.argmin(np.where(cand, dimg, np.inf)))
        rows.append((f, int(g.track_id.iloc[k]), g.team.iloc[k], "image" if inside[k] else "pitch"))
    return pd.DataFrame(rows, columns=["frame", "controller", "ctrl_team", "method"])


def possession_and_events(ctrl: pd.DataFrame, ball: pd.DataFrame, n_frames: int,
                          p: EventParams = EventParams()):
    """Possession timeline + pass / turnover events from per-frame control."""
    c = pd.DataFrame({"frame": np.arange(1, n_frames + 1)}).merge(ctrl, on="frame", how="left")
    c["controller"] = c.controller.fillna(-1).astype(int); c["ctrl_team"] = c.ctrl_team.fillna("")
    # control episodes: runs of the same controller (allow 2-frame dropouts)
    episodes = []
    cur, start, last = -1, None, None
    for f, pid in zip(c.frame, c.controller):
        if pid == -1:
            if cur != -1 and f - last > 2:
                episodes.append((cur, start, last)); cur = -1
            continue
        if pid == cur:
            last = f
        else:
            if cur != -1:
                episodes.append((cur, start, last))
            cur, start, last = pid, f, f
    if cur != -1:
        episodes.append((cur, start, last))
    team_of = dict(zip(ctrl.controller, ctrl.ctrl_team))
    episodes = [(pid, s, e) for pid, s, e in episodes if e - s + 1 >= p.min_ctrl]

    poss = np.array([""] * n_frames, dtype=object)
    for i, (pid, s, e) in enumerate(episodes):
        end = episodes[i + 1][1] - 1 if i + 1 < len(episodes) else n_frames
        if p.max_hold > 0:
            end = min(end, e + p.max_hold)
        poss[s - 1:end] = team_of.get(pid, "")
    c["possession"] = poss

    bpos = ball.set_index("frame")
    events = []
    for (p1, s1, e1), (p2, s2, e2) in zip(episodes[:-1], episodes[1:]):
        if p1 == p2 or s2 - e1 > p.max_pass_frames:
            continue
        t1, t2 = team_of.get(p1, ""), team_of.get(p2, "")
        dist = np.nan
        if "bx" in bpos and e1 in bpos.index and s2 in bpos.index:
            dist = float(np.hypot(bpos.at[s2, "bx"] - bpos.at[e1, "bx"], bpos.at[s2, "by"] - bpos.at[e1, "by"]))
        if t1 == t2:
            if not np.isnan(dist) and dist < p.min_pass_m:
                continue          # same team, ball hardly moved: tracking swap / dribble, not a pass
            kind = "Pass"
        else:
            kind = "Turnover"
        events.append({"start_frame": e1, "end_frame": s2, "from_player": p1, "to_player": p2,
                       "from_team": t1, "to_team": t2, "event_type": kind,
                       "ball_travel_m": round(dist, 2) if not np.isnan(dist) else np.nan})
    return c, pd.DataFrame(events, columns=["start_frame", "end_frame", "from_player", "to_player",
                                            "from_team", "to_team", "event_type", "ball_travel_m"])


def possession_share(c: pd.DataFrame) -> Dict[str, float]:
    v = c.possession[c.possession.isin(["A", "B"])]
    if v.empty:
        return {"A": np.nan, "B": np.nan}
    return {"A": float((v == "A").mean() * 100), "B": float((v == "B").mean() * 100)}


# ── physical statistics ───────────────────────────────────────────────────────

def player_stats(sm: pd.DataFrame, events: pd.DataFrame, fps: float = 25.0, min_seconds: float = 1.0) -> pd.DataFrame:
    rows = []
    passes = events[events.event_type == "Pass"].from_player.value_counts() if len(events) else pd.Series(dtype=int)
    for tid, g in sm.groupby("track_id"):
        g = g.dropna(subset=["sx"])
        if len(g) < fps * min_seconds:
            continue
        dist = 0.0
        for _, s in g.groupby("segment"):
            dist += float(np.hypot(np.diff(s.sx), np.diff(s.sy)).sum())
        sp = np.hypot(g.vx, g.vy)
        t = len(g) / fps
        rows.append({"track_id": int(tid), "role": g.role.iloc[0], "team_label": g.team.iloc[0],
                     "time_visible_s": round(t, 2), "distance_covered_m": round(dist, 2),
                     "avg_speed_kmh": round(dist / t * 3.6, 2),
                     "max_speed_kmh": round(float(np.percentile(sp, 98)) * 3.6, 2),
                     "passes_made": int(passes.get(tid, 0))})
    return pd.DataFrame(rows).sort_values("distance_covered_m", ascending=False) if rows else pd.DataFrame()


def team_shape(sm: pd.DataFrame, L: float = 105.0, W: float = 68.0) -> pd.DataFrame:
    """Per-frame width / depth / centroid of each team's outfield players (on-pitch, ≥3 players)."""
    p = sm[(sm.role == "player") & sm.team.isin(["A", "B"])].dropna(subset=["sx"])
    p = p[(p.sx >= -1) & (p.sx <= L + 1) & (p.sy >= -1) & (p.sy <= W + 1)]
    rows = []
    for f, g in p.groupby("frame"):
        r = {"frame_number": int(f)}
        for tm in ("A", "B"):
            q = g[g.team == tm]
            if len(q) >= 3:
                r[f"Team {tm}_width_m"] = round(q.sy.max() - q.sy.min(), 2)
                r[f"Team {tm}_depth_m"] = round(q.sx.max() - q.sx.min(), 2)
                r[f"Team {tm}_centroid_x"] = round(q.sx.mean(), 2)
            else:
                r[f"Team {tm}_width_m"] = np.nan; r[f"Team {tm}_depth_m"] = np.nan; r[f"Team {tm}_centroid_x"] = np.nan
        rows.append(r)
    return pd.DataFrame(rows)
