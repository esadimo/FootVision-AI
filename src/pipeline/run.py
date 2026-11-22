"""
src.pipeline.run — the accurate single-pass FootVision pipeline for one sequence.

    detect (cached)  →  track (BoT-SORT)  →  calibrate (PTZ camera, automatic)
      →  pitch coordinates  →  roles & teams  →  tracklet stitching  →  roles & teams
      →  ball trajectory (offline DP)  →  smoothing  →  control / possession / passes
      →  player & team statistics  →  outputs (+ legacy phase-file names)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import cv2
import numpy as np
import pandas as pd

from src.analytics.match_analytics import (EventParams, control_per_frame, player_stats, possession_and_events,
                                           possession_share, smooth_tracks, team_shape)
from src.calibration.camera_model import (_cues_for, _fit_frame, calibrate_sequence, pitch_error_m,
                                          repair_calibration)
from src.data.soccernet import Sequence
from src.pipeline.detect_track import BALL_CLASS, PERSON_CLASSES, dets_to_frame, load_or_detect, track_cached
from src.pipeline.roles_teams import assign_roles_and_teams
from src.teams.team_clusterer import extract_features
from src.teams.track_splitter import SplitParams, split_team_switches
from src.tracking.ball_trajectory import BallTrajectoryParams, solve_ball_trajectory
from src.tracking.ball_zoom import load_or_zoom
from src.tracking.stitching import StitchParams, foot_points_pitch, stitch_tracklets


@dataclass
class PipelineConfig:
    model: str = "models/detection/fv_yolo26s_1280.pt"
    tag: str = "ft1280"
    imgsz: int = 1280
    det_conf: float = 0.05
    device: str = "mps"
    cache_dir: str = "data/cache"
    tracker: Dict = field(default_factory=lambda: {
        "tracker_type": "botsort", "track_high_thresh": 0.35, "track_low_thresh": 0.1, "new_track_thresh": 0.25,
        "track_buffer": 30, "match_thresh": 0.9, "fuse_score": True, "gmc_method": "sparseOptFlow",
        "proximity_thresh": 0.5, "appearance_thresh": 0.8, "with_reid": False, "model": "auto"})
    mask_conf: float = 0.3            # detections used to mask people during calibration
    H_init: Optional[str] = None      # optional manual first-frame homography (else fully automatic)
    stitch: StitchParams = field(default_factory=StitchParams)
    ball: BallTrajectoryParams = field(default_factory=BallTrajectoryParams)
    events: EventParams = field(default_factory=EventParams)
    split_teams: bool = True          # cut tracks whose kit flips team (identity swaps between opponents)
    split: SplitParams = field(default_factory=SplitParams)
    ball_zoom: bool = True            # second detection pass on a crop around the expected ball
    calib_cache: bool = True
    calib_repair: bool = True         # self-check + repair of drifted frames (also upgrades cached calibrations)


def _frame_reader(seq: Sequence):
    cache = {}

    def read(f: int) -> np.ndarray:
        if f not in cache:
            if len(cache) > 4:
                cache.clear()
            cache[f] = cv2.imread(seq.frame_paths[f - 1])
        return cache[f]
    return read


def calibrate(seq: Sequence, dets: Dict, cfg: PipelineConfig, out_dir: str, log=print):
    path = os.path.join(out_dir, "calibration.json")
    df = dets_to_frame(dets)
    m = df[df.cls.isin(PERSON_CLASSES) & (df.conf >= cfg.mask_conf)]
    boxes = {int(f): g[["x1", "y1", "x2", "y2"]].to_numpy(float) for f, g in m.groupby("frame")}
    if cfg.calib_cache and os.path.exists(path):
        d = json.load(open(path))
        if cfg.calib_repair and not d.get("repaired"):
            # calibration cached before the repair stage existed: repair it in place
            C = np.asarray(d["camera_centre"], float)
            L, W = d.get("pitch", [105.0, 68.0])
            cues = [_cues_for(cv2.imread(p), boxes.get(i + 1)) for i, p in enumerate(seq.frame_paths)]
            cams = [_fit_frame(np.array(h), C, cues[i], L, W) for i, h in enumerate(d["H"])]
            cams, infos = repair_calibration(seq.frame_paths, cams, [{} for _ in cams], boxes, cues, C, verbose=False)
            d["H"] = [c.H().tolist() for c in cams]
            d["focal"] = [c.f for c in cams]
            d["self_check"] = [x.get("self_check") for x in infos]
            d["repaired_frames"] = int(sum(1 for x in infos if x.get("repaired")))
            d["repaired"] = True
            json.dump(d, open(path, "w"))
            log(f"  calibration repaired: {d['repaired_frames']} frames replaced")
        return {i + 1: np.array(h) for i, h in enumerate(d["H"])}, d
    H0 = np.array(json.load(open(cfg.H_init))["keyframes"][0]["H"]) if cfg.H_init else None
    cams, infos = calibrate_sequence(seq.frame_paths, H0, boxes, verbose=True, repair=cfg.calib_repair)
    # quality report on a sample of frames
    errs = []
    for i in range(0, seq.n_frames, 25):
        cue = _cues_for(cv2.imread(seq.frame_paths[i]), boxes.get(i + 1))
        errs.append(pitch_error_m(cams[i], *cue))
    q = pd.DataFrame(errs).median().to_dict()
    d = {"H": [c.H().tolist() for c in cams], "camera_centre": cams[0].C.tolist(),
         "pitch": [cams[0].L, cams[0].W], "focal": [c.f for c in cams],
         "line_in3": [x.get("in3") for x in infos], "quality": q,
         "self_check": [x.get("self_check") for x in infos],
         "repaired_frames": int(sum(1 for x in infos if x.get("repaired"))), "repaired": cfg.calib_repair}
    json.dump(d, open(path, "w"))
    log(f"  calibration: median line error {q.get('err_med_m', float('nan')):.3f} m, "
        f"{100 * q.get('within_0.5m', float('nan')):.1f}% within 0.5 m")
    return {i + 1: np.array(h) for i, h in enumerate(d["H"])}, d


def calibration_confidence(calib: Dict, bad: float = 0.5, low_share: float = 0.2) -> Dict:
    """Ground-truth-free calibration confidence from the per-frame people-size self-check."""
    sc = np.array([np.nan if v is None else v for v in calib.get("self_check", [])], float)
    if not len(sc) or np.isnan(sc).all():
        return {"calibration_self_check": None, "calibration_flagged_share": None, "calibration_low_confidence": None}
    flagged = float(np.mean(np.nan_to_num(sc, nan=1.0) < bad))
    return {"calibration_self_check": round(float(np.nanmean(sc)), 3), "calibration_flagged_share": round(flagged, 3),
            "calibration_low_confidence": flagged > low_share}


def run_sequence(seq_dir: str, out_root: str = "outputs", cfg: PipelineConfig = PipelineConfig(),
                 legacy: bool = True, log=print) -> Dict:
    seq = Sequence(seq_dir)
    out_dir = os.path.join(out_root, seq.name)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    fps = seq.fps
    read = _frame_reader(seq)

    log(f"[{seq.name}] detect")
    dets = load_or_detect(seq, cfg.model, cfg.cache_dir, cfg.tag, imgsz=cfg.imgsz, conf=cfg.det_conf, device=cfg.device)

    log(f"[{seq.name}] track")
    tracks = track_cached(seq, dets, cfg.tracker)

    log(f"[{seq.name}] calibrate")
    H, calib = calibrate(seq, dets, cfg, out_dir, log)
    L, W = calib["pitch"]

    tracks = foot_points_pitch(tracks, H)
    log(f"[{seq.name}] roles/teams + stitching")
    if cfg.split_teams:
        first, model = assign_roles_and_teams(tracks, read, pitch_len=L, pitch_wid=W, return_model=True)
        n0 = tracks.track_id.nunique()
        tracks = split_team_switches(first, read, model, cfg.split)[tracks.columns]
        log(f"  team-switch split: {n0} → {tracks.track_id.nunique()} tracks")
    feats = extract_features(read, tracks, max_per_track=30)
    app = {int(k): v / np.linalg.norm(v) for k, v in
           feats.groupby("track_id").feat.apply(lambda s: np.mean(np.stack(s.values), 0)).items()}
    first = assign_roles_and_teams(tracks, read, pitch_len=L, pitch_wid=W)
    lab = first.groupby("track_id").apply(lambda g: f"{g.role.iloc[0]}:{g.team.iloc[0]}").to_dict()
    tracks = stitch_tracklets(tracks, H, labels=lab, appearance=app, p=cfg.stitch)
    tracks = foot_points_pitch(tracks.drop(columns=["pitch_x", "pitch_y"], errors="ignore"), H)
    tracks = assign_roles_and_teams(tracks, read, pitch_len=L, pitch_wid=W)

    log(f"[{seq.name}] ball")
    dfd = dets_to_frame(dets)
    bc = dfd[dfd.cls == BALL_CLASS].copy()
    bc["x"] = (bc.x1 + bc.x2) / 2; bc["y"] = (bc.y1 + bc.y2) / 2
    ball = solve_ball_trajectory(bc, seq.n_frames, cfg.ball)
    if cfg.ball_zoom:
        zc = load_or_zoom(seq, ball, cfg.model, cfg.cache_dir, cfg.tag, device=cfg.device).assign(cls=BALL_CLASS)
        ball = solve_ball_trajectory(pd.concat([bc, zc[bc.columns]], ignore_index=True), seq.n_frames, cfg.ball)
    ball["bx"] = np.nan; ball["by"] = np.nan
    ok = ball.x.notna()
    for i in np.flatnonzero(ok):
        f = int(ball.frame.iloc[i])
        q = cv2.perspectiveTransform(np.array([[[ball.x.iloc[i], ball.y.iloc[i]]]], float), H[f])[0, 0]
        ball.loc[ball.index[i], ["bx", "by"]] = q

    log(f"[{seq.name}] analytics")
    sm = smooth_tracks(tracks, fps)
    ctrl = control_per_frame(sm, ball, cfg.events)
    poss, events = possession_and_events(ctrl, ball, seq.n_frames, cfg.events)
    stats = player_stats(sm, events, fps)
    shape = team_shape(sm, L, W)

    sm.to_csv(os.path.join(out_dir, "tracks.csv"), index=False)
    ball.to_csv(os.path.join(out_dir, "ball.csv"), index=False)
    poss.to_csv(os.path.join(out_dir, "possession.csv"), index=False)
    events.to_csv(os.path.join(out_dir, "events.csv"), index=False)
    stats.to_csv(os.path.join(out_dir, "player_stats.csv"), index=False)
    shape.to_csv(os.path.join(out_dir, "team_metrics.csv"), index=False)
    share = possession_share(poss)
    summary = {"sequence": seq.name, "seconds": round(time.time() - t0, 1), "possession_pct": share,
               "n_tracks": int(sm.track_id.nunique()), "n_passes": int((events.event_type == "Pass").sum()) if len(events) else 0,
               "n_turnovers": int((events.event_type == "Turnover").sum()) if len(events) else 0,
               "ball_frames_known": int(ball.x.notna().sum()), "calibration_quality": calib.get("quality"),
               **calibration_confidence(calib),
               "camera_centre": calib.get("camera_centre"), "pitch_dims": calib.get("pitch")}
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=2)
    if legacy:
        write_legacy(seq, out_root, sm, ball, poss, events, stats, shape, H, fps)
        write_kit_profile(seq, out_root, sm, read)
    log(f"[{seq.name}] done in {summary['seconds']}s — possession A {share['A']:.1f}% / B {share['B']:.1f}%")
    return summary


def write_legacy(seq, out_root, sm, ball, poss, events, stats, shape, H, fps):
    """Files with the phase-script names so app.py and phase14_ultimate_video.py keep working."""
    n = seq.name
    lab = sm.apply(lambda r: ("GK " if r.role == "goalkeeper" else "") + (f"Team {r.team}" if r.team else
                   ("Referee" if r.role == "referee" else "Staff/GK")), axis=1)
    pd.DataFrame({"frame_number": sm.frame, "timestamp_s": ((sm.frame - 1) / fps).round(4), "track_id": sm.track_id,
                  "team_label": lab, "bbox_x1": sm.x1.round().astype(int), "bbox_y1": sm.y1.round().astype(int),
                  "bbox_x2": sm.x2.round().astype(int), "bbox_y2": sm.y2.round().astype(int),
                  "foot_x_px": ((sm.x1 + sm.x2) / 2).round(1), "foot_y_px": sm.y2.round(1),
                  "pitch_x_m": sm.sx.fillna(sm.pitch_x).round(4), "pitch_y_m": sm.sy.fillna(sm.pitch_y).round(4)}
                 ).sort_values(["frame_number", "track_id"]).to_csv(os.path.join(out_root, f"{n}_phase9_pitch_coords.csv"), index=False)
    pd.DataFrame({"frame_number": ball.frame, "timestamp_s": ((ball.frame - 1) / fps).round(4),
                  "ball_detected": ball.x.notna().astype(int), "x_center": ball.x.fillna(-1).round(2),
                  "y_center": ball.y.fillna(-1).round(2), "is_interpolated": (ball.source == "interp").astype(int),
                  "confidence": ball.conf.fillna(0).round(3)}).to_csv(os.path.join(out_root, f"{n}_phase10_ball.csv"), index=False)
    pa = poss.copy()
    cum = pa.possession.isin(["A", "B"]).cumsum().replace(0, np.nan)
    pa["team_A_pct"] = ((pa.possession == "A").cumsum() / cum * 100).fillna(0).round(2)
    pa["team_B_pct"] = ((pa.possession == "B").cumsum() / cum * 100).fillna(0).round(2)
    pd.DataFrame({"frame_number": pa.frame, "timestamp_s": ((pa.frame - 1) / fps).round(4),
                  "possession_team": pa.possession.map({"A": "Team A", "B": "Team B"}).fillna("None"),
                  "closest_player_id": pa.controller, "closest_distance_m": np.where(pa.controller >= 0, 0.0, -1),
                  "team_A_pct": pa.team_A_pct, "team_B_pct": pa.team_B_pct}).to_csv(
        os.path.join(out_root, f"{n}_phase11_possession.csv"), index=False)
    ev = events.copy()
    ev["from_team"] = "Team " + ev.from_team.astype(str); ev["to_team"] = "Team " + ev.to_team.astype(str)
    ev.to_csv(os.path.join(out_root, f"{n}_phase12_events.csv"), index=False)
    st = stats.copy()
    if len(st):
        st["team_label"] = st.team_label.map(lambda t: f"Team {t}" if t else "Other")
    st.to_csv(os.path.join(out_root, f"{n}_phase13_player_stats.csv"), index=False)
    shape.to_csv(os.path.join(out_root, f"{n}_phase13_team_metrics.csv"), index=False)
    json.dump({"keyframes": [{"frame_idx": f - 1, "H": np.asarray(h).tolist()} for f, h in sorted(H.items())]},
              open(os.path.join(out_root, f"{n}_homography.json"), "w"))


def write_kit_profile(seq, out_root, sm, read, per_team: int = 60):
    """Display colours per team = median jersey colour (grass removed) — read by the phase-14 video."""
    from src.teams.team_clusterer import jersey_crop
    colours = {"Referee": [0, 215, 255], "Staff/GK": [60, 60, 60], "Unknown": [140, 140, 140]}
    for tm in ("A", "B"):
        rows = sm[(sm.team == tm) & (sm.role == "player")]
        if rows.empty:
            continue
        rows = rows.sample(min(per_team, len(rows)), random_state=0)
        px = []
        for _, r in rows.iterrows():
            c = jersey_crop(read(int(r.frame)), r.x1, r.y1, r.x2, r.y2)
            if c is None:
                continue
            hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV).reshape(-1, 3)
            grass = (hsv[:, 0] >= 32) & (hsv[:, 0] <= 85) & (hsv[:, 1] >= 60)
            px.append(c.reshape(-1, 3)[~grass])
        if px:
            colours[f"Team {tm}"] = [int(v) for v in np.median(np.concatenate(px), axis=0)]
            colours[f"GK Team {tm}"] = [0, 140, 255] if tm == "A" else [255, 200, 0]
    json.dump({"is_fitted": True, "team_colors_bgr": colours, "source": "src.pipeline.run"},
              open(os.path.join(out_root, f"{seq.name}_kit_profile.json"), "w"), indent=2)
