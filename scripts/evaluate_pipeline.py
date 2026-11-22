"""
Score a full pipeline run (outputs/<seq>/) against SoccerNet ground truth.

Detection, tracking, roles, teams and ball are scored directly against GT.
Possession and passes have no human labels in SoccerNet-Tracking, so they are
scored against the *same analytics applied to the ground-truth tracks, teams and
ball* ("GT-derived") — this measures how much upstream errors distort them.
Physical stats are compared the same way (distance per matched player).

Usage:
    python scripts/evaluate_pipeline.py --runs runs/pipeline_test --seqs data/SoccerNet/tracking/train/SNMOT-062
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.analytics.match_analytics import (control_per_frame, possession_and_events, possession_share,  # noqa: E402
                                           smooth_tracks)
from src.data.soccernet import Sequence  # noqa: E402
from src.evaluation.metrics import (_matched_pairs, calibration_height_check, calibration_keeper_check,  # noqa: E402
                                    evaluate_ball,
                                    evaluate_roles, evaluate_teams, evaluate_tracking, gt_ball)
from src.tracking.stitching import foot_points_pitch  # noqa: E402


def gt_analytics(seq: Sequence, H: dict, team_map: dict, params=None):
    """Run the analytics on GT boxes/roles/teams/ball (team names mapped to the prediction's A/B)."""
    g = pd.DataFrame(seq.gt)
    p = g[g.role != "ball"].rename(columns={"id": "track_id"}).copy()
    p["team"] = p.team.map(team_map).fillna("")
    p["conf"] = 1.0
    p = foot_points_pitch(p, H)
    sm = smooth_tracks(p, seq.fps)
    b = gt_ball(seq)
    b["bx"] = np.nan; b["by"] = np.nan
    import cv2
    for i, r in b.iterrows():
        q = cv2.perspectiveTransform(np.array([[[r.x, r.y]]], float), H[int(r.frame)])[0, 0]
        b.loc[i, ["bx", "by"]] = q
    from src.analytics.match_analytics import EventParams
    params = params or EventParams()
    ctrl = control_per_frame(sm, b, params)
    poss, ev = possession_and_events(ctrl, b, seq.n_frames, params)
    return sm, poss, ev


def event_f1(pred: pd.DataFrame, gt: pd.DataFrame, kind: str, tol: int = 10):
    """Events match if same type and both start and end frames within ±tol."""
    P = pred[pred.event_type == kind][["start_frame", "end_frame"]].to_numpy()
    G = gt[gt.event_type == kind][["start_frame", "end_frame"]].to_numpy()
    used = set(); tp = 0
    for s, e in P:
        for j, (gs, ge) in enumerate(G):
            if j not in used and abs(s - gs) <= tol and abs(e - ge) <= tol:
                used.add(j); tp += 1; break
    prec = tp / max(len(P), 1); rec = tp / max(len(G), 1)
    return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / max(prec + rec, 1e-9), "n_pred": len(P), "n_gt": len(G)}


def evaluate(seq_dir: str, run_root: str, event_params=None) -> dict:
    seq = Sequence(seq_dir)
    d = os.path.join(run_root, seq.name)
    tr = pd.read_csv(os.path.join(d, "tracks.csv"))
    ball = pd.read_csv(os.path.join(d, "ball.csv"))
    poss = pd.read_csv(os.path.join(d, "possession.csv")).fillna({"possession": ""})
    ev = pd.read_csv(os.path.join(d, "events.csv"))
    cal = json.load(open(os.path.join(d, "calibration.json")))
    H = {i + 1: np.array(h) for i, h in enumerate(cal["H"])}
    tr["team"] = tr.team.fillna("")
    out = {"seq": seq.name}
    out.update({f"trk_{k}": v for k, v in evaluate_tracking(seq, tr).items()})
    out.update({f"role_{k}": v for k, v in evaluate_roles(seq, tr).items()})
    out.update(evaluate_teams(seq, tr))
    for tol in (10, 20):
        out.update({f"{k}@{tol}": v for k, v in evaluate_ball(seq, ball, tol).items() if k in ("ball_recall", "ball_precision")})
    out["calib_err_med_m"] = cal["quality"].get("err_med_m")
    out["calib_within_0.5m"] = cal["quality"].get("within_0.5m")
    # GT-based calibration check (person height / feet on pitch) — independent of the line detector
    out.update({f"calib_{k}": v for k, v in calibration_height_check(cal["H"], pd.DataFrame(seq.gt)).items()})
    out.update({f"calib_{k}": v for k, v in
                calibration_keeper_check(cal["H"], pd.DataFrame(seq.gt), cal.get("pitch", [105.0])[0]).items()})

    # map GT left/right to predicted A/B by majority over matched detections
    m = _matched_pairs(seq, tr)
    m = m[m.gt_team.isin(["left", "right"]) & m.pred_team.isin(["A", "B"])]
    agree = ((m.gt_team == "left") & (m.pred_team == "A")).sum() + ((m.gt_team == "right") & (m.pred_team == "B")).sum()
    team_map = {"left": "A", "right": "B"} if agree >= len(m) / 2 else {"left": "B", "right": "A"}
    gsm, gposs, gev = gt_analytics(seq, H, team_map, event_params)

    j = poss.merge(gposs[["frame", "possession"]].fillna(""), on="frame", suffixes=("", "_gt"))
    both = j.possession_gt.isin(["A", "B"])
    out["possession_frame_acc"] = float((j.possession[both] == j.possession_gt[both]).mean()) if both.any() else np.nan
    ps, gs = possession_share(poss), possession_share(gposs)
    out["possession_A_pred"] = ps["A"]; out["possession_A_gt"] = gs["A"]
    out["possession_abs_err_pts"] = abs(ps["A"] - gs["A"]) if not np.isnan(ps["A"]) and not np.isnan(gs["A"]) else np.nan
    for kind in ("Pass", "Turnover"):
        out.update({f"{kind.lower()}_{k}": v for k, v in event_f1(ev, gev, kind).items()})

    # distance covered: per GT player, compare to the predicted track(s) matched to it
    gdist = {}
    for gid, g in gsm.dropna(subset=["sx"]).groupby("track_id"):
        dsum = 0.0
        for _, s in g.groupby("segment"):
            dsum += float(np.hypot(np.diff(s.sx), np.diff(s.sy)).sum())
        gdist[gid] = dsum
    mm = _matched_pairs(seq, tr)
    errs = []
    pd_dist = {}
    for tid, g in tr.dropna(subset=["sx"]).groupby("track_id"):
        dsum = 0.0
        for _, s in g.groupby("segment"):
            dsum += float(np.hypot(np.diff(s.sx), np.diff(s.sy)).sum())
        pd_dist[tid] = dsum
    if len(mm):
        # assign each predicted track to the GT player it overlaps most; sum per GT player
        best = mm.groupby("pred_track_id").gt_id.agg(lambda s: s.value_counts().idxmax())
        per_gt = {}
        for tid, gid in best.items():
            per_gt[gid] = per_gt.get(gid, 0.0) + pd_dist.get(tid, 0.0)
        for gid, dist in per_gt.items():
            if gid in gdist and gdist[gid] > 10:
                errs.append(abs(dist - gdist[gid]) / gdist[gid])
    out["distance_rel_err_median"] = float(np.median(errs)) if errs else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = []
    for p in sorted({p for s in a.seqs for p in glob.glob(s)}):
        if os.path.exists(os.path.join(a.runs, os.path.basename(p), "tracks.csv")):
            rows.append(evaluate(p, a.runs))
            print(os.path.basename(p), "evaluated", flush=True)
    r = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(r.drop(columns=["seq"]).mean(numeric_only=True).round(4).to_string())
    if a.out:
        r.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
