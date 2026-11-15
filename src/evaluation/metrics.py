"""
src.evaluation.metrics — Ground-truth evaluation for every pipeline stage.

All functions take pandas DataFrames so that any stage's output can be scored
against a SoccerNet `Sequence`:

    detections / tracks : columns frame, x1, y1, x2, y2, conf
                          (+ track_id for tracking, + role / team when present)
    ball                : columns frame, x, y   (one row per frame with a ball estimate)

Metrics
-------
evaluate_detection   precision / recall / F1 at IoU 0.5 for people, AP50 over confidences
evaluate_tracking    HOTA, DetA, AssA, MOTA, IDF1, ID switches (TrackEval implementation)
evaluate_roles       accuracy of player / goalkeeper / referee classification
evaluate_teams       team accuracy (best of the two left/right ↔ A/B mappings)
evaluate_ball        per-frame ball precision / recall at a pixel tolerance, mean error
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from src.data.soccernet import Sequence

PERSON_ROLES = ("player", "goalkeeper", "referee", "other")


# ── geometry helpers ──────────────────────────────────────────────────────────

def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between boxes a (N,4) and b (M,4) in x1,y1,x2,y2."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0).astype(np.float32)


def _match(gt_boxes: np.ndarray, pr_boxes: np.ndarray, thr: float = 0.5):
    """Hungarian matching on IoU; returns list of (gt_idx, pr_idx) with IoU ≥ thr."""
    iou = iou_matrix(gt_boxes, pr_boxes)
    if iou.size == 0:
        return [], iou
    r, c = linear_sum_assignment(-iou)
    keep = iou[r, c] >= thr
    return list(zip(r[keep], c[keep])), iou


def _gt_people(seq: Sequence) -> pd.DataFrame:
    g = pd.DataFrame(seq.gt)
    return g[g.role != "ball"].reset_index(drop=True)


def _boxes(df: pd.DataFrame) -> np.ndarray:
    return df[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float32)


# ── detection ─────────────────────────────────────────────────────────────────

def evaluate_detection(seq: Sequence, det: pd.DataFrame, conf_thr: float = 0.0,
                       iou_thr: float = 0.5, frames: Optional[set] = None) -> Dict[str, float]:
    """Class-agnostic person detection quality (ball rows in `det` must be excluded by caller)."""
    gt = _gt_people(seq)
    if frames is not None:
        gt = gt[gt.frame.isin(frames)]
        det = det[det.frame.isin(frames)]
    det = det[det.conf >= conf_thr]
    g_by = {k: v for k, v in gt.groupby("frame")}
    d_by = {k: v for k, v in det.groupby("frame")}

    tp = fp = fn = 0
    scores, is_tp = [], []
    for f in sorted(set(g_by) | set(d_by)):
        g = g_by.get(f, gt.iloc[:0]); d = d_by.get(f, det.iloc[:0])
        m, _ = _match(_boxes(g), _boxes(d), iou_thr)
        tp += len(m); fp += len(d) - len(m); fn += len(g) - len(m)
        matched = {p for _, p in m}
        for j, s in enumerate(d.conf.to_numpy()):
            scores.append(s); is_tp.append(j in matched)

    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    # AP50 (all-point interpolation)
    order = np.argsort(-np.asarray(scores))
    hits = np.asarray(is_tp, dtype=float)[order]
    ctp = np.cumsum(hits); cfp = np.cumsum(1 - hits)
    n_gt = len(gt)
    rc = ctp / max(n_gt, 1); pc = ctp / np.maximum(ctp + cfp, 1e-9)
    mpre = np.concatenate([[0], pc, [0]]); mrec = np.concatenate([[0], rc, [rc[-1] if len(rc) else 0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = float(np.sum((mrec[1:] - mrec[:-1]) * mpre[1:]))
    return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / max(prec + rec, 1e-9),
            "AP50": ap, "tp": tp, "fp": fp, "fn": fn}


# ── tracking (TrackEval) ──────────────────────────────────────────────────────

def evaluate_tracking(seq: Sequence, trk: pd.DataFrame) -> Dict[str, float]:
    """HOTA / CLEAR / Identity metrics for people tracking using TrackEval's implementations."""
    from trackeval.metrics import HOTA, CLEAR, Identity

    gt = _gt_people(seq)
    n_t = seq.n_frames
    gt_ids_map = {v: i for i, v in enumerate(sorted(gt.id.unique()))}
    tr_ids_map = {v: i for i, v in enumerate(sorted(trk.track_id.unique()))}
    g_by = {k: v for k, v in gt.groupby("frame")}
    t_by = {k: v for k, v in trk.groupby("frame")}

    data = {"num_timesteps": n_t, "num_gt_ids": len(gt_ids_map), "num_tracker_ids": len(tr_ids_map),
            "num_gt_dets": len(gt), "num_tracker_dets": len(trk),
            "gt_ids": [], "tracker_ids": [], "similarity_scores": []}
    for f in range(1, n_t + 1):
        g = g_by.get(f, gt.iloc[:0]); t = t_by.get(f, trk.iloc[:0])
        data["gt_ids"].append(np.array([gt_ids_map[i] for i in g.id], dtype=int))
        data["tracker_ids"].append(np.array([tr_ids_map[i] for i in t.track_id], dtype=int))
        data["similarity_scores"].append(iou_matrix(_boxes(g), _boxes(t)))

    out = {}
    cfgq = {"PRINT_CONFIG": False}
    h = HOTA(cfgq).eval_sequence(data)
    out["HOTA"] = float(np.mean(h["HOTA"])); out["DetA"] = float(np.mean(h["DetA"]))
    out["AssA"] = float(np.mean(h["AssA"]))
    c = CLEAR(cfgq).eval_sequence(data)
    out["MOTA"] = float(c["MOTA"]); out["IDSW"] = int(c["IDSW"]); out["Frag"] = int(c["Frag"])
    i = Identity(cfgq).eval_sequence(data)
    out["IDF1"] = float(i["IDF1"])
    out["n_pred_ids"] = len(tr_ids_map); out["n_gt_ids"] = len(gt_ids_map)
    return out


# ── role & team classification ────────────────────────────────────────────────

def _matched_pairs(seq: Sequence, trk: pd.DataFrame, iou_thr: float = 0.5) -> pd.DataFrame:
    """Per-frame matches between GT people and predicted rows; returns joined rows."""
    gt = _gt_people(seq)
    g_by = {k: v for k, v in gt.groupby("frame")}
    rows = []
    for f, t in trk.groupby("frame"):
        g = g_by.get(f)
        if g is None:
            continue
        m, _ = _match(_boxes(g), _boxes(t), iou_thr)
        for gi, ti in m:
            gr = g.iloc[gi]; tr = t.iloc[ti]
            rows.append({"frame": f, "gt_id": gr.id, "gt_role": gr.role, "gt_team": gr.team,
                         **{f"pred_{k}": tr[k] for k in ("track_id", "role", "team") if k in t.columns}})
    return pd.DataFrame(rows)


def evaluate_roles(seq: Sequence, trk: pd.DataFrame) -> Dict[str, float]:
    """Accuracy of predicted role ∈ {player, goalkeeper, referee} on matched GT people."""
    m = _matched_pairs(seq, trk)
    m = m[m.gt_role.isin(["player", "goalkeeper", "referee"])]
    if m.empty:
        return {}
    out = {"role_acc": float((m.gt_role == m.pred_role).mean())}
    for r in ("player", "goalkeeper", "referee"):
        s = m[m.gt_role == r]
        if len(s):
            out[f"{r}_recall"] = float((s.pred_role == r).mean())
        p = m[m.pred_role == r]
        if len(p):
            out[f"{r}_precision"] = float((p.gt_role == r).mean())
    return out


def evaluate_teams(seq: Sequence, trk: pd.DataFrame) -> Dict[str, float]:
    """
    Team accuracy over matched GT players/goalkeepers that have a team.
    Predicted team values are 'A' / 'B' (anything else counts as wrong);
    the better of the two possible A/B ↔ left/right mappings is used.
    """
    m = _matched_pairs(seq, trk)
    m = m[m.gt_team.isin(["left", "right"])]
    if m.empty or "pred_team" not in m:
        return {}
    a1 = ((m.gt_team == "left") & (m.pred_team == "A")) | ((m.gt_team == "right") & (m.pred_team == "B"))
    a2 = ((m.gt_team == "left") & (m.pred_team == "B")) | ((m.gt_team == "right") & (m.pred_team == "A"))
    acc = max(a1.mean(), a2.mean())
    correct = a1 if a1.mean() >= a2.mean() else a2
    per_team = {f"team_{t}_acc": float(correct[m.gt_team == t].mean()) for t in ("left", "right")}
    # players only (goalkeepers often wear a distinct kit)
    pm = m.gt_role == "player"
    return {"team_acc": float(acc), "team_acc_players": float(correct[pm].mean()) if pm.any() else np.nan,
            **per_team, "n": int(len(m))}


# ── ball ──────────────────────────────────────────────────────────────────────

def gt_ball(seq: Sequence) -> pd.DataFrame:
    g = pd.DataFrame(seq.gt)
    b = g[g.role == "ball"].copy()
    b["x"] = (b.x1 + b.x2) / 2; b["y"] = (b.y1 + b.y2) / 2
    return b.groupby("frame")[["x", "y"]].first().reset_index()


def evaluate_ball(seq: Sequence, ball: pd.DataFrame, tol_px: float = 15.0) -> Dict[str, float]:
    """
    Frame-level ball localisation.
      recall    : fraction of GT-ball frames with a prediction within tol_px
      precision : fraction of predicted frames that are within tol_px of GT
                  (predictions on frames without a GT ball count as false)
    """
    gb = gt_ball(seq).set_index("frame")
    pb = ball.dropna(subset=["x", "y"]).groupby("frame")[["x", "y"]].first()
    joined = pb.merge(gb, left_index=True, right_index=True, how="outer", suffixes=("_p", "_g"))
    err = np.hypot(joined.x_p - joined.x_g, joined.y_p - joined.y_g)
    has_p = joined.x_p.notna(); has_g = joined.x_g.notna()
    good = has_p & has_g & (err <= tol_px)
    return {"ball_recall": float(good.sum() / max(has_g.sum(), 1)),
            "ball_precision": float(good.sum() / max(has_p.sum(), 1)),
            "ball_median_err_px": float(err[has_p & has_g].median()) if (has_p & has_g).any() else np.nan,
            "gt_ball_frames": int(has_g.sum()), "pred_ball_frames": int(has_p.sum())}


def calibration_height_check(H_frames, gt: pd.DataFrame, every: int = 5, person_h: float = 1.8,
                             tol: float = 0.25) -> dict:
    """
    Calibration check against ground truth that does not depend on any line detector.
    For every GT person box, the foot point is mapped to the pitch with the frame's
    image→pitch homography, a `person_h` tall vertical segment is projected back with
    the full camera (K, R, C recovered from the homography), and its pixel height is
    compared with the GT box height.  A wrong camera (bad tilt/height/zoom) gets the
    perspective scale wrong and the ratios drift away from ~1.

    H_frames : list of 3×3 image→pitch homographies, index 0 = frame 1
    Returns median ratio, fraction within ±tol, and fraction of feet inside the pitch (±3 m).
    """
    import cv2
    from src.calibration.camera_model import decompose
    g = gt[(gt.role != "ball") & ((gt.frame - 1) % every == 0)]
    ratios, inside = [], []
    for f, rows in g.groupby("frame"):
        if f - 1 >= len(H_frames):
            continue
        H = np.asarray(H_frames[f - 1], float)
        cam = decompose(H)
        R = cv2.Rodrigues(cam.rvec)[0]
        K = np.array([[cam.f, 0, cam.cx], [0, cam.f, cam.cy], [0, 0, 1.0]])
        feet = np.c_[((rows.x1 + rows.x2) / 2).to_numpy(float), rows.y2.to_numpy(float)]
        P = cv2.perspectiveTransform(feet[None], H)[0]
        inside.extend(((P[:, 0] > -3) & (P[:, 0] < 108) & (P[:, 1] > -3) & (P[:, 1] < 71)).tolist())
        for (px, py), (fx, fy), bh in zip(P, feet, (rows.y2 - rows.y1).to_numpy(float)):
            Xc = R @ (np.array([px, py, -person_h]) - cam.C)   # z < 0 is above the grass
            if Xc[2] <= 0 or bh < 10:
                continue
            u = K @ Xc
            ratios.append(abs(fy - u[1] / u[2]) / bh)
    r = np.array(ratios)
    if not len(r):
        return {"height_ratio_med": np.nan, "height_ok": np.nan, "feet_on_pitch": np.nan}
    return {"height_ratio_med": float(np.median(r)),
            "height_ok": float((np.abs(np.log(r)) < np.log(1 + tol)).mean()),
            "feet_on_pitch": float(np.mean(inside))}
