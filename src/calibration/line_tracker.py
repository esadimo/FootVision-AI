"""
src.calibration.line_tracker — Drift-free per-frame homography by pitch-line alignment.

Problem with the old approach (scripts/phase9_auto_homography.py): it chained
frame-to-frame ECC warps of the *whole image* starting from one manual
calibration.  Errors accumulate and the moving players / crowd corrupt the
estimate — by frame 400 of SNMOT-062 the halfway line was projected 8–20 m off.

Approach here, per frame t:
  1. Predict H_t from H_{t-1} composed with the camera motion between the two
     frames (sparse optical flow on static background points, players masked).
  2. Detect pitch-line pixels: bright, thin, low-saturation structures on the
     grass (white top-hat on luminance, inside a grass mask, outside player
     boxes).  Build a distance transform of that line mask.
  3. Refine H_t by minimising the (robust, Huber) distance between densely
     sampled template line points projected into the image and the nearest
     detected line pixel, starting from the prediction.  A weak prior keeps the
     solution close to the prediction when only a few lines are visible.

Because each frame is re-anchored to the actual painted lines, errors do not
accumulate.  `alignment_score` exposes the same residual as a quality metric
(median / % of template points within N px of a detected line).

All homographies are image→pitch (pixels → metres), matching
src.calibration.homography.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares

from src.calibration.pitch_model import (CENTER_CIRCLE_RADIUS, GOAL_AREA_DEPTH, GOAL_AREA_WIDTH,
                                         PENALTY_BOX_DEPTH, PENALTY_BOX_WIDTH, PENALTY_SPOT_DIST,
                                         PITCH_LENGTH, PITCH_WIDTH)

L, W = PITCH_LENGTH, PITCH_WIDTH


# ── pitch template ────────────────────────────────────────────────────────────

def _segments(L: float, W: float):
    segs: List[Tuple[Tuple[float, float], Tuple[float, float]]] = [
        ((0, 0), (L, 0)), ((0, W), (L, W)), ((0, 0), (0, W)), ((L, 0), (L, W)),
        ((L / 2, 0), (L / 2, W)),
    ]
    for gx, sgn in ((0.0, 1), (L, -1)):
        for depth, width in ((PENALTY_BOX_DEPTH, PENALTY_BOX_WIDTH), (GOAL_AREA_DEPTH, GOAL_AREA_WIDTH)):
            y0, y1 = (W - width) / 2, (W + width) / 2
            x1 = gx + sgn * depth
            segs += [((gx, y0), (x1, y0)), ((gx, y1), (x1, y1)), ((x1, y0), (x1, y1))]
    return segs


def template_points(step: float = 0.5, L: float = PITCH_LENGTH, W: float = PITCH_WIDTH) -> np.ndarray:
    """Points (metres) sampled along every painted pitch line, for a pitch of L × W metres."""
    segs = _segments(L, W)
    # sample counts come from the nominal 105×68 pitch so that every point is an
    # exact linear function of (L, W) — needed to fit L and W by optimisation
    nominal = _segments(PITCH_LENGTH, PITCH_WIDTH)
    pts = []
    for ((ax, ay), (bx, by)), ((nax, nay), (nbx, nby)) in zip(segs, nominal):
        n = max(2, int(np.hypot(nbx - nax, nby - nay) / step))
        t = np.linspace(0, 1, n)
        pts.append(np.stack([ax + t * (bx - ax), ay + t * (by - ay)], 1))
    # centre circle
    n = int(2 * np.pi * CENTER_CIRCLE_RADIUS / step)
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts.append(np.stack([L / 2 + CENTER_CIRCLE_RADIUS * np.cos(a), W / 2 + CENTER_CIRCLE_RADIUS * np.sin(a)], 1))
    # penalty arcs (part outside the box)
    for sx, sgn in ((PENALTY_SPOT_DIST, 1), (L - PENALTY_SPOT_DIST, -1)):
        px = sx + CENTER_CIRCLE_RADIUS * np.cos(a)
        keep = (px - (PENALTY_BOX_DEPTH if sgn > 0 else L - PENALTY_BOX_DEPTH)) * sgn > 0
        pts.append(np.stack([px[keep], W / 2 + CENTER_CIRCLE_RADIUS * np.sin(a)[keep]], 1))
    return np.concatenate(pts).astype(np.float64)


TEMPLATE = template_points()


# ── image cues ────────────────────────────────────────────────────────────────

def grass_mask(frame: np.ndarray) -> np.ndarray:
    """Playing-surface mask: largest saturated-green region, holes (lines, players) filled."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    small = cv2.resize(hsv, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    m = cv2.inRange(small, (32, 90, 50), (85, 255, 255))
    # bridge the painted lines first: otherwise they cut the pitch into pieces and only the
    # largest piece survives (e.g. a whole shaded half of the pitch was dropped)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return np.zeros(frame.shape[:2], np.uint8)
    area = stats[1:, cv2.CC_STAT_AREA]
    keep = 1 + np.flatnonzero(area >= 0.1 * area.max())
    m = np.isin(lab, keep).astype(np.uint8) * 255
    # fill interior holes column-wise: the pitch is everything between its top edge and the image bottom
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, cnts, -1, 255, -1)
    return cv2.resize(filled, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)


def line_mask(frame: np.ndarray, boxes: Optional[np.ndarray] = None,
              grass: Optional[np.ndarray] = None) -> np.ndarray:
    """Binary mask of painted pitch-line pixels (thin, brighter and less saturated than local grass)."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lum = lab[:, :, 0]
    tophat = cv2.morphologyEx(lum, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    local_sat = cv2.blur(sat, (21, 21))
    m = ((tophat > 12) & (sat < 0.8 * local_sat)).astype(np.uint8) * 255
    g = grass if grass is not None else grass_mask(frame)
    g = cv2.erode(g, np.ones((15, 15), np.uint8))
    m &= g
    if boxes is not None:
        for x1, y1, x2, y2 in boxes.astype(int):
            pad = int(0.15 * (x2 - x1)) + 3
            m[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return m


def distance_field(mask: np.ndarray) -> np.ndarray:
    return cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5).astype(np.float32)


# ── geometry ──────────────────────────────────────────────────────────────────

def _project(Hinv: np.ndarray, pts_m: np.ndarray) -> np.ndarray:
    p = np.c_[pts_m, np.ones(len(pts_m))] @ Hinv.T
    return p[:, :2] / p[:, 2:3]


def _bilinear(df: np.ndarray, xy: np.ndarray) -> np.ndarray:
    return cv2.remap(df, xy[:, 0].astype(np.float32).reshape(-1, 1), xy[:, 1].astype(np.float32).reshape(-1, 1),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).ravel()


def _params_to_Hinv(p: np.ndarray, Hinv0: np.ndarray) -> np.ndarray:
    D = np.array([[1 + p[0], p[1], p[2]], [p[3], 1 + p[4], p[5]], [p[6], p[7], 1.0]])
    return D @ Hinv0


def _visible_template(Hinv: np.ndarray, shape, margin: int = 5) -> np.ndarray:
    h, w = shape[:2]
    q = np.c_[TEMPLATE, np.ones(len(TEMPLATE))] @ Hinv.T
    ok = q[:, 2] > 1e-6
    xy = q[:, :2] / np.where(ok, q[:, 2], 1)[:, None]
    ok &= (xy[:, 0] > margin) & (xy[:, 0] < w - margin) & (xy[:, 1] > margin) & (xy[:, 1] < h - margin)
    return TEMPLATE[ok]


def refine_homography(H: np.ndarray, dfield: np.ndarray, valid: np.ndarray,
                      prior_weight: float = 0.02, max_dist: float = 25.0) -> Tuple[np.ndarray, Dict]:
    """
    Refine image→pitch homography H so projected template lines land on detected lines.

    dfield : distance transform to nearest line pixel
    valid  : uint8 mask of image regions where lines can be observed (grass, no players)
    """
    Hinv0 = np.linalg.inv(H)
    Hinv0 = Hinv0 / Hinv0[2, 2]
    pts = _visible_template(Hinv0, dfield.shape)
    info = {"n_pts": int(len(pts))}
    if len(pts) < 40:
        info["status"] = "too_few_points"
        return H, info

    xy0 = _project(Hinv0, pts)
    obs = _bilinear(valid.astype(np.float32), xy0) > 0.5
    pts = pts[obs]
    if len(pts) < 40:
        info["status"] = "too_few_observable"
        return H, info

    # Normalise parameters so that each has a comparable pixel effect
    scale = np.array([1e-3, 1e-3, 1.0, 1e-3, 1e-3, 1.0, 1e-6, 1e-6])

    def resid(q):
        p = q * scale
        Hi = _params_to_Hinv(p, Hinv0)
        xy = _project(Hi, pts)
        d = _bilinear(dfield, xy)
        d = np.minimum(d, max_dist)
        return np.concatenate([d / np.sqrt(len(pts)), prior_weight * q])

    r0 = resid(np.zeros(8))
    sol = least_squares(resid, np.zeros(8), loss="soft_l1", f_scale=3.0 / np.sqrt(len(pts)),
                        method="trf", max_nfev=60, diff_step=0.5)
    Hi = _params_to_Hinv(sol.x * scale, Hinv0)
    Hn = np.linalg.inv(Hi)
    Hn = Hn / Hn[2, 2]
    d_before = r0[: len(pts)] * np.sqrt(len(pts))
    d_after = sol.fun[: len(pts)] * np.sqrt(len(pts))
    info.update({"status": "ok", "med_before": float(np.median(d_before)),
                 "med_after": float(np.median(d_after)), "n_obs": int(len(pts))})
    return Hn, info


def shift_search(H: np.ndarray, dfield: np.ndarray, valid: np.ndarray,
                 max_shift: int = 240, step: int = 8, clip: float = 10.0) -> Tuple[np.ndarray, Dict]:
    """
    Coarse global search for an image-space translation of the projected template
    that best overlays the detected lines (captures fast pans the motion model missed).
    """
    Hinv = np.linalg.inv(H)
    pts = _visible_template(Hinv, dfield.shape, margin=-max_shift)
    if len(pts) < 40:
        return H, {"shift": (0, 0)}
    xy = _project(Hinv, pts)
    s = 4
    dsm = cv2.resize(np.minimum(dfield, clip * s), None, fx=1 / s, fy=1 / s, interpolation=cv2.INTER_AREA) / s
    vsm = cv2.resize(valid, None, fx=1 / s, fy=1 / s, interpolation=cv2.INTER_NEAREST) > 0
    hs, ws = dsm.shape
    xy_s = xy / s
    shifts = np.arange(-max_shift, max_shift + 1, step) / s
    best, best_score, base = (0.0, 0.0), None, None
    for dx in shifts:
        xs = np.round(xy_s[:, 0] + dx).astype(int)
        okx = (xs >= 0) & (xs < ws)
        for dy in shifts:
            ys = np.round(xy_s[:, 1] + dy).astype(int)
            ok = okx & (ys >= 0) & (ys < hs)
            if ok.sum() < 40:
                continue
            obs = vsm[ys[ok], xs[ok]]
            if obs.sum() < 40:
                continue
            d = dsm[ys[ok][obs], xs[ok][obs]]
            # mean clipped distance, favouring overlays that hit many line pixels
            sc = float(np.mean(np.minimum(d, clip))) - 0.002 * obs.sum()
            if dx == 0 and dy == 0:
                base = sc
            if best_score is None or sc < best_score:
                best_score, best = sc, (dx * s, dy * s)
    if best_score is None or base is None or best == (0.0, 0.0) or best_score > base - 0.5:
        return H, {"shift": (0, 0)}
    T = np.array([[1, 0, best[0]], [0, 1, best[1]], [0, 0, 1.0]])
    Hn = np.linalg.inv(T @ Hinv)
    return Hn / Hn[2, 2], {"shift": best}


def alignment_score(H: np.ndarray, dfield: np.ndarray, valid: np.ndarray) -> Dict[str, float]:
    """How well H explains the visible lines: median px distance & % of points within 3/5 px."""
    Hinv = np.linalg.inv(H)
    pts = _visible_template(Hinv, dfield.shape)
    if len(pts) == 0:
        return {"line_med_px": np.nan, "line_in3": np.nan, "line_in5": np.nan, "n": 0}
    xy = _project(Hinv, pts)
    obs = _bilinear(valid.astype(np.float32), xy) > 0.5
    d = _bilinear(dfield, xy[obs])
    if len(d) == 0:
        return {"line_med_px": np.nan, "line_in3": np.nan, "line_in5": np.nan, "n": 0}
    return {"line_med_px": float(np.median(d)), "line_in3": float((d <= 3).mean()),
            "line_in5": float((d <= 5).mean()), "n": int(len(d))}


# ── camera motion between frames ──────────────────────────────────────────────

def camera_motion(prev_gray: np.ndarray, gray: np.ndarray, mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Homography mapping prev-frame pixels → current-frame pixels (static background only)."""
    pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=1500, qualityLevel=0.005, minDistance=8, mask=mask)
    if pts is None or len(pts) < 30:
        return None
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, winSize=(21, 21), maxLevel=4)
    good = st.ravel() == 1
    if good.sum() < 30:
        return None
    M, inl = cv2.findHomography(pts[good], nxt[good], cv2.RANSAC, 2.0)
    if M is None or inl is None or inl.sum() < 20:
        return None
    return M


class LineHomographyTracker:
    """
    Per-frame homography tracker.

    tracker = LineHomographyTracker(H_init)        # image→pitch for the first frame
    for frame, person_boxes in ...:
        H, info = tracker.step(frame, person_boxes)
    """

    def __init__(self, H_init: np.ndarray, refine_every: int = 1):
        self.H = H_init / H_init[2, 2]
        self.prev_gray = None
        self.refine_every = refine_every
        self.t = 0

    def step(self, frame: np.ndarray, boxes: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        info: Dict = {}

        if self.prev_gray is not None:
            # motion from background features; mask out people (they move independently)
            m = np.full((h, w), 255, np.uint8)
            if boxes is not None:
                for x1, y1, x2, y2 in boxes.astype(int):
                    m[max(0, y1 - 5):y2 + 5, max(0, x1 - 5):x2 + 5] = 0
            M = camera_motion(self.prev_gray, gray, m)
            if M is not None:
                # p_pitch = H_prev p_prev ; p_cur = M p_prev  →  H_cur = H_prev M^-1
                self.H = self.H @ np.linalg.inv(M)
                self.H /= self.H[2, 2]
                info["motion"] = "ok"
            else:
                info["motion"] = "fail"
        self.prev_gray = gray

        if self.t % self.refine_every == 0:
            lm = line_mask(frame, boxes)
            df = distance_field(lm)
            valid = grass_mask(frame)
            if boxes is not None:
                for x1, y1, x2, y2 in boxes.astype(int):
                    valid[max(0, y1 - 3):y2 + 3, max(0, x1 - 3):x2 + 3] = 0
            # global translation search, then coarse→fine projective refinement
            H0, i0 = shift_search(self.H, df, valid)
            info["shift"] = i0["shift"]
            H1, i1 = refine_homography(H0, df, valid, max_dist=40.0)
            H2, i2 = refine_homography(H1, df, valid, max_dist=12.0)
            if _sane(H2, self.H, (w, h), max_shift_m=12.0):
                self.H = H2
                info.update({k: v for k, v in i2.items()})
            else:
                info["status"] = "rejected"
            info.update(alignment_score(self.H, df, valid))
        self.t += 1
        return self.H.copy(), info


def _sane(Hn: np.ndarray, Hp: np.ndarray, size, max_shift_m: float = 6.0) -> bool:
    """Reject refinements that move the image footprint implausibly far in one frame."""
    w, h = size
    pts = np.array([[[w * 0.5, h * 0.5]], [[w * 0.1, h * 0.9]], [[w * 0.9, h * 0.9]], [[w * 0.5, h * 0.35]]], np.float64)
    a = cv2.perspectiveTransform(pts, Hn)[:, 0]
    b = cv2.perspectiveTransform(pts, Hp)[:, 0]
    if not np.all(np.isfinite(a)):
        return False
    return float(np.max(np.linalg.norm(a - b, axis=1))) < max_shift_m
