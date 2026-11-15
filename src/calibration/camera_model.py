"""
src.calibration.camera_model — physically constrained broadcast-camera calibration.

A main broadcast camera sits at a fixed position C in the stadium and only
pans / tilts / rolls / zooms.  Its pitch→image mapping is therefore

    G = K(f) · R · [e1  e2  -C]          (pitch plane Z = 0, metres)
    H = G⁻¹                               (image → pitch, as used everywhere else)

with K = [[f,0,cx],[0,f,cy],[0,0,1]] (principal point at the image centre,
square pixels).  Per frame there are only 4 unknowns (rotation vector ω, focal
f) instead of 8 for a free homography, and C (3 values) is shared by the whole
sequence.  That makes the geometry correct across the entire pitch even when
only one or two lines are visible — a free homography fitted to landmarks in
one area extrapolates badly elsewhere (the cause of the 8–20 m errors in the
old pipeline).

World frame: X along the pitch (0..105), Y across (0..68), Z up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares

from functools import lru_cache

from src.calibration.line_tracker import TEMPLATE, _bilinear, _project, template_points


@lru_cache(maxsize=64)
def _template_cached(L: float, W: float) -> np.ndarray:
    return template_points(0.5, L, W)


def get_template(L: float = 105.0, W: float = 68.0) -> np.ndarray:
    """Pitch-line template for a pitch of L × W metres (rounded to 1 cm for caching)."""
    return _template_cached(round(float(L), 2), round(float(W), 2))


F_MIN, F_MAX = 300.0, 20000.0      # plausible focal range (px) for 1080p broadcast zoom lenses


@dataclass
class CameraParams:
    rvec: np.ndarray     # (3,) world→camera rotation (Rodrigues)
    f: float             # focal length in pixels
    C: np.ndarray        # (3,) camera centre in metres (X, Y, Z)
    cx: float = 960.0
    cy: float = 540.0
    L: float = 105.0     # pitch length (m) — fitted per stadium, not assumed
    W: float = 68.0      # pitch width (m)

    def K(self) -> np.ndarray:
        return np.array([[self.f, 0, self.cx], [0, self.f, self.cy], [0, 0, 1.0]])

    def R(self) -> np.ndarray:
        return cv2.Rodrigues(np.asarray(self.rvec, np.float64))[0]

    def G(self) -> np.ndarray:
        """pitch (X, Y, 1) → image homography."""
        R = self.R()
        M = np.c_[R[:, 0], R[:, 1], -R @ self.C]
        return self.K() @ M

    def H(self) -> np.ndarray:
        """image → pitch homography."""
        Hm = np.linalg.inv(self.G())
        return Hm / Hm[2, 2]

    def copy(self) -> "CameraParams":
        return CameraParams(np.array(self.rvec, float), float(self.f), np.array(self.C, float), self.cx, self.cy,
                            self.L, self.W)


def G_from(rvec, f, C, cx, cy) -> np.ndarray:
    R = cv2.Rodrigues(np.asarray(rvec, np.float64))[0]
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    return K @ np.c_[R[:, 0], R[:, 1], -R @ np.asarray(C, float)]


def decompose(H: np.ndarray, cx: float = 960.0, cy: float = 540.0) -> CameraParams:
    """Recover (R, f, C) from an image→pitch homography (Zhang-style, known principal point)."""
    G = np.linalg.inv(H)
    T = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1.0]])
    g = T @ G
    cands = []
    num = -(g[0, 0] * g[0, 1] + g[1, 0] * g[1, 1]); den = g[2, 0] * g[2, 1]
    if abs(den) > 1e-12 and num / den > 0:
        cands.append(num / den)
    num2 = g[0, 1] ** 2 + g[1, 1] ** 2 - g[0, 0] ** 2 - g[1, 0] ** 2; den2 = g[2, 0] ** 2 - g[2, 1] ** 2
    if abs(den2) > 1e-12 and num2 / den2 > 0:
        cands.append(num2 / den2)
    f = float(np.sqrt(np.mean(cands))) if cands else 2000.0
    Kinv = np.diag([1 / f, 1 / f, 1.0])
    a = Kinv @ g
    lam = 2.0 / (np.linalg.norm(a[:, 0]) + np.linalg.norm(a[:, 1]))
    r1, r2, t = lam * a[:, 0], lam * a[:, 1], lam * a[:, 2]
    # camera must look at the pitch: centre of pitch has positive depth
    if (r1 * 52.5 + r2 * 34 + t)[2] < 0:
        r1, r2, t = -r1, -r2, -t
    R = np.c_[r1, r2, np.cross(r1, r2)]
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
    C = -R.T @ t
    return CameraParams(cv2.Rodrigues(R)[0].ravel(), f, C, cx, cy)


def fit_to_homography(H: np.ndarray, cam: CameraParams, size=(1920, 1080)) -> CameraParams:
    """Best (ω, f) for fixed C so that the camera model reproduces H on an image grid."""
    w, h = size
    gx, gy = np.meshgrid(np.linspace(0.05 * w, 0.95 * w, 7), np.linspace(0.3 * h, 0.98 * h, 5))
    img = np.stack([gx.ravel(), gy.ravel()], 1)
    pitch = cv2.perspectiveTransform(img[None].astype(np.float64), H)[0]
    ok = np.all(np.isfinite(pitch), 1) & (np.abs(pitch[:, 0] - 52.5) < 120) & (np.abs(pitch[:, 1] - 34) < 90)
    pitch, img = pitch[ok], img[ok]

    def res(p):
        Gm = G_from(p[:3], np.exp(p[3]), cam.C, cam.cx, cam.cy)
        q = np.c_[pitch, np.ones(len(pitch))] @ Gm.T
        return (q[:, :2] / q[:, 2:3] - img).ravel()

    p0 = np.r_[cam.rvec, np.clip(np.log(cam.f), np.log(F_MIN) + 1e-3, np.log(F_MAX) - 1e-3)]
    lo = np.r_[np.full(3, -np.inf), np.log(F_MIN)]; hi = np.r_[np.full(3, np.inf), np.log(F_MAX)]
    try:
        s = least_squares(res, p0, method="trf", bounds=(lo, hi), max_nfev=200)
    except ValueError:
        return cam.copy()
    if not np.all(np.isfinite(s.x)):
        return cam.copy()
    out = cam.copy(); out.rvec = s.x[:3]; out.f = float(np.exp(s.x[3]))
    return out


# ── line-alignment objective ──────────────────────────────────────────────────

def visible_template(G: np.ndarray, shape, margin: int = 5, template: Optional[np.ndarray] = None) -> np.ndarray:
    T = TEMPLATE if template is None else template
    h, w = shape[:2]
    q = np.c_[T, np.ones(len(T))] @ G.T
    ok = q[:, 2] > 1e-6
    xy = q[:, :2] / np.where(ok, q[:, 2], 1)[:, None]
    ok &= (xy[:, 0] > margin) & (xy[:, 0] < w - margin) & (xy[:, 1] > margin) & (xy[:, 1] < h - margin)
    return T[ok]


def robust_lm(resid_fn, x0: np.ndarray, h: np.ndarray, clip: float, delta: float = 2.0,
              iters: int = 40, prior_w: float = 0.0) -> np.ndarray:
    """
    Levenberg–Marquardt with central-difference Jacobian and Huber IRLS weights.

    resid_fn(x) -> distances (≥0, clipped at `clip`). Points at the clip carry no
    gradient information and get zero weight.  `h` is the finite-difference step per
    parameter (chosen ≈ 0.5–1 px of image motion).  A weak ridge prior pulls toward x0.
    """
    x = x0.astype(float).copy()
    r = resid_fn(x)

    def cost(rr, xx):
        a = np.abs(rr)
        hub = np.where(a <= delta, 0.5 * a ** 2, delta * (a - 0.5 * delta))
        return float(hub.sum() + 0.5 * prior_w * np.sum(((xx - x0) / h) ** 2))

    c = cost(r, x)
    lam = 1e-2
    for _ in range(iters):
        J = np.empty((len(r), len(x)))
        for k in range(len(x)):
            e = np.zeros(len(x)); e[k] = h[k]
            J[:, k] = (resid_fn(x + e) - resid_fn(x - e)) / (2 * h[k])
        a = np.abs(r)
        w = np.where(a >= clip - 1e-3, 0.0, np.where(a <= delta, 1.0, delta / np.maximum(a, 1e-9)))
        A = (J * w[:, None]).T @ J + prior_w * np.diag(1 / h ** 2)
        g = (J * w[:, None]).T @ r + prior_w * (x - x0) / h ** 2
        improved = False
        for _ in range(8):
            try:
                dx = -np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), g)
            except np.linalg.LinAlgError:
                lam *= 10; continue
            xn = x + dx
            rn = resid_fn(xn); cn = cost(rn, xn)
            if cn < c:
                x, r, c = xn, rn, cn
                lam = max(lam / 3, 1e-6); improved = True
                break
            lam *= 4
        if not improved or np.all(np.abs(dx) < 0.01 * h):
            break
    return x


def _residuals(pts, dfield, G, max_dist):
    q = np.c_[pts, np.ones(len(pts))] @ G.T
    xy = q[:, :2] / q[:, 2:3]
    return np.minimum(_bilinear(dfield, xy), max_dist)


def refine_camera(cam: CameraParams, dfield: np.ndarray, valid: np.ndarray, max_dist: float = 15.0,
                  prior: Optional[CameraParams] = None, prior_w: float = 0.5,
                  min_pts: int = 30) -> Tuple[CameraParams, Dict]:
    """Optimise (ω, f) of one frame so template lines land on detected line pixels."""
    if not (np.isfinite(cam.f) and F_MIN <= cam.f <= F_MAX and np.all(np.isfinite(cam.rvec))):
        return cam, {"status": "invalid", "n_obs": 0}
    G0 = cam.G()
    pts = visible_template(G0, dfield.shape, template=get_template(cam.L, cam.W))
    if len(pts) < min_pts:
        return cam, {"status": "few_pts", "n_obs": len(pts)}
    q = np.c_[pts, np.ones(len(pts))] @ G0.T
    xy0 = q[:, :2] / q[:, 2:3]
    pts = pts[_bilinear(valid.astype(np.float32), xy0) > 0.5]
    if len(pts) < min_pts:
        return cam, {"status": "few_obs", "n_obs": len(pts)}
    ref = prior if prior is not None else cam
    x_ref = np.r_[ref.rvec, np.log(ref.f)]
    # finite-difference steps ≈ 0.5 px of image motion at this zoom
    h = np.r_[np.full(3, 0.5 / cam.f), 0.5 / 1000.0]

    def rf(x):
        return _residuals(pts, dfield, G_from(x[:3], np.exp(x[3]), cam.C, cam.cx, cam.cy), max_dist)

    x0 = np.r_[cam.rvec, np.log(cam.f)]
    before = rf(x0)
    # ridge prior centred on the prior camera (weak; prevents drifting along unconstrained directions)
    x = robust_lm(lambda xx: rf(xx), x0, h, clip=max_dist, prior_w=prior_w * 0.02 if prior is not None else 0.0)
    after = rf(x)
    out = cam.copy(); out.rvec = x[:3]; out.f = float(np.clip(np.exp(x[3]), F_MIN, F_MAX))
    return out, {"status": "ok", "n_obs": n_pts(after), "med_before": float(np.median(before)),
                 "med_after": float(np.median(after)), "in3": float((after <= 3).mean())}


def n_pts(a):
    return int(len(a))


# Physically plausible broadcast-camera centres (pitch frame: x along the 105 m touchline, y across
# with the main camera beyond y = W, z negative = above the grass).  A centre near the ground plane
# is degenerate: the whole template collapses onto the horizon and can "match" any long straight
# edge (ad boards), which games the line-agreement score.
C_BOUNDS = np.array([[-15.0, 120.0], [72.0, 220.0], [-60.0, -5.0]])


def plausible_centre(C) -> bool:
    C = np.asarray(C, float)
    return bool(np.all(C >= C_BOUNDS[:, 0]) and np.all(C <= C_BOUNDS[:, 1]))


def refine_position(cams: List[CameraParams], dfields: List[np.ndarray], valids: List[np.ndarray],
                    max_dist: float = 12.0, fit_dims: bool = True) -> Tuple[np.ndarray, List[CameraParams]]:
    """
    Jointly optimise the sequence-level unknowns — camera centre C and pitch
    dimensions (L, W) — together with per-frame (ω, f) over several frames.
    Pitch dimensions vary by stadium (100–110 × 64–75 m); assuming 105 × 68
    leaves metre-level errors in pitch coordinates.
    """
    C0 = cams[0].C.copy(); L0, W0 = cams[0].L, cams[0].W
    T0 = get_template(105.0, 68.0)
    dTdL = get_template(106.0, 68.0) - T0          # exact: template is linear in L and W
    dTdW = get_template(105.0, 69.0) - T0

    def template(L, W):
        return T0 + (L - 105.0) * dTdL + (W - 68.0) * dTdW

    Tcur = template(L0, W0)
    idx_list = []
    for cam, df, v in zip(cams, dfields, valids):
        G = cam.G()
        h_img, w_img = df.shape[:2]
        q = np.c_[Tcur, np.ones(len(Tcur))] @ G.T
        okz = q[:, 2] > 1e-6
        xy = q[:, :2] / np.where(okz, q[:, 2], 1)[:, None]
        ok = okz & (xy[:, 0] > 5) & (xy[:, 0] < w_img - 5) & (xy[:, 1] > 5) & (xy[:, 1] < h_img - 5)
        idx = np.flatnonzero(ok)
        if len(idx):
            idx = idx[_bilinear(v.astype(np.float32), xy[idx]) > 0.5]
        idx_list.append(idx)
    keep = [i for i, p in enumerate(idx_list) if len(p) >= 30]
    if len(keep) < 3:
        return C0, cams
    nd = 2 if fit_dims else 0
    x0 = np.r_[C0, [L0, W0][:nd], np.concatenate([np.r_[cams[i].rvec, np.log(cams[i].f)] for i in keep])]
    h = np.r_[np.full(3, 0.1), np.full(nd, 0.1),
              np.concatenate([np.r_[np.full(3, 0.5 / cams[i].f), 0.5 / 1000.0] for i in keep])]
    o = 3 + nd

    def rf(x):
        C = x[:3]
        L, W = (x[3], x[4]) if fit_dims else (L0, W0)
        T = template(L, W)
        out = []
        for j, i in enumerate(keep):
            p = x[o + 4 * j: o + 4 + 4 * j]
            out.append(_residuals(T[idx_list[i]], dfields[i],
                                  G_from(p[:3], np.exp(p[3]), C, cams[i].cx, cams[i].cy), max_dist))
        return np.concatenate(out)

    xopt = robust_lm(rf, x0, h, clip=max_dist, iters=80)
    C = xopt[:3]
    if not plausible_centre(C):
        return C0, cams
    L, W = (float(xopt[3]), float(xopt[4])) if fit_dims else (L0, W0)
    new = [c.copy() for c in cams]
    for j, i in enumerate(keep):
        p = xopt[o + 4 * j: o + 4 + 4 * j]
        new[i].rvec = p[:3]; new[i].f = float(np.exp(p[3]))
    for c in new:
        c.C = C.copy(); c.L = L; c.W = W
    return C, new


# ── sequence tracker ──────────────────────────────────────────────────────────

class PTZTracker:
    """
    Per-frame PTZ camera tracking:
        prediction  : previous camera ∘ frame-to-frame motion (LK on background, people masked)
        correction  : line-alignment refinement of (ω, f) with C fixed
    `run_sequence` additionally re-estimates the shared camera centre C from
    well-observed frames and re-tracks (alternating optimisation).
    """

    def __init__(self, cam: CameraParams):
        self.cam = cam.copy()
        self.prev_gray = None

    def step(self, frame: np.ndarray, boxes: Optional[np.ndarray] = None,
             cues: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Tuple[CameraParams, Dict]:
        from src.calibration.line_tracker import camera_motion, distance_field, grass_mask, line_mask
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        info: Dict = {}
        if self.prev_gray is not None:
            m = np.full((h, w), 255, np.uint8)
            if boxes is not None:
                for x1, y1, x2, y2 in boxes.astype(int):
                    m[max(0, y1 - 8):y2 + 8, max(0, x1 - 8):x2 + 8] = 0
            M = camera_motion(self.prev_gray, gray, m)
            if M is not None:
                Hp = self.cam.H() @ np.linalg.inv(M)
                self.cam = fit_to_homography(Hp / Hp[2, 2], self.cam, (w, h))
        self.prev_gray = gray
        pred = self.cam.copy()

        if cues is None:
            g = grass_mask(frame)
            valid = g.copy()
            if boxes is not None:
                for x1, y1, x2, y2 in boxes.astype(int):
                    valid[max(0, y1 - 3):y2 + 3, max(0, x1 - 3):x2 + 3] = 0
            df = distance_field(line_mask(frame, boxes, grass=g))
        else:
            df, valid = cues
        s_pred, n_pred = line_in3(pred, df, valid)
        q_pred = fit_quality(pred, df, valid, boxes)
        best, best_s, best_n, best_q = pred, s_pred, n_pred, q_pred
        # coarse→fine; keep whichever candidate explains the most line pixels (within 3 px)
        cand = pred
        for md in (20.0, 8.0, 4.0):
            cand, _ = refine_camera(cand, df, valid, max_dist=md, prior=pred, prior_w=0.2)
            s, n = line_in3(cand, df, valid)
            q = fit_quality(cand, df, valid, boxes)
            if q > best_q + 0.005:
                best, best_s, best_n, best_q = cand, s, n, q
        if best is not pred:
            info["corrected"] = True
        self.cam = best
        info.update({"in3": best_s, "in3_pred": s_pred, "n_obs": best_n, "quality": best_q})
        return self.cam.copy(), info


def track_sequence(frame_paths: List[str], cam0: CameraParams, boxes_by_frame: Dict[int, np.ndarray],
                   n_iters: int = 2, sample_every: int = 15, verbose: bool = True):
    """
    Full calibration of a sequence with alternating optimisation of the camera centre.
    boxes_by_frame keys are 1-indexed frame numbers (people boxes to mask out).
    Returns (list of CameraParams per frame, list of info dicts).
    """
    from src.calibration.line_tracker import distance_field, grass_mask, line_mask
    # pre-compute cues once (line distance field + valid mask) — the expensive part
    cues = []
    for i, p in enumerate(frame_paths):
        f = cv2.imread(p)
        b = boxes_by_frame.get(i + 1)
        g = grass_mask(f)
        v = g.copy()
        if b is not None:
            for x1, y1, x2, y2 in b.astype(int):
                v[max(0, y1 - 3):y2 + 3, max(0, x1 - 3):x2 + 3] = 0
        cues.append((distance_field(line_mask(f, b, grass=g)), v))

    cam = cam0.copy()
    for it in range(n_iters):
        tr = PTZTracker(cam)
        cams, infos = [], []
        for i, p in enumerate(frame_paths):
            c, info = tr.step(cv2.imread(p), boxes_by_frame.get(i + 1), cues=cues[i])
            cams.append(c); infos.append(info)
        if verbose:
            in3 = np.nanmean([x.get("in3", np.nan) for x in infos])
            print(f"  [calib iter {it}] C={np.round(cam.C, 2)}  mean line in3 {in3:.3f}")
        if it == n_iters - 1:
            break
        idx = [i for i in range(0, len(cams), sample_every) if infos[i].get("n_obs", 0) >= 60]
        if 0 not in idx:
            idx = [0] + idx
        C, new = refine_position([cams[i] for i in idx], [cues[i][0] for i in idx], [cues[i][1] for i in idx])
        cam = new[0]   # frame 0 jointly re-optimised with the new centre
    return cams, infos


# ── global camera-position search + full sequence calibration ─────────────────

def _cues_for(frame: np.ndarray, boxes: Optional[np.ndarray]):
    from src.calibration.line_tracker import distance_field, grass_mask, line_mask
    g = grass_mask(frame)
    v = g.copy()
    if boxes is not None:
        for x1, y1, x2, y2 in np.asarray(boxes).astype(int):
            v[max(0, y1 - 3):y2 + 3, max(0, x1 - 3):x2 + 3] = 0
    return distance_field(line_mask(frame, boxes, grass=g)), v


def line_in3(cam: CameraParams, dfield: np.ndarray, valid: np.ndarray, min_pts: int = 30) -> Tuple[float, int]:
    """Fraction of visible+observable template points within 3 px of a detected line."""
    G = cam.G()
    T = visible_template(G, dfield.shape, template=get_template(cam.L, cam.W))
    if len(T) < min_pts:
        return float("nan"), len(T)
    q = np.c_[T, np.ones(len(T))] @ G.T
    T = T[_bilinear(valid.astype(np.float32), q[:, :2] / q[:, 2:3]) > 0.5]
    if len(T) < min_pts:
        return float("nan"), len(T)
    return float((_residuals(T, dfield, G, 50.0) <= 3).mean()), len(T)


def _fit_frame(H: np.ndarray, C: np.ndarray, cue, L=105.0, W=68.0) -> CameraParams:
    dc = decompose(H)
    c = fit_to_homography(H, CameraParams(dc.rvec, dc.f, np.asarray(C, float), L=L, W=W))
    for md in (30.0, 12.0, 5.0):
        c, _ = refine_camera(c, cue[0], cue[1], max_dist=md)
    return c


def search_camera_position(H_frames: List[np.ndarray], cues_list: List, verbose: bool = True,
                           boxes_list: Optional[List] = None,
                           x_range=(36.0, 69.0), y_range=(72.0, 126.0), z_range=(-30.0, -4.0)) -> np.ndarray:
    """
    Multi-start search for the shared camera centre.  For each candidate C, every
    sample frame's (ω, f) is re-fitted to its detected lines; candidates are ranked
    by mean line agreement.  Coarse grid over a wide range of stadium geometries,
    then two finer grids around the best.
    """
    def score(C):
        vals = []
        for j, (H, cue) in enumerate(zip(H_frames, cues_list)):
            try:
                s = fit_quality(_fit_frame(H, C, cue), cue[0], cue[1], boxes_list[j] if boxes_list else None)
            except (np.linalg.LinAlgError, ValueError, ZeroDivisionError):
                s = 0.0
            vals.append(0.0 if np.isnan(s) else s)
        return float(np.mean(vals))

    steps = np.array([5.5, 9.0, 6.5])
    grid = [(x, y, z) for x in np.linspace(*x_range, 7) for y in np.linspace(*y_range, 7) for z in np.linspace(*z_range, 5)]
    best_s, best = max(((score(np.array(c)), c) for c in grid), key=lambda t: t[0])
    if verbose:
        print(f"  [calib] coarse best C={np.round(best, 1)} quality={best_s:.3f}")
    for shrink in (0.5, 0.25):
        d = steps * shrink
        fine = [(best[0] + dx * d[0], best[1] + dy * d[1], best[2] + dz * d[2])
                for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
        fine = [c for c in fine if plausible_centre(c)]
        cand = max(((score(np.array(c)), c) for c in fine), key=lambda t: t[0])
        if cand[0] > best_s:
            best_s, best = cand
    if verbose:
        print(f"  [calib] refined best C={np.round(best, 1)} quality={best_s:.3f}")
    return np.array(best, float)


def calibrate_sequence(frame_paths: List[str], H_init: Optional[np.ndarray], boxes_by_frame: Dict[int, np.ndarray],
                       n_anchors: int = 25, fit_dims: bool = False, verbose: bool = True, n_auto: int = 10,
                       reinit_anchors: bool = True):
    """
    End-to-end calibration of one clip from a single rough image→pitch homography
    for the first frame (from the landmark picker).

      1. rough pass: decomposed camera + motion chaining + per-frame line refinement
      2. global search of the camera centre on frames sampled across the clip
      3. joint refinement of the centre (and optionally pitch size)
      4. anchors: frames spread over the clip re-fitted with the final centre
      5. forward and backward tracking passes that re-anchor whenever an anchor
         explains the lines better; per frame the better of the two passes is kept
    Returns (cams, infos).
    """
    cues = [_cues_for(cv2.imread(p), boxes_by_frame.get(i + 1)) for i, p in enumerate(frame_paths)]
    n = len(frame_paths)

    def run(cam0, order, anchors):
        tr = PTZTracker(cam0)
        out = {}
        for i in order:
            c, info = tr.step(cv2.imread(frame_paths[i]), boxes_by_frame.get(i + 1), cues=cues[i])
            if i in anchors:
                a_q = fit_quality(anchors[i], cues[i][0], cues[i][1], boxes_by_frame.get(i + 1))
                if a_q > info.get("quality", 0.0):
                    tr.cam = anchors[i].copy(); c = tr.cam.copy()
                    info = {**info, "in3": line_in3(c, cues[i][0], cues[i][1])[0], "quality": a_q, "anchored": True}
            out[i] = (c, info)
        return out

    if H_init is not None:
        rough = run(decompose(H_init), range(n), {})
    else:
        # fully automatic: calibrate several frames from scratch, start from the most confident one
        best = (-1.0, None, 0)
        for k in [int(i) for i in np.linspace(0, n - 1, n_auto)]:
            cam_k, sc = auto_init_camera(cues[k], boxes=boxes_by_frame.get(k + 1))
            if verbose:
                print(f"  [calib] auto-init frame {k}: score {sc:.3f}")
            if cam_k is not None and sc > best[0]:
                best = (sc, cam_k, k)
        _, cam_k, k = best
        rough = run(cam_k, range(k, n), {})
        rough.update(run(cam_k, range(k, -1, -1), {}))
    cams = [rough[i][0] for i in range(n)]
    if verbose:
        print(f"  [calib] rough pass mean in3 {np.nanmean([rough[i][1].get('in3', np.nan) for i in range(n)]):.3f}")

    idx = [int(i) for i in np.linspace(0, n - 1, 12)]
    C = search_camera_position([cams[i].H() for i in idx], [cues[i] for i in idx], verbose=verbose,
                               boxes_list=[boxes_by_frame.get(i + 1) for i in idx])
    L = W = None
    sample = [_fit_frame(cams[i].H(), C, cues[i]) for i in idx]
    before = np.mean([fit_quality(c, cues[i][0], cues[i][1], boxes_by_frame.get(i + 1)) for c, i in zip(sample, idx)])
    C2, sample2 = C, sample
    for md in (20.0, 10.0, 5.0):
        C2, sample2 = refine_position(sample2, [cues[i][0] for i in idx], [cues[i][1] for i in idx],
                                      max_dist=md, fit_dims=fit_dims)
    after = np.mean([fit_quality(c, cues[i][0], cues[i][1], boxes_by_frame.get(i + 1)) for c, i in zip(sample2, idx)])
    if after > before and plausible_centre(C2):
        C, L, W = C2, sample2[0].L, sample2[0].W
    L = L or 105.0; W = W or 68.0
    if verbose:
        print(f"  [calib] camera centre {np.round(C, 2)}  pitch {L:.2f}x{W:.2f}  sample quality {max(before, after):.3f}")

    # anchors: re-fit evenly spaced frames from the rough pass with the final centre
    anchors = {}
    for i in [int(k) for k in np.linspace(0, n - 1, n_anchors)]:
        bx = boxes_by_frame.get(i + 1)
        c = _fit_frame(cams[i].H(), C, cues[i], L, W)
        # the rough pass can drift; an independent from-scratch fit with the known centre
        # gives anchors that can pull the tracker back
        if reinit_anchors:
            try:
                c2, _ = auto_init_camera(cues[i], C_guess=C, boxes=bx)
            except (np.linalg.LinAlgError, ValueError, ZeroDivisionError, cv2.error):
                c2 = None
            if c2 is not None:
                c2.L, c2.W = L, W
                if fit_quality(c2, cues[i][0], cues[i][1], bx) > fit_quality(c, cues[i][0], cues[i][1], bx):
                    c = c2
        s, _ = line_in3(c, cues[i][0], cues[i][1])
        hs = person_height_score(c, bx)
        if (not np.isnan(s) and s >= 0.25 and line_coverage(c, cues[i][0], cues[i][1]) >= 0.15
                and (np.isnan(hs) or hs >= 0.5)):
            anchors[i] = c
    first = min(anchors) if anchors else 0
    last = max(anchors) if anchors else n - 1
    start_f = anchors.get(first) or _fit_frame(cams[0].H(), C, cues[0], L, W)
    start_b = anchors.get(last) or _fit_frame(cams[-1].H(), C, cues[-1], L, W)
    fwd = run(start_f, list(range(first, n)) , anchors)
    # frames before the first anchor: track backwards from it
    fwd.update(run(start_f, list(range(first, -1, -1)), anchors))
    bwd = run(start_b, list(range(last, -1, -1)), anchors)
    bwd.update(run(start_b, list(range(last, n)), anchors))
    final_cams, infos = [], []
    for i in range(n):
        cf, inf_f = fwd[i]; cb, inf_b = bwd[i]
        if inf_b.get("quality", 0.0) > inf_f.get("quality", 0.0):
            final_cams.append(cb); infos.append({**inf_b, "pass": "bwd"})
        else:
            final_cams.append(cf); infos.append({**inf_f, "pass": "fwd"})
    if verbose:
        print(f"  [calib] {len(anchors)} anchors; final mean in3 {np.nanmean([x.get('in3', np.nan) for x in infos]):.3f} quality {np.mean([x.get('quality', 0.0) for x in infos]):.3f}")
    return final_cams, infos


# ── fully automatic initialisation (no manual landmarks) ──────────────────────

def rot_from_ptr(pan: float, tilt: float, roll: float = 0.0) -> np.ndarray:
    """
    World→camera rotation for a camera on the near side looking at the pitch.
    pan  : radians, + turns toward the right goal (+X)
    tilt : radians, + looks down
    World: X along pitch, Y toward the near side (camera side), Z down.
    """
    fwd = np.array([np.sin(pan) * np.cos(tilt), -np.cos(pan) * np.cos(tilt), np.sin(tilt)])
    right = np.array([np.cos(pan), np.sin(pan), 0.0])
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    if roll:
        cr, sr = np.cos(roll), np.sin(roll)
        R = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1.0]]) @ R
    return R


def line_coverage(cam: CameraParams, dfield: np.ndarray, valid: np.ndarray, s4: int = 4) -> float:
    """Fraction of detected line pixels (in the observable area) explained by the projected template."""
    h, w = dfield.shape[:2]
    lm = cv2.resize((dfield <= 1.0).astype(np.uint8), (w // s4, h // s4), interpolation=cv2.INTER_NEAREST) > 0
    lm &= cv2.resize(valid, (w // s4, h // s4), interpolation=cv2.INTER_NEAREST) > 0
    T = get_template(cam.L, cam.W)
    q = np.c_[T, np.ones(len(T))] @ cam.G().T
    ok = q[:, 2] > 1e-3
    uv = q[ok, :2] / q[ok, 2:3] / s4
    ok2 = (uv[:, 0] >= 0) & (uv[:, 0] < w // s4) & (uv[:, 1] >= 0) & (uv[:, 1] < h // s4)
    m = np.zeros((h // s4, w // s4), np.uint8)
    m[uv[ok2, 1].astype(int), uv[ok2, 0].astype(int)] = 1
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    return float((m.astype(bool) & lm).sum()) / max(int(lm.sum()), 1)


def person_height_score(cam: CameraParams, boxes: Optional[np.ndarray], person_h: float = 1.8,
                        sigma: float = 0.15, min_boxes: int = 3, margin_m: float = 4.0) -> float:
    """
    Fraction of person boxes whose height matches a `person_h` tall person standing at the
    box's foot point under this camera (soft score, log-ratio σ = `sigma`) and whose foot
    point lies on the pitch (± margin_m).  Independent of the line detector:
    a wrong tilt / zoom / camera height gets the perspective scale wrong, and degenerate
    fits that squash the template onto one edge get it wildly wrong.
    Returns NaN when there are too few usable boxes.
    """
    if boxes is None or len(boxes) == 0:
        return float("nan")
    b = np.asarray(boxes, float)
    bh = b[:, 3] - b[:, 1]
    h_img = 2 * cam.cy
    use = (bh >= 15) & (b[:, 1] > 2) & (b[:, 3] < h_img - 2)        # skip tiny and border-cut boxes
    if use.sum() < min_boxes:
        return float("nan")
    b, bh = b[use], bh[use]
    feet = np.c_[(b[:, 0] + b[:, 2]) / 2, b[:, 3]]
    try:
        P = cv2.perspectiveTransform(feet[None], cam.H())[0]
    except cv2.error:
        return 0.0
    R = cam.R()
    Xc = (np.c_[P, np.full(len(P), -person_h)] - cam.C) @ R.T           # z < 0 is above the grass
    ok = Xc[:, 2] > 1e-6
    if not ok.any():
        return 0.0
    v_head = cam.f * Xc[ok, 1] / Xc[ok, 2] + cam.cy
    r = np.abs(feet[ok, 1] - v_head) / bh[ok]
    # soft agreement: 1 at the right scale, ~0.6 at ±15 %, ~0.2 at ±30 % (a hard ±tol window
    # cannot tell a 2 % from a 20 % scale error, so it cannot steer the camera-centre search)
    good = np.zeros(len(b), float)
    good[np.flatnonzero(ok)] = np.exp(-0.5 * (np.log(np.maximum(r, 1e-6)) / sigma) ** 2)
    # people stand on (or right next to) the pitch: rules out fits that slide the template along
    # a set of parallel lines (e.g. the penalty-area line taken for the goal line, ≈16.5 m off)
    m = margin_m
    good *= (P[:, 0] > -m) & (P[:, 0] < cam.L + m) & (P[:, 1] > -m) & (P[:, 1] < cam.W + m)
    return float(good.mean())


def fit_quality(cam: CameraParams, dfield: np.ndarray, valid: np.ndarray,
                boxes: Optional[np.ndarray] = None) -> float:
    """
    Calibration fit score = precision (template points on detected lines, `line_in3`)
    × coverage (detected line pixels explained by the template, `line_coverage`)
    × person-height consistency (when people are visible).
    Precision alone rewards degenerate fits that squash the whole template onto one
    long edge; coverage alone rewards over-dense projections; line terms alone can
    prefer a wrong zoom/tilt that the people's apparent size rules out.
    """
    s, _ = line_in3(cam, dfield, valid)
    if np.isnan(s):
        return 0.0
    q = float(s) * line_coverage(cam, dfield, valid)
    hs = person_height_score(cam, boxes)
    if not np.isnan(hs):
        q *= 0.1 + 0.9 * hs
    return q


def auto_init_camera(cue, C_guess=(52.5, 100.0, -14.0), size=(1920, 1080), top_k: int = 8,
                     verbose: bool = False, n_keep: int = 60, n_rerank: int = 600,
                     boxes: Optional[np.ndarray] = None) -> Tuple[CameraParams, float]:
    """
    Calibrate a single frame from scratch: exhaustive search over pan / tilt / zoom with a
    coarse template, then local line refinement of the best candidates.
    """
    df, valid = cue
    w, h = size
    cx, cy = w / 2, h / 2
    T = template_points(2.0)                                 # coarse template (~360 pts)
    Th = np.c_[T, np.ones(len(T))]
    C = np.asarray(C_guess, float)
    dsm = np.minimum(df, 30.0)
    vf = valid > 0
    focals = np.exp(np.linspace(np.log(900), np.log(12000), 22))
    cands = []
    for f in focals:
        step = max(np.deg2rad(0.3), 30.0 / f)              # ≈30 px of image motion per grid step
        pans = np.arange(np.deg2rad(-65), np.deg2rad(65), step)
        tilts = np.arange(np.deg2rad(3), np.deg2rad(40), step)
        P, Tt = np.meshgrid(pans, tilts, indexing="ij")
        P, Tt = P.ravel(), Tt.ravel()
        # rotation rows for all (pan, tilt) at once
        fwd = np.stack([np.sin(P) * np.cos(Tt), -np.cos(P) * np.cos(Tt), np.sin(Tt)], 1)
        right = np.stack([np.cos(P), np.sin(P), np.zeros_like(P)], 1)
        down = np.cross(fwd, right)
        rel = T - C[:2]                                   # template relative to camera (Z: 0 - C_z)
        relz = -C[2]
        def proj(axis):
            return rel[None, :, 0] * axis[:, None, 0] + rel[None, :, 1] * axis[:, None, 1] + relz * axis[:, None, 2]
        zc = proj(fwd); xc = proj(right); yc = proj(down)
        front = zc > 1e-3
        u = f * xc / np.where(front, zc, 1) + cx
        v = f * yc / np.where(front, zc, 1) + cy
        ok = front & (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
        ui = np.clip(u, 0, w - 1).astype(int); vi = np.clip(v, 0, h - 1).astype(int)
        obs = ok & vf[vi, ui]
        near = obs & (dsm[vi, ui] <= 20.0)
        n_obs = obs.sum(1)
        prec = near.sum(1) / np.maximum(n_obs, 1)
        score = prec * np.minimum(1.0, n_obs / 40.0)
        keep = np.argsort(-score)[:n_keep]
        cands += [(float(score[k]), float(P[k]), float(Tt[k]), float(f)) for k in keep if n_obs[k] >= 25]
    cands.sort(reverse=True)
    # re-rank by two-sided agreement: template→lines precision × lines→template coverage
    s4 = 4
    lm_small = cv2.resize((df <= 1.0).astype(np.uint8), (w // s4, h // s4), interpolation=cv2.INTER_NEAREST) > 0
    lm_small &= cv2.resize(valid, (w // s4, h // s4), interpolation=cv2.INTER_NEAREST) > 0
    n_line = max(int(lm_small.sum()), 1)
    Td = template_points(0.5); Tdh = np.c_[Td, np.ones(len(Td))]
    rer = []
    for score, p, t, f in cands[:n_rerank]:
        R = rot_from_ptr(p, t)
        G = np.diag([f, f, 1.0]) @ np.c_[R[:, 0], R[:, 1], -R @ C]
        G[0] += cx * G[2]; G[1] += cy * G[2]
        q = Tdh @ G.T
        ok = q[:, 2] > 1e-3
        uv = q[ok, :2] / q[ok, 2:3] / s4
        ok2 = (uv[:, 0] >= 0) & (uv[:, 0] < w // s4) & (uv[:, 1] >= 0) & (uv[:, 1] < h // s4)
        m = np.zeros((h // s4, w // s4), np.uint8)
        m[uv[ok2, 1].astype(int), uv[ok2, 0].astype(int)] = 1
        m = cv2.dilate(m, np.ones((3, 3), np.uint8))
        coverage = float((m.astype(bool) & lm_small).sum()) / n_line
        comb = score * coverage
        if boxes is not None:
            hs = person_height_score(CameraParams(cv2.Rodrigues(R)[0].ravel(), f, C.copy(), cx, cy), boxes)
            if not np.isnan(hs):
                comb *= 0.1 + 0.9 * hs
        rer.append((comb, score, coverage, p, t, f))
    rer.sort(reverse=True)
    best, best_s = None, -1.0
    for comb, score, cov, p, t, f in rer[:top_k]:
        cam = CameraParams(cv2.Rodrigues(rot_from_ptr(p, t))[0].ravel(), f, C.copy(), cx, cy)
        for md in (40.0, 20.0, 8.0, 4.0):
            cam, _ = refine_camera(cam, df, valid, max_dist=md)
        s, n = line_in3(cam, df, valid)
        cov2 = line_coverage(cam, df, valid)
        comb2 = float(np.nan_to_num(s)) * cov2
        hs = person_height_score(cam, boxes)
        if not np.isnan(hs):
            comb2 *= 0.1 + 0.9 * hs
        if verbose:
            print(f"    cand pan={np.rad2deg(p):5.1f} tilt={np.rad2deg(t):4.1f} f={f:6.0f} prec={score:.3f} cov={cov:.3f}"
                  f" -> in3={s:.3f} cov={cov2:.3f} score={comb2:.3f}")
        if comb2 > best_s:
            best, best_s = cam, comb2
    return best, best_s


def pitch_error_m(cam: CameraParams, dfield: np.ndarray, valid: np.ndarray, max_px: int = 4000) -> Dict[str, float]:
    """
    Calibration error in metres: detected line pixels are back-projected onto the pitch
    and compared with the nearest painted line of the template.  Reports the median and
    the fraction within 0.5 m / 1 m (pixels farther than 3 m from any line are treated as
    non-line clutter and excluded from the median but reported as `clutter`).
    """
    from scipy.spatial import cKDTree
    ys, xs = np.nonzero((dfield <= 0.5) & (valid > 0))
    if len(xs) < 50:
        return {"err_med_m": np.nan, "within_0.5m": np.nan, "within_1m": np.nan, "clutter": np.nan}
    if len(xs) > max_px:
        sel = np.random.default_rng(0).choice(len(xs), max_px, replace=False)
        xs, ys = xs[sel], ys[sel]
    P = cv2.perspectiveTransform(np.stack([xs, ys], 1)[None].astype(np.float64), cam.H())[0]
    T = template_points(0.1, cam.L, cam.W)
    d, _ = cKDTree(T).query(P)
    good = d <= 3.0
    return {"err_med_m": float(np.median(d[good])) if good.any() else np.nan,
            "within_0.5m": float((d <= 0.5).sum() / max(good.sum(), 1)),
            "within_1m": float((d <= 1.0).sum() / max(good.sum(), 1)),
            "clutter": float(1 - good.mean())}
