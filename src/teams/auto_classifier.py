"""
src.teams.auto_classifier — Automatic K-Means Kit Color Discovery.

Replaces the hardcoded rule-based MatchKitClassifier with a fully data-driven
approach that works on ANY SoccerNet sequence without prior knowledge of kit colors.

How it works
------------
Stage 1 — FIT (color discovery, ~100 frames):
  1. Run YOLO on a sample of frames, extract torso LAB color vectors per detection.
  2. Run K-Means (K=4) on all collected LAB samples.
  3. Auto-label the 4 clusters via universal physical rules:
       - Darkest cluster (lowest L*)  → "Staff/GK"
       - Most-yellow cluster (highest b*, H < 40) → "Referee"
       - Two remaining clusters → "Team A" (larger) and "Team B" (smaller)
  4. Persist the cluster centroids + label map to a JSON file for instant reuse.

Stage 2 — PREDICT (full video):
  5. Each player torso is assigned the nearest centroid in LAB space.
  6. Temporal majority-voting smoothing (same as old classifier).

Usage
-----
    from src.teams.auto_classifier import AutoKitClassifier
    clf = AutoKitClassifier()
    clf.fit(seq_dir="data/raw/SNMOT-116", model=yolo_model, n_sample_frames=120)
    clf.save("outputs/SNMOT-116_kit_profile.json")
    # ... later in the pipeline ...
    label = clf.predict_single(torso_crop_bgr)
    stable = clf.update_track(track_id, label)
"""

import json
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────
ROLE_STAFF_GK = "Staff/GK"
ROLE_REFEREE  = "Referee"
ROLE_TEAM_A   = "Team A"
ROLE_TEAM_B   = "Team B"
ROLE_UNKNOWN  = "Unknown"

ALL_ROLES = [ROLE_TEAM_A, ROLE_TEAM_B, ROLE_REFEREE, ROLE_STAFF_GK, ROLE_UNKNOWN]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _bgr_to_lab_vec(crop_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Extract the median LAB (L, a, b) of a chest crop as a float32 array."""
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    if crop_bgr.shape[0] < 3 or crop_bgr.shape[1] < 3:
        return None
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    med = np.median(lab.reshape(-1, 3), axis=0).astype(np.float32)
    return med  # [L*, a*, b*] in OpenCV scale: L*=[0,255], a*=[0,255], b*=[0,255]


def _lab_to_normalized(lab_vec: np.ndarray) -> Dict[str, float]:
    """Convert raw OpenCV LAB vector to normalized [-1, 1] / [0, 1] metrics dict."""
    L_norm = float(lab_vec[0]) / 255.0
    a_norm = (float(lab_vec[1]) - 128.0) / 128.0
    b_norm = (float(lab_vec[2]) - 128.0) / 128.0
    return {"L": L_norm, "a": a_norm, "b": b_norm}


def _cluster_label_by_physics(centroids_raw: np.ndarray) -> Dict[int, str]:
    """
    Auto-label K-Means cluster indices to semantic roles using universal rules:
      1. Darkest (L*) cluster  → Staff/GK
      2. Most yellow-dominant (b* max, a* negative, H-equivalent < 40) → Referee
      3. Remaining two clusters sorted by size → Team A (larger), Team B (smaller)

    Parameters
    ----------
    centroids_raw : np.ndarray shape (K, 3)
        K-Means centroids in raw OpenCV LAB scale.

    Returns
    -------
    dict mapping cluster_index → role_string
    """
    K = len(centroids_raw)
    norms = [_lab_to_normalized(c) for c in centroids_raw]

    # Find darkest → Staff/GK
    L_vals = [n["L"] for n in norms]
    gk_idx = int(np.argmin(L_vals))

    remaining = [i for i in range(K) if i != gk_idx]

    # Among remaining, find most yellow (b* high, a* not too positive = not red)
    # Yellow in LAB: b* large positive, a* around 0 or slightly negative
    # We score yellow-ness as b_norm - abs(a_norm * 0.5)
    def yellow_score(i: int) -> float:
        n = norms[i]
        return n["b"] - abs(n["a"]) * 0.5

    ref_idx = max(remaining, key=yellow_score)
    remaining = [i for i in remaining if i != ref_idx]

    # Only if the referee cluster actually looks yellow-ish (b* > 0.05 in norm)
    # Otherwise it could just be a bright white cluster — don't mislabel it
    if norms[ref_idx]["b"] < 0.05:
        # No clearly yellow cluster → referee might be wearing neutral color
        # Fall back to picking the second-darkest (sometimes refs wear dark)
        # Put ref_idx back and pick the lightest of the remaining two as ref
        remaining_with_ref = [ref_idx] + remaining
        ref_idx = max(remaining_with_ref, key=lambda i: L_vals[i])
        remaining = [i for i in remaining_with_ref if i != ref_idx]

    # Remaining two → Team A and Team B; we'll sort by cluster L* (lighter = A by convention)
    # or we simply call them A and B — downstream the only requirement is consistency
    assert len(remaining) == 2, f"Expected 2 team clusters, got {len(remaining)}"
    ta_idx, tb_idx = sorted(remaining, key=lambda i: -L_vals[i])  # lighter = Team A

    return {
        gk_idx:  ROLE_STAFF_GK,
        ref_idx: ROLE_REFEREE,
        ta_idx:  ROLE_TEAM_A,
        tb_idx:  ROLE_TEAM_B,
    }


# ── Main Class ─────────────────────────────────────────────────────────────────

class AutoKitClassifier:
    """
    Fully automatic kit color classifier.

    Works on any football sequence without any prior knowledge of kit colors.
    Use `fit()` once per sequence, then `predict_single()` / `update_track()` per frame.
    """

    def __init__(self):
        self._centroids_raw: Optional[np.ndarray] = None   # shape (K, 3) raw LAB
        self._label_map: Dict[int, str] = {}               # cluster_idx → role
        self._team_colors_bgr: Dict[str, Tuple[int, int, int]] = {
            ROLE_TEAM_A:   (240, 240, 240),
            ROLE_TEAM_B:   (35, 190, 50),
            ROLE_REFEREE:  (0, 215, 255),
            ROLE_STAFF_GK: (45, 45, 45),
            ROLE_UNKNOWN:  (140, 140, 140),
        }
        self._track_history: Dict[int, List[str]] = defaultdict(list)
        self._stable_labels: Dict[int, str] = {}
        self.is_fitted: bool = False

    # ── Fitting ────────────────────────────────────────────────────────────────

    def fit(
        self,
        seq_dir: str,
        model,                      # ultralytics YOLO instance
        n_sample_frames: int = 120,
        conf_threshold: float = 0.20,
        n_clusters: int = 4,
        min_samples: int = 80,
    ) -> "AutoKitClassifier":
        """
        Discover kit colors automatically from the first `n_sample_frames` of `seq_dir`.

        Parameters
        ----------
        seq_dir : str
            Path to the SoccerNet MOT sequence folder (must contain img1/).
        model : YOLO
            Pre-loaded YOLO model for person detection.
        n_sample_frames : int
            Number of frames to scan for color samples (default 120 ≈ 5 s at 25 fps).
        conf_threshold : float
            YOLO confidence threshold for person detection during fit.
        n_clusters : int
            Number of K-Means clusters (default 4: A, B, Ref, GK).
        min_samples : int
            Minimum number of valid torso samples required to fit reliably.

        Returns
        -------
        self
        """
        from src.teams.crop_extractor import extract_torso_crop

        img_dir = os.path.join(seq_dir, "img1")
        valid_ext = (".jpg", ".jpeg", ".png")
        all_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(valid_ext)])

        # Sub-sample frames evenly across the sequence for diverse coverage
        total = len(all_files)
        indices = np.linspace(0, total - 1, min(n_sample_frames, total), dtype=int)
        sample_files = [all_files[i] for i in indices]

        print(f"\n[AutoKitClassifier] Fitting on {len(sample_files)} sample frames from '{os.path.basename(seq_dir)}'...")

        lab_samples: List[np.ndarray] = []

        for fname in sample_files:
            frame = cv2.imread(os.path.join(img_dir, fname))
            if frame is None:
                continue

            results = model.predict(
                source=frame,
                conf=conf_threshold,
                classes=[0],          # persons only
                verbose=False,
            )[0]

            boxes = results.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for xyxy in boxes.xyxy.int().cpu().tolist():
                x1, y1, x2, y2 = xyxy
                torso = extract_torso_crop(frame, (x1, y1, x2, y2))
                vec = _bgr_to_lab_vec(torso)
                if vec is not None:
                    lab_samples.append(vec)

        n_collected = len(lab_samples)
        print(f"[AutoKitClassifier] Collected {n_collected} torso color samples.")

        if n_collected < min_samples:
            print(f"[AutoKitClassifier] WARNING: only {n_collected} samples (< {min_samples}). "
                  f"Results may be inaccurate. Falling back to rule-based fallback.")
            self._use_rule_fallback()
            return self

        # K-Means clustering
        data = np.stack(lab_samples, axis=0).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
        _, labels, centroids = cv2.kmeans(
            data, n_clusters, None, criteria, attempts=10,
            flags=cv2.KMEANS_PP_CENTERS
        )

        self._centroids_raw = centroids  # shape (K, 3)
        self._label_map = _cluster_label_by_physics(centroids)

        # Derive BGR colors for visualization from centroid LAB values
        self._derive_team_colors()

        self.is_fitted = True
        self._print_summary()
        return self

    def _derive_team_colors(self) -> None:
        """Compute representative BGR colors for each cluster from centroids."""
        for cluster_idx, role in self._label_map.items():
            if self._centroids_raw is None:
                break
            lab_pixel = self._centroids_raw[cluster_idx].reshape(1, 1, 3).astype(np.uint8)
            bgr_pixel = cv2.cvtColor(lab_pixel, cv2.COLOR_LAB2BGR)
            b, g, r = int(bgr_pixel[0, 0, 0]), int(bgr_pixel[0, 0, 1]), int(bgr_pixel[0, 0, 2])
            # Clamp very dark GK color so it's visible on dark backgrounds
            if role == ROLE_STAFF_GK:
                self._team_colors_bgr[role] = (60, 60, 60)
            else:
                self._team_colors_bgr[role] = (b, g, r)

    def _use_rule_fallback(self) -> None:
        """Fallback: use a generic neutral palette when not enough samples."""
        self._centroids_raw = None
        self._label_map = {}
        self.is_fitted = False  # will use rule-based path

    def _print_summary(self) -> None:
        """Print discovered kit profile to console."""
        print("[AutoKitClassifier] Kit discovery complete:")
        norms = [_lab_to_normalized(c) for c in self._centroids_raw]
        for idx, role in self._label_map.items():
            n = norms[idx]
            bgr = self._team_colors_bgr[role]
            print(f"  Cluster {idx} → {role:12s}  "
                  f"L*={n['L']:.2f}  a*={n['a']:+.2f}  b*={n['b']:+.2f}  "
                  f"BGR={bgr}")

    # ── Inference ──────────────────────────────────────────────────────────────

    def predict_single(self, torso_bgr: Optional[np.ndarray]) -> str:
        """
        Classify a single chest crop to the nearest cluster centroid.

        Falls back to rule-based heuristics if classifier is not fitted.
        """
        if torso_bgr is None:
            return ROLE_UNKNOWN

        vec = _bgr_to_lab_vec(torso_bgr)
        if vec is None:
            return ROLE_UNKNOWN

        if not self.is_fitted or self._centroids_raw is None:
            return self._rule_based_predict(vec)

        # Nearest centroid in LAB space (L2 distance)
        dists = np.linalg.norm(self._centroids_raw - vec[np.newaxis, :], axis=1)
        nearest_cluster = int(np.argmin(dists))
        return self._label_map.get(nearest_cluster, ROLE_UNKNOWN)

    def _rule_based_predict(self, lab_vec: np.ndarray) -> str:
        """
        Generic rule-based fallback (used when K-Means fitting failed).
        These rules are more conservative than the old SNMOT-062 specific ones.
        """
        n = _lab_to_normalized(lab_vec)
        L, a, b = n["L"], n["a"], n["b"]

        if L < 0.30:
            return ROLE_STAFF_GK
        if b > 0.15 and b > abs(a) and L > 0.40:
            return ROLE_REFEREE
        if a < -0.05:
            return ROLE_TEAM_B
        if L >= 0.55:
            return ROLE_TEAM_A
        return ROLE_TEAM_A

    def update_track(self, track_id: int, instant_label: str, min_votes: int = 5) -> str:
        """
        Temporal sliding-window majority voting over the track's label history.
        Returns the stable (smoothed) label for the given track.
        """
        if instant_label != ROLE_UNKNOWN:
            self._track_history[track_id].append(instant_label)

        history = self._track_history[track_id]
        if len(history) < min_votes:
            return instant_label if instant_label != ROLE_UNKNOWN else ROLE_UNKNOWN

        recent = history[-25:]
        vote_counts = Counter(recent)
        most_common, _ = vote_counts.most_common(1)[0]
        self._stable_labels[track_id] = most_common
        return most_common

    def get_color(self, label: str) -> Tuple[int, int, int]:
        """Returns the BGR display color for a given role label."""
        return self._team_colors_bgr.get(label, (140, 140, 140))

    def get_all_colors(self) -> Dict[str, Tuple[int, int, int]]:
        """Returns the full role→BGR color mapping."""
        return dict(self._team_colors_bgr)

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Persist the fitted classifier profile to a JSON file."""
        if not self.is_fitted:
            print("[AutoKitClassifier] Warning: saving unfitted classifier (no centroids).")

        profile = {
            "is_fitted": self.is_fitted,
            "centroids_raw": (
                self._centroids_raw.tolist() if self._centroids_raw is not None else None
            ),
            "label_map": {str(k): v for k, v in self._label_map.items()},
            "team_colors_bgr": {k: list(v) for k, v in self._team_colors_bgr.items()},
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(profile, f, indent=2)
        print(f"[AutoKitClassifier] Profile saved → {path}")

    def load(self, path: str) -> "AutoKitClassifier":
        """Load a previously fitted classifier profile from JSON."""
        with open(path) as f:
            profile = json.load(f)

        self.is_fitted = profile.get("is_fitted", False)
        raw = profile.get("centroids_raw")
        self._centroids_raw = np.array(raw, dtype=np.float32) if raw is not None else None
        self._label_map = {int(k): v for k, v in profile.get("label_map", {}).items()}
        raw_colors = profile.get("team_colors_bgr", {})
        self._team_colors_bgr = {k: tuple(v) for k, v in raw_colors.items()}
        print(f"[AutoKitClassifier] Profile loaded ← {path}")
        return self

    @classmethod
    def from_file(cls, path: str) -> "AutoKitClassifier":
        """Class-method convenience: load and return a fitted instance."""
        return cls().load(path)
