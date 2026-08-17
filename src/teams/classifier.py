"""
src.teams.classifier — Robust team clustering with outlier detection (Referee / Goalkeeper / Other)
and temporal majority voting per track.
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple


class RobustTeamClassifier:
    """
    Classifies players into Team A, Team B, or Other/Ref/GK by clustering 4D LAB+S features
    with outlier distance rejection and temporal smoothing.
    """

    def __init__(self, outlier_dist_thresh: float = 0.28):
        self.outlier_dist_thresh = outlier_dist_thresh
        self.team_centroids: Dict[str, np.ndarray] = {}
        self.team_colors_bgr: Dict[str, Tuple[int, int, int]] = {
            "Team A": (220, 220, 220),   # Light default
            "Team B": (180, 80, 50),     # Dark/Color default
            "Other": (0, 220, 255),      # Yellow for Ref / GK / Outliers
            "Unknown": (140, 140, 140)   # Gray
        }
        # Temporal history per track ID: {track_id: [predictions]}
        self.track_history: Dict[int, List[str]] = defaultdict(list)
        self.stable_labels: Dict[int, str] = {}

    def fit(self, features: np.ndarray, representative_bgrs: np.ndarray) -> None:
        """
        Fits K-Means to discover the two main outfield teams from the feature distribution,
        and records their centroids.

        Parameters
        ----------
        features : np.ndarray of shape (N, 4)
            Array of [L_norm, a_norm, b_norm, S_norm] vectors.
        representative_bgrs : np.ndarray of shape (N, 3)
            Corresponding median BGR color of each sampled crop.
        """
        if len(features) < 10:
            raise ValueError(f"Need at least 10 feature samples to fit team classifier, got {len(features)}")

        # Fit 2 outfield clusters
        kmeans = KMeans(n_clusters=2, n_init=15, random_state=42)
        labels = kmeans.fit_predict(features)
        
        c0 = kmeans.cluster_centers_[0]
        c1 = kmeans.cluster_centers_[1]

        # Determine which cluster is lighter (higher L) and assign Team A / Team B consistently
        if c0[0] >= c1[0]:
            self.team_centroids["Team A"] = c0
            self.team_centroids["Team B"] = c1
            mask_a = (labels == 0)
            mask_b = (labels == 1)
        else:
            self.team_centroids["Team A"] = c1
            self.team_centroids["Team B"] = c0
            mask_a = (labels == 1)
            mask_b = (labels == 0)

        # Average BGR color for clean canvas rendering
        if np.any(mask_a):
            bgr_a = np.median(representative_bgrs[mask_a], axis=0).astype(int)
            self.team_colors_bgr["Team A"] = (int(bgr_a[0]), int(bgr_a[1]), int(bgr_a[2]))
        if np.any(mask_b):
            bgr_b = np.median(representative_bgrs[mask_b], axis=0).astype(int)
            self.team_colors_bgr["Team B"] = (int(bgr_b[0]), int(bgr_b[1]), int(bgr_b[2]))

    def predict_single(self, feature: np.ndarray) -> str:
        """
        Predicts team membership with outlier rejection.

        Returns
        -------
        "Team A", "Team B", or "Other" (for Referees, Goalkeepers, or non-matching kits).
        """
        if feature is None or not self.team_centroids:
            return "Unknown"

        c_a = self.team_centroids["Team A"]
        c_b = self.team_centroids["Team B"]

        # Weighted Euclidean distance (give high weight to Lightness and Chromaticity)
        weights = np.array([1.5, 1.2, 1.2, 0.8], dtype=np.float32)
        
        dist_a = np.sqrt(np.sum(weights * ((feature - c_a) ** 2)))
        dist_b = np.sqrt(np.sum(weights * ((feature - c_b) ** 2)))

        min_dist = min(dist_a, dist_b)

        # Outlier rejection: if distance to BOTH team clusters is too large, it's a referee/GK
        if min_dist > self.outlier_dist_thresh:
            return "Other"

        # Margin check: if the difference between dist_a and dist_b is too ambiguous
        if abs(dist_a - dist_b) < 0.04 and min_dist > 0.18:
            return "Other"

        return "Team A" if dist_a < dist_b else "Team B"

    def update_track(self, track_id: int, instant_label: str, min_votes: int = 5) -> str:
        """
        Maintains temporal majority voting per track to eliminate jitter.
        """
        if instant_label != "Unknown":
            self.track_history[track_id].append(instant_label)

        history = self.track_history[track_id]
        if len(history) < min_votes:
            return instant_label if instant_label != "Unknown" else "Unknown"

        # Count votes over the recent 25 frames
        recent_history = history[-25:]
        vote_counts = Counter(recent_history)
        most_common, count = vote_counts.most_common(1)[0]
        
        # If no single category has > 55% majority, mark as ambiguous Other
        if count / len(recent_history) < 0.55:
            return "Other"

        self.stable_labels[track_id] = most_common
        return most_common

    def get_color(self, team_label: str) -> Tuple[int, int, int]:
        """Returns BGR color tuple for drawing."""
        return self.team_colors_bgr.get(team_label, (140, 140, 140))
