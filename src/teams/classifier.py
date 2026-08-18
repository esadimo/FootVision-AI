"""
src.teams.classifier — 4-Component Kit Classifier for Football Analysis.
Properly separates:
  - Team A (Primary Outfield Kit)
  - Team B (Secondary Outfield Kit)
  - Referee (High-saturation / Bright Yellow kit)
  - Goalkeeper (Low-luminance / Distinct GK kit)
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple


class RobustKitClassifier:
    """
    Classifies players into Team A, Team B, Referee, or Goalkeeper using 4-component
    clustering with domain-informed heuristics and temporal track smoothing.
    """

    def __init__(self):
        self.team_centroids: Dict[str, np.ndarray] = {}
        self.team_colors_bgr: Dict[str, Tuple[int, int, int]] = {
            "Team A": (240, 240, 240),   # White (Team A)
            "Team B": (180, 80, 40),     # Dark Blue/Colored (Team B)
            "Referee": (0, 215, 255),    # Yellow/Gold (Referee)
            "Goalkeeper": (40, 40, 40),  # Dark Gray/Black (Goalkeeper)
            "Other": (0, 215, 255),
            "Unknown": (140, 140, 140)
        }
        self.track_history: Dict[int, List[str]] = defaultdict(list)
        self.stable_labels: Dict[int, str] = {}

    def fit(self, features: np.ndarray, representative_bgrs: np.ndarray) -> None:
        """
        Fits 4-component clustering to separate Team A, Team B, Referee, and Goalkeeper.

        Parameters
        ----------
        features : np.ndarray of shape (N, 4)
            [L_norm, a_norm, b_norm, S_norm] vectors.
        representative_bgrs : np.ndarray of shape (N, 3)
            Median BGR color for each sampled crop.
        """
        if len(features) < 20:
            raise ValueError(f"Need at least 20 crop samples to fit kit classifier, got {len(features)}")

        # Fit 4 clusters
        kmeans = KMeans(n_clusters=4, n_init=20, random_state=42)
        labels = kmeans.fit_predict(features)
        centers = kmeans.cluster_centers_

        # Classify each discovered cluster centroid based on domain properties:
        # L = centers[:, 0], a = centers[:, 1], b = centers[:, 2], S = centers[:, 3]
        cluster_roles = {}
        outfield_candidates = []

        for k in range(4):
            L, a, b, S = centers[k]
            
            # 1. Referee: High saturation (S > 0.50) or strong yellow chromaticity (b > 0.30)
            if S > 0.50 or b > 0.30:
                cluster_roles[k] = "Referee"
            # 2. Goalkeeper / Extreme Dark: Very low luminance (L < 0.30)
            elif L < 0.30:
                cluster_roles[k] = "Goalkeeper"
            else:
                outfield_candidates.append(k)

        # If we didn't identify exactly 2 outfield candidates, fallback by sorting by L
        if len(outfield_candidates) != 2:
            sorted_by_L = sorted(range(4), key=lambda k: centers[k][0], reverse=True)
            # Two highest L non-referee clusters are outfield teams
            non_ref = [k for k in sorted_by_L if cluster_roles.get(k) != "Referee"]
            if len(non_ref) >= 2:
                outfield_candidates = non_ref[:2]
            else:
                outfield_candidates = sorted_by_L[:2]

        # The lighter outfield cluster is Team A (White), the other is Team B
        c_first, c_second = outfield_candidates[0], outfield_candidates[1]
        if centers[c_first][0] >= centers[c_second][0]:
            cluster_roles[c_first] = "Team A"
            cluster_roles[c_second] = "Team B"
        else:
            cluster_roles[c_first] = "Team B"
            cluster_roles[c_second] = "Team A"

        # Record centroids
        for k, role in cluster_roles.items():
            if role in ["Team A", "Team B"]:
                self.team_centroids[role] = centers[k]
                mask = (labels == k)
                if np.any(mask):
                    bgr_mean = np.median(representative_bgrs[mask], axis=0).astype(int)
                    self.team_colors_bgr[role] = (int(bgr_mean[0]), int(bgr_mean[1]), int(bgr_mean[2]))

        print(f"  [Kit Classifier Initialized]")
        print(f"    Team A Centroid (L={self.team_centroids['Team A'][0]:.2f}, S={self.team_centroids['Team A'][3]:.2f}) -> BGR: {self.team_colors_bgr['Team A']}")
        print(f"    Team B Centroid (L={self.team_centroids['Team B'][0]:.2f}, S={self.team_centroids['Team B'][3]:.2f}) -> BGR: {self.team_colors_bgr['Team B']}")

    def predict_single(self, feature: np.ndarray) -> str:
        """
        Classifies an individual player feature vector into:
          - 'Team A' (Primary outfield kit)
          - 'Team B' (Secondary outfield kit)
          - 'Referee' (Referees with yellow/gold/distinct colors)
          - 'Goalkeeper' (Goalkeepers with dark/distinct kit)
        """
        if feature is None or not self.team_centroids:
            return "Unknown"

        L, a, b, S = feature

        # 1. Rule-based Referee detection (High saturation / Yellow-orange chromaticity)
        if S > 0.50 or (b > 0.25 and S > 0.35):
            return "Referee"

        # 2. Rule-based Goalkeeper detection (Extremely low lightness)
        if L < 0.30:
            return "Goalkeeper"

        # 3. Outfield Team Assignment (Euclidean distance in weighted LAB+S space)
        c_a = self.team_centroids["Team A"]
        c_b = self.team_centroids["Team B"]

        weights = np.array([2.0, 1.0, 1.0, 1.5], dtype=np.float32)
        dist_a = np.sqrt(np.sum(weights * ((feature - c_a) ** 2)))
        dist_b = np.sqrt(np.sum(weights * ((feature - c_b) ** 2)))

        return "Team A" if dist_a <= dist_b else "Team B"

    def update_track(self, track_id: int, instant_label: str, min_votes: int = 5) -> str:
        """
        Applies temporal smoothing across the track's history to prevent per-frame jitter.
        """
        if instant_label != "Unknown":
            self.track_history[track_id].append(instant_label)

        history = self.track_history[track_id]
        if len(history) < min_votes:
            return instant_label if instant_label != "Unknown" else "Unknown"

        # Sliding window majority vote over recent 30 frames
        recent = history[-30:]
        vote_counts = Counter(recent)
        most_common, _ = vote_counts.most_common(1)[0]

        self.stable_labels[track_id] = most_common
        return most_common

    def get_color(self, label: str) -> Tuple[int, int, int]:
        """Returns the BGR color for canvas rendering."""
        return self.team_colors_bgr.get(label, (0, 215, 255))
