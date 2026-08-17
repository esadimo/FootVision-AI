"""
src.teams.classifier — Unsupervised team clustering and temporal majority voting.
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple


class TeamClassifier:
    """
    Classifies players into Team A, Team B, or Other/Ref by clustering color features,
    and applies temporal smoothing across each track's history.
    """

    def __init__(self, n_teams: int = 2, color_space: str = "hsv"):
        self.n_teams = n_teams
        self.color_space = color_space
        self.kmeans: Optional[KMeans] = None
        self.team_colors_bgr: Dict[str, Tuple[int, int, int]] = {
            "Team A": (0, 0, 255),       # Red
            "Team B": (255, 255, 255),   # White
            "Referee": (0, 255, 255),    # Yellow
            "Unknown": (180, 180, 180)   # Gray
        }
        # Temporal history buffer: {track_id: [list of instantaneous team predictions]}
        self.track_history: Dict[int, List[str]] = defaultdict(list)
        # Resolved stable labels: {track_id: "Team A" / "Team B"}
        self.stable_labels: Dict[int, str] = {}

    def fit(self, features: np.ndarray) -> None:
        """
        Fits K-Means clustering on the collected color feature vectors across players.

        Parameters
        ----------
        features : np.ndarray of shape (N, D)
            Array of dominant color feature vectors extracted from player crops.
        """
        if len(features) < self.n_teams:
            raise ValueError(f"Need at least {self.n_teams} samples to fit TeamClassifier, got {len(features)}")

        # Fit K-Means with n_teams clusters
        self.kmeans = KMeans(n_clusters=self.n_teams, n_init=10, random_state=42)
        self.kmeans.fit(features)

        # Determine visual BGR representation for each discovered cluster
        centers = self.kmeans.cluster_centers_
        if self.color_space == "hsv":
            for k in range(self.n_teams):
                hsv_val = np.uint8([[[int(centers[k][0]), int(centers[k][1]), int(centers[k][2])]]])
                bgr_val = cv2.cvtColor(hsv_val, cv2.COLOR_HSV2BGR)[0][0]
                team_name = f"Team {'A' if k == 0 else 'B'}"
                self.team_colors_bgr[team_name] = (int(bgr_val[0]), int(bgr_val[1]), int(bgr_val[2]))

    def predict_single(self, feature: np.ndarray) -> str:
        """
        Assigns an instantaneous team label for a single player feature vector.
        """
        if self.kmeans is None or feature is None:
            return "Unknown"

        cluster_idx = int(self.kmeans.predict(feature.reshape(1, -1))[0])
        return f"Team {'A' if cluster_idx == 0 else 'B'}"

    def update_track(self, track_id: int, instantaneous_label: str, min_votes: int = 5) -> str:
        """
        Applies temporal majority voting over the history of a track to prevent flickering.

        Parameters
        ----------
        track_id : int
            Unique player track ID.
        instantaneous_label : str
            The team predicted on the current frame.
        min_votes : int
            Minimum number of frames required before confirming a stable team label.

        Returns
        -------
        str
            Smoothed stable team label ("Team A", "Team B", or "Unknown").
        """
        if instantaneous_label != "Unknown":
            self.track_history[track_id].append(instantaneous_label)

        history = self.track_history[track_id]
        if len(history) < min_votes:
            # Fall back to instantaneous if not enough votes yet
            return instantaneous_label if instantaneous_label != "Unknown" else "Unknown"

        # Majority vote
        vote_counts = Counter(history)
        most_common_team, count = vote_counts.most_common(1)[0]
        
        # Lock in stable label
        self.stable_labels[track_id] = most_common_team
        return most_common_team

    def get_color(self, team_label: str) -> Tuple[int, int, int]:
        """Returns BGR color tuple for rendering the team label on canvas."""
        return self.team_colors_bgr.get(team_label, (180, 180, 180))
