"""
src.teams.classifier — Match-Specific Kit & Role Classifier.
Models:
  - Team A: White / Navy Kit (High Lightness, Low Saturation)
  - Team B: Green / White Kit (Green Hue 38-85, Negative 'a' in LAB, S >= 0.20)
  - Referee: Yellow Kit (Yellow Hue 20-38, High Saturation S >= 0.40)
  - Goalkeeper / Staff: Black / Dark Clothing (Very Low Lightness L < 0.35)
"""

import cv2
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple


class MatchKitClassifier:
    """
    Classifies players into Team A (White), Team B (Green), Referee (Yellow),
    or Goalkeeper/Staff (Black) using chromaticity and luminance rules,
    combined with temporal track smoothing.
    """

    def __init__(self):
        self.team_colors_bgr: Dict[str, Tuple[int, int, int]] = {
            "Team A": (240, 240, 240),      # White / Light Gray
            "Team B": (35, 180, 50),        # Bright Green
            "Referee": (0, 215, 255),       # Bright Yellow / Gold
            "Staff/GK": (45, 45, 45),       # Dark Black / Gray
            "Unknown": (140, 140, 140)
        }
        self.track_history: Dict[int, List[str]] = defaultdict(list)
        self.stable_labels: Dict[int, str] = {}

    def predict_single(self, metrics: Optional[Dict[str, float]]) -> str:
        """
        Classifies an individual player's chest color metrics.

        Parameters
        ----------
        metrics : dict with keys 'L', 'a', 'b', 'H', 'S', 'V'

        Returns
        -------
        'Team A', 'Team B', 'Referee', 'Staff/GK', or 'Unknown'
        """
        if metrics is None:
            return "Unknown"

        L = metrics["L"]
        a = metrics["a"]
        b = metrics["b"]
        H = metrics["H"]
        S = metrics["S"]
        V = metrics["V"]

        # 1. Check for Goalkeeper / Sideline Technical Staff (Black / Dark)
        if L < 0.35 or V < 0.30:
            return "Staff/GK"

        # 2. Check for Referee (Yellow / Gold shirt)
        # Hue between 18 and 38, high saturation S >= 0.38
        if (18.0 <= H <= 38.0 and S >= 0.35) or (b > 0.30 and S > 0.35):
            return "Referee"

        # 3. Check for Team B (Green / White kit)
        # Green Hue is between 38 and 85 in OpenCV HSV scale, with clear saturation
        # In LAB, 'a' is negative for green
        if (38.0 <= H <= 85.0 and S >= 0.18) or (a < -0.05 and S >= 0.15):
            return "Team B"

        # 4. Check for Team A (White / Navy kit)
        # High lightness L > 0.55, low saturation S < 0.28
        if L >= 0.55 and S < 0.28:
            return "Team A"

        # Margin resolver:
        # If there's green chromaticity -> Team B, else White Team A
        if a < -0.03:
            return "Team B"
        
        return "Team A"

    def update_track(self, track_id: int, instant_label: str, min_votes: int = 5) -> str:
        """
        Applies temporal sliding-window majority voting over the track's history.
        """
        if instant_label != "Unknown":
            self.track_history[track_id].append(instant_label)

        history = self.track_history[track_id]
        if len(history) < min_votes:
            return instant_label if instant_label != "Unknown" else "Unknown"

        # Sliding window over the recent 25 frames
        recent = history[-25:]
        vote_counts = Counter(recent)
        most_common, _ = vote_counts.most_common(1)[0]

        self.stable_labels[track_id] = most_common
        return most_common

    def get_color(self, label: str) -> Tuple[int, int, int]:
        """Returns BGR color for canvas rendering."""
        return self.team_colors_bgr.get(label, (140, 140, 140))
