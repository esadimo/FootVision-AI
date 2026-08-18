"""
src.teams.classifier — Match-Specific Kit & Role Classifier with Sharpened Hue-Chromaticity Boundaries.
Models:
  - Team A: White / Navy Kit (High Lightness, Low Saturation)
  - Team B: Green / White Kit (Green Hue, Negative 'a' in LAB dominant over 'b')
  - Referee: Yellow Kit (Yellow Hue, Positive 'b' in LAB dominant over '-a')
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
            "Team B": (35, 190, 50),        # Bright Green
            "Referee": (0, 215, 255),       # Bright Yellow / Gold
            "Staff/GK": (45, 45, 45),       # Dark Black / Gray
            "Unknown": (140, 140, 140)
        }
        self.track_history: Dict[int, List[str]] = defaultdict(list)
        self.stable_labels: Dict[int, str] = {}

    def predict_single(self, metrics: Optional[Dict[str, float]]) -> str:
        """
        Classifies an individual player's chest color metrics with sharp Ref vs Green separation.

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
        a = metrics["a"]   # Negative = Green, Positive = Magenta/Red
        b = metrics["b"]   # Positive = Yellow, Negative = Blue
        H = metrics["H"]   # OpenCV Hue: 0-180 (Yellow: ~20-35, Green: ~36-85)
        S = metrics["S"]
        V = metrics["V"]

        # 1. Goalkeeper / Sideline Technical Staff (Black / Dark)
        if L < 0.35 or V < 0.30:
            return "Staff/GK"

        # 2. Referee (Yellow / Gold shirt)
        # Yellow has strong positive 'b' (yellow axis) that exceeds the green component (-a),
        # with hue strictly in the yellow-gold range (H <= 36) and high saturation.
        if (b > 0.20 and b > (-a) and H <= 36.0 and S >= 0.35) or (18.0 <= H <= 34.0 and S >= 0.40):
            return "Referee"

        # 3. Team B (Green / White kit)
        # Green has negative 'a' (green axis) where (-a) exceeds yellow 'b', OR Hue in green range (H >= 36)
        if (H >= 36.0 and H <= 85.0 and S >= 0.16) or (a < -0.04 and S >= 0.14) or ((-a) > b and S >= 0.16):
            return "Team B"

        # 4. Team A (White / Navy kit)
        # Neutral lightness L >= 0.55, low saturation S < 0.28
        if L >= 0.55 and S < 0.28:
            return "Team A"

        # Margin Resolver:
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
