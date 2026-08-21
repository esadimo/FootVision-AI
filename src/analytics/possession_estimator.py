"""
src.analytics.possession_estimator — Determines which team has possession based on ball proximity.
"""

from typing import Dict, Tuple, Optional
import numpy as np

class PossessionEstimator:
    def __init__(self, proximity_threshold_m: float = 2.0, smoothing_frames: int = 3):
        """
        Estimates ball possession using pitch metric coordinates.
        
        Parameters
        ----------
        proximity_threshold_m : float
            Maximum distance in meters for a player to be considered in "control" or "touching" the ball.
        smoothing_frames : int
            Number of consecutive frames a team must be closest to the ball to officially steal possession.
            Prevents flickering when a defender makes a split-second tackle but doesn't win the ball.
        """
        self.proximity_threshold_m = proximity_threshold_m
        self.smoothing_frames = smoothing_frames
        
        self.current_possession = "None"  # "Team A", "Team B", or "None"
        self.frame_counts = {"Team A": 0, "Team B": 0, "None": 0}
        
        # State machine for stealing possession
        self._candidate_team = None
        self._candidate_frames = 0

    def update(self, 
               ball_pos_m: Optional[Tuple[float, float]], 
               players: Dict[int, Tuple[str, float, float]]) -> Tuple[str, Optional[int], float]:
        """
        Updates the possession state for the current frame.
        
        Parameters
        ----------
        ball_pos_m : (x_m, y_m) or None
            The pitch coordinates of the ball in meters.
        players : dict
            {track_id: (team_label, x_m, y_m)}
            
        Returns
        -------
        (current_possession_team, closest_track_id, closest_distance)
        """
        closest_id = None
        closest_dist = float('inf')
        closest_team = None
        
        if ball_pos_m is not None and len(players) > 0:
            bx, by = ball_pos_m
            
            # Find closest player
            for tid, (team, px, py) in players.items():
                if team not in ["Team A", "Team B"]:
                    continue # Ignore referees or staff for possession
                    
                dist = np.hypot(bx - px, by - py)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_id = tid
                    closest_team = team
        
        # Update Possession State Machine
        if closest_dist <= self.proximity_threshold_m and closest_team is not None:
            # Someone is touching the ball
            if closest_team == self.current_possession:
                # Retaining possession, reset steal candidate
                self._candidate_team = None
                self._candidate_frames = 0
            else:
                # Opponent is touching the ball
                if self._candidate_team == closest_team:
                    self._candidate_frames += 1
                    if self._candidate_frames >= self.smoothing_frames:
                        # Possession officially changes hands
                        self.current_possession = closest_team
                        self._candidate_team = None
                        self._candidate_frames = 0
                else:
                    self._candidate_team = closest_team
                    self._candidate_frames = 1
        else:
            # Ball is loose (pass in transit). Possession remains with the last team.
            self._candidate_team = None
            self._candidate_frames = 0
            
        self.frame_counts[self.current_possession] += 1
        
        return self.current_possession, closest_id, closest_dist

    def get_possession_percentages(self) -> Dict[str, float]:
        """Returns the percentage of possession for Team A and Team B."""
        total = self.frame_counts["Team A"] + self.frame_counts["Team B"]
        if total == 0:
            return {"Team A": 0.0, "Team B": 0.0}
            
        return {
            "Team A": (self.frame_counts["Team A"] / total) * 100.0,
            "Team B": (self.frame_counts["Team B"] / total) * 100.0
        }
