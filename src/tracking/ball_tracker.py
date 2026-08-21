"""
src.tracking.ball_tracker — Tracks the ball and handles missing states via interpolation.
"""

import collections
from typing import Tuple, Optional, List

class BallTracker:
    def __init__(self, max_gap_frames: int = 15, history_len: int = 45):
        """
        Tracks a high-velocity, frequently occluded sports ball.
        
        Parameters
        ----------
        max_gap_frames : int
            Maximum number of consecutive missing frames to interpolate before dropping.
        history_len : int
            Number of recent frames to store for trajectory trails and velocity calculation.
        """
        self.max_gap_frames = max_gap_frames
        self.history = collections.deque(maxlen=history_len)
        self.missing_frames = 0
        self.current_frame = 0

    def update(self, detection: Optional[Tuple[float, float, float, float]]) -> Tuple[Optional[Tuple[float, float]], bool]:
        """
        Updates the tracker with a new bounding box detection (or None).
        
        Parameters
        ----------
        detection : (x1, y1, x2, y2) or None
            The bounding box of the detected ball in the current frame.
            
        Returns
        -------
        (pos, is_interpolated)
            pos : (cx, cy) center coordinate of the ball, or None if lost.
            is_interpolated : True if the position was guessed via kinematics.
        """
        self.current_frame += 1

        if detection is not None:
            # Valid detection received
            x1, y1, x2, y2 = detection
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            pos = (cx, cy)
            
            self.history.append({
                'frame': self.current_frame, 
                'pos': pos, 
                'interpolated': False
            })
            self.missing_frames = 0
            return pos, False
            
        else:
            # Detection is missing this frame
            self.missing_frames += 1
            
            if self.missing_frames <= self.max_gap_frames and len(self.history) >= 2:
                # Attempt to interpolate based on recent valid history
                valid_hist = [h for h in self.history if not h['interpolated'] and h['pos'] is not None]
                
                if len(valid_hist) >= 2:
                    p1 = valid_hist[-2]['pos']
                    f1 = valid_hist[-2]['frame']
                    p2 = valid_hist[-1]['pos']
                    f2 = valid_hist[-1]['frame']
                    
                    df = f2 - f1
                    if df > 0:
                        # Linear velocity estimation
                        vx = (p2[0] - p1[0]) / df
                        vy = (p2[1] - p1[1]) / df
                        
                        # Apply velocity to extrapolate
                        # Note: self.missing_frames represents frames since last valid detection
                        # Wait, p2 is the *last valid*. So frames since p2 is self.missing_frames.
                        extrap_x = p2[0] + vx * self.missing_frames
                        extrap_y = p2[1] + vy * self.missing_frames
                        
                        # Simple dampening to prevent flying off to infinity
                        dampening = 0.95 ** self.missing_frames
                        
                        # Refined extrapolation with dampening
                        extrap_x = p2[0] + (vx * self.missing_frames * dampening)
                        extrap_y = p2[1] + (vy * self.missing_frames * dampening)
                        
                        pos = (extrap_x, extrap_y)
                        self.history.append({
                            'frame': self.current_frame, 
                            'pos': pos, 
                            'interpolated': True
                        })
                        return pos, True

            # If we reach here, we've lost the ball or can't interpolate
            self.history.append({
                'frame': self.current_frame, 
                'pos': None, 
                'interpolated': False
            })
            return None, False

    def get_trail(self, length: int = 12) -> List[Tuple[int, Tuple[float, float]]]:
        """Returns recent consecutive historical trail of the ball (frame, pos)."""
        trail = []
        recent = list(self.history)[-length:]
        for item in recent:
            if item['pos'] is not None:
                trail.append((item['frame'], item['pos']))
            else:
                trail.clear()  # Reset trail on lost frames so lines don't cross gaps
        return trail
