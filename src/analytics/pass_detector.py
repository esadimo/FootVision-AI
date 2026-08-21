"""
src.analytics.pass_detector — Identifies passing events and turnovers from possession data.
"""

import csv
from typing import List, Dict, Any

class PassDetector:
    def __init__(self, touch_threshold_m: float = 2.5):
        """
        Detects passes and turnovers based on changes in ball control.
        
        Parameters
        ----------
        touch_threshold_m : float
            Distance in meters within which a player is considered to be touching/controlling the ball.
        """
        self.touch_threshold_m = touch_threshold_m

    def detect_passes(self, possession_csv: str, player_csv: str) -> List[Dict[str, Any]]:
        """
        Scans the possession and player CSVs to extract discrete passing and turnover events.
        
        Returns
        -------
        events : List[Dict]
            List of event dictionaries with keys:
            ['start_frame', 'end_frame', 'from_player', 'to_player', 
             'from_team', 'to_team', 'event_type']
        """
        # 1. Map track_id to team_label for fast lookup
        player_teams = {}
        with open(player_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                player_teams[int(row['track_id'])] = row['team_label']

        events = []
        last_touch_pid = None
        last_touch_team = None
        last_touch_frame = -1

        # 2. Iterate through possession timeline
        with open(possession_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame = int(row['frame_number'])
                pid = int(row['closest_player_id'])
                dist = float(row['closest_distance_m'])

                if pid != -1 and dist <= self.touch_threshold_m:
                    team = player_teams.get(pid, "Unknown")
                    if team not in ["Team A", "Team B"]:
                        continue
                        
                    if last_touch_pid is None:
                        # First touch of the sequence
                        last_touch_pid = pid
                        last_touch_team = team
                        last_touch_frame = frame
                    elif pid != last_touch_pid:
                        # Control has transferred to a new player!
                        event_type = "Pass" if team == last_touch_team else "Turnover"
                        
                        # Only record if there is some frame gap (avoids instant micro-transfers in scrums)
                        if frame > last_touch_frame:
                            events.append({
                                'start_frame': last_touch_frame,  # Frame the ball was released
                                'end_frame': frame,               # Frame the ball was received
                                'from_player': last_touch_pid,
                                'to_player': pid,
                                'from_team': last_touch_team,
                                'to_team': team,
                                'event_type': event_type
                            })
                        
                        # Update the active controller
                        last_touch_pid = pid
                        last_touch_team = team
                        last_touch_frame = frame
                    else:
                        # Same player is still dribbling/controlling
                        last_touch_frame = frame
                        
        return events
