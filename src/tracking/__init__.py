"""
src.tracking — Multi-object tracking modules.

Modules
-------
player_tracker      : Wrap a tracker (ByteTrack / BoT-SORT) and assign temporary
                      track IDs to player detections across consecutive frames.
ball_tracker        : Dedicated ball tracker with state estimation and gap filling.
trajectory_filter   : Smooth raw tracked positions, remove outliers and interpolate
                      short gaps using Kalman filtering or Savitzky-Golay.
"""
