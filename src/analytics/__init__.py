"""
src.analytics — Football analytics modules.

Modules
-------
possession  : Estimate ball possession for each frame and aggregate team totals.
passes      : Detect completed passes, turnovers and interceptions from
              possession-state sequences and ball trajectories.
distance    : Calculate visible distance covered per player using smoothed tracks.
shape       : Compute team shape metrics — width, depth, compactness, convex hull.
formations  : Estimate team formations by clustering average player positions
              into defensive, midfield and attacking lines.
"""
