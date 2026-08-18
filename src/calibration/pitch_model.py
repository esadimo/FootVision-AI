"""
src.calibration.pitch_model — Standard FIFA football pitch 2D top-down drawing engine.
Pitch dimensions: 105m × 68m (standard international dimensions).
"""

import cv2
import numpy as np


# ─── Standard FIFA Pitch Dimensions (meters) ──────────────────────────────────
PITCH_LENGTH   = 105.0   # x-axis: left goal line → right goal line
PITCH_WIDTH    =  68.0   # y-axis: top touchline → bottom touchline

# Penalty box dimensions
PENALTY_BOX_DEPTH  = 16.5   # meters from goal line
PENALTY_BOX_WIDTH  = 40.32  # meters (total width of box)

# Goal area (6-yard box) dimensions
GOAL_AREA_DEPTH = 5.5
GOAL_AREA_WIDTH = 18.32

# Center circle
CENTER_CIRCLE_RADIUS = 9.15
CENTER_SPOT_X = PITCH_LENGTH / 2
CENTER_SPOT_Y = PITCH_WIDTH  / 2

# Penalty spots (from goal line)
PENALTY_SPOT_DIST = 11.0

# Goal dimensions
GOAL_WIDTH = 7.32
GOAL_DEPTH = 2.44   # (not drawn but noted)

# Corner arcs radius
CORNER_ARC_RADIUS = 1.0


# ─── Known Pitch Landmark Coordinates (meters) ────────────────────────────────
# These are the standard correspondence landmarks used for homography calibration.
# Format: (X_meters, Y_meters) where origin is the top-left corner of the pitch.

PITCH_LANDMARKS = {
    "top_left_corner":            (0.0,    0.0),
    "top_right_corner":           (105.0,  0.0),
    "bottom_left_corner":         (0.0,   68.0),
    "bottom_right_corner":        (105.0, 68.0),
    "center_spot":                (52.5,  34.0),
    "halfway_top":                (52.5,   0.0),
    "halfway_bottom":             (52.5,  68.0),
    # Left penalty area corners
    "left_penalty_top_left":      (0.0,   13.84),
    "left_penalty_top_right":     (16.5,  13.84),
    "left_penalty_bottom_left":   (0.0,   54.16),
    "left_penalty_bottom_right":  (16.5,  54.16),
    # Right penalty area corners
    "right_penalty_top_left":     (88.5,  13.84),
    "right_penalty_top_right":    (105.0, 13.84),
    "right_penalty_bottom_left":  (88.5,  54.16),
    "right_penalty_bottom_right": (105.0, 54.16),
    # Left goal area corners
    "left_goalarea_top_left":     (0.0,   24.84),
    "left_goalarea_top_right":    (5.5,   24.84),
    "left_goalarea_bottom_left":  (0.0,   43.16),
    "left_goalarea_bottom_right": (5.5,   43.16),
    # Right goal area corners
    "right_goalarea_top_left":    (99.5,  24.84),
    "right_goalarea_top_right":   (105.0, 24.84),
    "right_goalarea_bottom_left": (99.5,  43.16),
    "right_goalarea_bottom_right":(105.0, 43.16),
    # Center circle intersections with halfway line & axes
    "center_circle_top":          (52.5,  24.85),
    "center_circle_bottom":       (52.5,  43.15),
    "center_circle_left":         (43.35, 34.0),
    "center_circle_right":        (61.65, 34.0),
    # Penalty spots
    "left_penalty_spot":          (11.0,  34.0),
    "right_penalty_spot":         (94.0,  34.0),
}


# ─── Pitch Rendering Engine ───────────────────────────────────────────────────

def draw_pitch(canvas_width: int = 800,
               canvas_height: int = 520,
               background_color: tuple = (34, 139, 34),
               line_color: tuple = (255, 255, 255),
               line_thickness: int = 2) -> np.ndarray:
    """
    Draws a clean top-down 2D football pitch on a canvas.

    Parameters
    ----------
    canvas_width : int
        Pixel width of the canvas (maps to PITCH_LENGTH).
    canvas_height : int
        Pixel height of the canvas (maps to PITCH_WIDTH).
    background_color : tuple
        BGR color for the pitch grass.
    line_color : tuple
        BGR color for pitch markings.
    line_thickness : int
        Pixel thickness of lines.

    Returns
    -------
    np.ndarray
        Pitch canvas in BGR format.
    """
    canvas = np.full((canvas_height, canvas_width, 3), background_color, dtype=np.uint8)

    def m2p(x_m: float, y_m: float):
        """Convert meter coordinates to canvas pixel coordinates."""
        px = int(x_m / PITCH_LENGTH * canvas_width)
        py = int(y_m / PITCH_WIDTH  * canvas_height)
        return (px, py)

    # 1. Outer pitch boundary
    cv2.rectangle(canvas, m2p(0, 0), m2p(PITCH_LENGTH, PITCH_WIDTH), line_color, line_thickness)

    # 2. Halfway line
    cv2.line(canvas, m2p(52.5, 0), m2p(52.5, PITCH_WIDTH), line_color, line_thickness)

    # 3. Center circle
    cx, cy = m2p(CENTER_SPOT_X, CENTER_SPOT_Y)
    r_px = int(CENTER_CIRCLE_RADIUS / PITCH_LENGTH * canvas_width)
    cv2.circle(canvas, (cx, cy), r_px, line_color, line_thickness)

    # 4. Center spot
    cv2.circle(canvas, (cx, cy), 3, line_color, -1)

    # 5. Left penalty area
    lp_y_top = (PITCH_WIDTH - PENALTY_BOX_WIDTH) / 2
    cv2.rectangle(canvas,
                  m2p(0, lp_y_top),
                  m2p(PENALTY_BOX_DEPTH, lp_y_top + PENALTY_BOX_WIDTH),
                  line_color, line_thickness)

    # 6. Right penalty area
    rp_y_top = (PITCH_WIDTH - PENALTY_BOX_WIDTH) / 2
    cv2.rectangle(canvas,
                  m2p(PITCH_LENGTH - PENALTY_BOX_DEPTH, rp_y_top),
                  m2p(PITCH_LENGTH, rp_y_top + PENALTY_BOX_WIDTH),
                  line_color, line_thickness)

    # 7. Left goal area
    lg_y_top = (PITCH_WIDTH - GOAL_AREA_WIDTH) / 2
    cv2.rectangle(canvas,
                  m2p(0, lg_y_top),
                  m2p(GOAL_AREA_DEPTH, lg_y_top + GOAL_AREA_WIDTH),
                  line_color, line_thickness)

    # 8. Right goal area
    rg_y_top = (PITCH_WIDTH - GOAL_AREA_WIDTH) / 2
    cv2.rectangle(canvas,
                  m2p(PITCH_LENGTH - GOAL_AREA_DEPTH, rg_y_top),
                  m2p(PITCH_LENGTH, rg_y_top + GOAL_AREA_WIDTH),
                  line_color, line_thickness)

    # 9. Penalty arcs (partial circles around penalty spots)
    left_spot_px  = m2p(PENALTY_SPOT_DIST, CENTER_SPOT_Y)
    right_spot_px = m2p(PITCH_LENGTH - PENALTY_SPOT_DIST, CENTER_SPOT_Y)
    arc_r_px = int(CENTER_CIRCLE_RADIUS / PITCH_LENGTH * canvas_width)
    cv2.ellipse(canvas, left_spot_px,  (arc_r_px, arc_r_px), 0,  -60,  60, line_color, line_thickness)
    cv2.ellipse(canvas, right_spot_px, (arc_r_px, arc_r_px), 0, 120, 240, line_color, line_thickness)

    # 10. Penalty spots
    cv2.circle(canvas, m2p(PENALTY_SPOT_DIST, CENTER_SPOT_Y), 3, line_color, -1)
    cv2.circle(canvas, m2p(PITCH_LENGTH - PENALTY_SPOT_DIST, CENTER_SPOT_Y), 3, line_color, -1)

    # 11. Corner arcs
    c_r = int(CORNER_ARC_RADIUS / PITCH_LENGTH * canvas_width)
    for (x_m, y_m, angle_start, angle_end) in [
        (0, 0, 0, 90),
        (PITCH_LENGTH, 0, 90, 180),
        (0, PITCH_WIDTH, 270, 360),
        (PITCH_LENGTH, PITCH_WIDTH, 180, 270)
    ]:
        cv2.ellipse(canvas, m2p(x_m, y_m), (c_r, c_r), 0, angle_start, angle_end, line_color, line_thickness)

    # 12. Goals (white boxes beyond touchline)
    goal_y_top = (PITCH_WIDTH - GOAL_WIDTH) / 2
    goal_y_bottom = goal_y_top + GOAL_WIDTH
    left_goal_depth = -2.44 / PITCH_LENGTH * canvas_width
    right_goal_depth = canvas_width + 2.44 / PITCH_LENGTH * canvas_width
    
    cv2.rectangle(canvas, m2p(0, goal_y_top),
                  (max(0, int(left_goal_depth + canvas_width * 0.02)), m2p(0, goal_y_bottom)[1]),
                  line_color, line_thickness)
    cv2.rectangle(canvas, m2p(PITCH_LENGTH, goal_y_top),
                  (min(canvas_width, int(right_goal_depth - canvas_width * 0.02)), m2p(PITCH_LENGTH, goal_y_bottom)[1]),
                  line_color, line_thickness)

    return canvas
