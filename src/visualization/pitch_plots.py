"""
src.visualization.pitch_plots — Composite broadcast + 2D radar frame renderer.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional

from src.calibration.pitch_model import draw_pitch, PITCH_LENGTH, PITCH_WIDTH


# ─── Team Colors for 2D Radar Dots ───────────────────────────────────────────
TEAM_DOT_COLORS: Dict[str, Tuple[int, int, int]] = {
    "Team A":    (200, 200, 255),   # Light red/white (navy-white kit)
    "Team B":    (50,  200,  50),   # Green
    "Referee":   (0,   215, 255),   # Yellow
    "Staff/GK":  (128, 128, 128),   # Gray
    "Unknown":   (180, 180, 180),   # Light gray
}

TEAM_TEXT_COLORS: Dict[str, Tuple[int, int, int]] = {
    "Team A":    (200, 200, 255),
    "Team B":    (80,  255,  80),
    "Referee":   (0,   215, 255),
    "Staff/GK":  (150, 150, 150),
    "Unknown":   (200, 200, 200),
}

# Radar canvas dimensions
RADAR_W = 700
RADAR_H = 460
RADAR_MARGIN = 20   # pixels of padding inside radar canvas

# Player dot radius on radar
DOT_RADIUS = 6
DOT_BORDER  = 1


def _meters_to_radar_px(x_m: float, y_m: float,
                         w: int = RADAR_W, h: int = RADAR_H,
                         margin: int = RADAR_MARGIN) -> Tuple[int, int]:
    """Converts pitch meter coordinates to radar canvas pixel coordinates."""
    usable_w = w - 2 * margin
    usable_h = h - 2 * margin
    px = int(margin + np.clip(x_m / PITCH_LENGTH, 0, 1) * usable_w)
    py = int(margin + np.clip(y_m / PITCH_WIDTH,  0, 1) * usable_h)
    return (px, py)


def draw_pitch_radar(
    player_positions: List[Dict],
    canvas_w: int = RADAR_W,
    canvas_h: int = RADAR_H,
    margin: int = RADAR_MARGIN,
    frame_number: Optional[int] = None,
    timestamp: Optional[float] = None
) -> np.ndarray:
    """
    Draws the 2D tactical radar frame: top-down pitch with colored player dots.

    Parameters
    ----------
    player_positions : list of dicts
        Each dict must have keys:
            'track_id'   : int
            'team_label' : str (e.g. "Team A", "Team B", "Referee", ...)
            'pitch_x'    : float (pitch X coordinate in meters)
            'pitch_y'    : float (pitch Y coordinate in meters)
    canvas_w, canvas_h : int
        Dimensions of the radar canvas.
    margin : int
        Padding in pixels around the radar drawing area.
    frame_number : int, optional
        Frame number for HUD.
    timestamp : float, optional
        Timestamp in seconds for HUD.

    Returns
    -------
    np.ndarray
        BGR radar canvas.
    """
    # Draw base pitch
    radar = draw_pitch(canvas_w, canvas_h, line_thickness=1)

    # Draw player dots
    for p in player_positions:
        x_m = p.get("pitch_x", None)
        y_m = p.get("pitch_y", None)
        label = p.get("team_label", "Unknown")
        tid = p.get("track_id", -1)

        if x_m is None or y_m is None:
            continue
        if not (0 <= x_m <= PITCH_LENGTH and 0 <= y_m <= PITCH_WIDTH):
            continue   # Out of pitch bounds — skip

        px, py = _meters_to_radar_px(x_m, y_m, canvas_w, canvas_h, margin)
        color = TEAM_DOT_COLORS.get(label, TEAM_DOT_COLORS["Unknown"])

        # Draw dot with white border
        cv2.circle(radar, (px, py), DOT_RADIUS + DOT_BORDER, (255, 255, 255), -1)
        cv2.circle(radar, (px, py), DOT_RADIUS, color, -1)

        # Track ID label
        cv2.putText(radar, str(tid), (px + DOT_RADIUS + 2, py + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1, cv2.LINE_AA)

    # HUD: frame / time
    if frame_number is not None or timestamp is not None:
        hud_parts = []
        if frame_number is not None:
            hud_parts.append(f"Frame {frame_number}")
        if timestamp is not None:
            hud_parts.append(f"{timestamp:.2f}s")
        hud_text = "  |  ".join(hud_parts)
        cv2.putText(radar, hud_text, (margin, canvas_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

    # Legend
    _draw_legend(radar, canvas_w, canvas_h, margin)

    return radar


def _draw_legend(canvas: np.ndarray, w: int, h: int, margin: int) -> None:
    """Draws a compact legend in the top-left corner of the radar."""
    legend_items = [
        ("Team A",   TEAM_DOT_COLORS["Team A"]),
        ("Team B",   TEAM_DOT_COLORS["Team B"]),
        ("Referee",  TEAM_DOT_COLORS["Referee"]),
        ("GK/Staff", TEAM_DOT_COLORS["Staff/GK"]),
    ]
    x0, y0 = margin + 4, margin + 8
    for i, (name, color) in enumerate(legend_items):
        yy = y0 + i * 16
        cv2.circle(canvas, (x0, yy), 5, color, -1)
        cv2.putText(canvas, name, (x0 + 10, yy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)


def build_composite_frame(
    broadcast_frame: np.ndarray,
    radar_frame: np.ndarray,
    target_height: int = 540,
    divider_width: int = 4
) -> np.ndarray:
    """
    Builds a side-by-side composite frame: broadcast (left) | radar (right).
    Both sides are scaled to the same target_height.

    Parameters
    ----------
    broadcast_frame : np.ndarray
        Annotated broadcast frame (BGR).
    radar_frame : np.ndarray
        Radar canvas from draw_pitch_radar() (BGR).
    target_height : int
        Target pixel height for both panels.
    divider_width : int
        Width of the white divider line between panels.

    Returns
    -------
    np.ndarray
        Horizontally stacked composite frame.
    """
    # Scale broadcast to target_height
    bh, bw = broadcast_frame.shape[:2]
    b_scale = target_height / bh
    broadcast_resized = cv2.resize(broadcast_frame,
                                   (int(bw * b_scale), target_height),
                                   interpolation=cv2.INTER_LINEAR)

    # Scale radar to target_height
    rh, rw = radar_frame.shape[:2]
    r_scale = target_height / rh
    radar_resized = cv2.resize(radar_frame,
                               (int(rw * r_scale), target_height),
                               interpolation=cv2.INTER_LINEAR)

    # White divider
    divider = np.ones((target_height, divider_width, 3), dtype=np.uint8) * 220

    # Concatenate horizontally
    composite = np.hstack([broadcast_resized, divider, radar_resized])
    return composite
