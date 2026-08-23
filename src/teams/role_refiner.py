"""
src.teams.role_refiner — Spatial & Temporal Role Refinement Engine.

After the K-Means color classifier assigns initial labels (Team A, Team B,
Referee, Staff/GK), this module applies a sequence of additional post-processing
rules that use pitch-coordinate geometry and track-level statistics to correct
the most common misclassifications.

Refinement Passes
-----------------
1.  Off-Pitch Staff Filter
    Players whose median pitch X or Y coordinate falls *outside* the pitch
    boundary (with a small tolerance) cannot be outfield players. They are
    tagged as "Staff/GK" regardless of their jersey color.

2.  Goalkeeper Identification
    Each team defends one goal end.  The goalkeeper is the team member who:
      a. Spends the majority of frames at the extreme pitch-X end (left or
         right) that their team defends, AND
      b. Has a very low median pitch-Y displacement (stays near the goal line)
         AND a very narrow X-range (does not leave the box).
    Implementation:
      - The defending sides for each team are discovered automatically from
        the median X of all team-A and team-B detections.
      - For each defending side, the track with the lowest median |foot_x_px|
        distance to that goal line (and within ±40 m pitch range) is the GK.
      - A second GK candidate is found for the other team symmetrically.
    The selected tracks are relabelled "GK Team A" or "GK Team B".

3.  Referee Off-Pitch Promotion
    If a track classified as Referee has a median X or Y that is permanently
    outside the pitch (e.g. running the line on the sideline), it stays as
    Referee (this is fine — assistant referees ARE outside the pitch).
    However, if a Referee track has a high L* (very bright jersey) and its
    position is mostly off-pitch, it may be a waterboy/staff — relabel to
    Staff/GK.

4.  Lock-in Smoothing
    Once a goalkeeper or off-pitch staff ID has been identified it is locked
    in for ALL frames so that a single bad color-vote frame cannot flip the
    label back.

Usage
-----
    from src.teams.role_refiner import RoleRefiner

    refiner = RoleRefiner(pitch_length=105.0, pitch_width=68.0)
    refiner.add_frame(frame_num, track_data)   # during Pass 1
    refined_labels = refiner.compute_refined_labels()
    # refined_labels: {track_id: role_string}
"""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import median, mean
from typing import Dict, List, Optional, Tuple

from src.teams.auto_classifier import (
    ROLE_TEAM_A, ROLE_TEAM_B, ROLE_GK_TEAM_A, ROLE_GK_TEAM_B,
    ROLE_REFEREE, ROLE_STAFF_GK, ROLE_UNKNOWN,
)

# ── Pitch constants (FIFA standard) ──────────────────────────────────────────
_PITCH_LENGTH_DEFAULT = 105.0
_PITCH_WIDTH_DEFAULT  = 68.0

# How many metres outside the pitch boundary a player may sit before being
# considered off-pitch technical staff
_OFF_PITCH_TOLERANCE_M = 4.0

# A track must be visible for at least this many frames to be considered for
# goalkeeper promotion (avoids misidentifying a briefly-visible edge player)
_MIN_GK_TRACK_FRAMES = 30

# The goalkeeper must spend at least this fraction of their frames within the
# penalty-box depth (16.5 m from goal line)
_GK_BOX_FRACTION = 0.40

# Maximum allowed X-spread (metres) for a goalkeeper — stays near the goal
_GK_MAX_X_RANGE = 30.0

# Minimum fraction of frames a track must be off-pitch to be "Staff/GK"
_OFF_PITCH_STAFF_FRACTION = 0.70


class TrackRecord:
    """Accumulates per-frame observations for one ByteTrack ID."""

    def __init__(self, track_id: int):
        self.track_id = track_id
        # Raw color votes from K-Means classifier per frame
        self.color_votes: List[str] = []
        # Pitch-coordinate observations (meters)
        self.pitch_xs: List[float] = []
        self.pitch_ys: List[float] = []
        # Foot pixel observations (for on-screen annotation)
        self.foot_xs: List[float] = []
        self.foot_ys: List[float] = []
        # Bounding box heights (rough proxy for distance from camera)
        self.box_heights: List[int] = []

    # ── Derived properties ───────────────────────────────────────────────────

    @property
    def n_frames(self) -> int:
        return len(self.color_votes)

    @property
    def majority_label(self) -> str:
        """The most common color-classifier label across all frames."""
        if not self.color_votes:
            return ROLE_UNKNOWN
        from collections import Counter
        votes = [v for v in self.color_votes if v != ROLE_UNKNOWN]
        if not votes:
            return ROLE_UNKNOWN
        return Counter(votes).most_common(1)[0][0]

    @property
    def median_pitch_x(self) -> Optional[float]:
        return median(self.pitch_xs) if self.pitch_xs else None

    @property
    def median_pitch_y(self) -> Optional[float]:
        return median(self.pitch_ys) if self.pitch_ys else None

    @property
    def x_range(self) -> float:
        if not self.pitch_xs:
            return 0.0
        return max(self.pitch_xs) - min(self.pitch_xs)

    @property
    def off_pitch_fraction(self) -> float:
        """Fraction of frames where this track is clearly outside the pitch."""
        if not self.pitch_xs:
            return 0.0
        off = 0
        for x, y in zip(self.pitch_xs, self.pitch_ys):
            if (x < -_OFF_PITCH_TOLERANCE_M or x > _PITCH_LENGTH_DEFAULT + _OFF_PITCH_TOLERANCE_M or
                    y < -_OFF_PITCH_TOLERANCE_M or y > _PITCH_WIDTH_DEFAULT + _OFF_PITCH_TOLERANCE_M):
                off += 1
        return off / len(self.pitch_xs)


class RoleRefiner:
    """
    Post-processing engine that corrects team/GK/staff labels using
    pitch-coordinate geometry and track-level temporal statistics.

    Call add_frame() during the tracking loop (Pass 1), then
    compute_refined_labels() once the entire sequence has been processed.
    """

    def __init__(
        self,
        pitch_length: float = _PITCH_LENGTH_DEFAULT,
        pitch_width:  float = _PITCH_WIDTH_DEFAULT,
    ):
        self._pitch_length = pitch_length
        self._pitch_width  = pitch_width
        self._tracks: Dict[int, TrackRecord] = {}

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def add_frame(
        self,
        track_id:   int,
        color_vote: str,
        pitch_x:    float,
        pitch_y:    float,
        foot_x_px:  float,
        foot_y_px:  float,
        box_height: int = 0,
    ) -> None:
        """Record one frame of observations for the given track."""
        if track_id not in self._tracks:
            self._tracks[track_id] = TrackRecord(track_id)
        rec = self._tracks[track_id]
        rec.color_votes.append(color_vote)
        rec.pitch_xs.append(pitch_x)
        rec.pitch_ys.append(pitch_y)
        rec.foot_xs.append(foot_x_px)
        rec.foot_ys.append(foot_y_px)
        rec.box_heights.append(box_height)

    # ── Refinement ────────────────────────────────────────────────────────────

    def compute_refined_labels(self) -> Dict[int, str]:
        """
        Analyse all accumulated track data and return a corrected label for
        every track_id.

        Returns
        -------
        dict : {track_id: refined_label_string}
        """
        refined: Dict[int, str] = {}

        # Step A: seed each track's label from majority color vote
        for tid, rec in self._tracks.items():
            refined[tid] = rec.majority_label

        # Step B: off-pitch staff promotion
        self._apply_off_pitch_filter(refined)

        # Step C: goalkeeper identification
        self._apply_goalkeeper_detection(refined)

        return refined

    # ── Pass B: Off-pitch Staff Filter ───────────────────────────────────────

    def _apply_off_pitch_filter(self, labels: Dict[int, str]) -> None:
        """
        Tracks that spend the majority of their time clearly outside the pitch
        are relabelled Staff/GK — they cannot be outfield players.
        This catches cameramen, waterboys, ball-boys, and sideline staff.
        """
        for tid, rec in self._tracks.items():
            if rec.n_frames < 5:
                continue
            current = labels.get(tid, ROLE_UNKNOWN)
            # Already tagged as staff or unknown — nothing to change
            if current in (ROLE_STAFF_GK, ROLE_UNKNOWN):
                continue
            # Referees near the touchline are allowed to be off-pitch
            if current == ROLE_REFEREE:
                # Only promote to Staff/GK if WAY off-pitch AND very bright
                # (waterboy wearing white-ish bib sometimes looks like Ref)
                if rec.off_pitch_fraction >= _OFF_PITCH_STAFF_FRACTION:
                    labels[tid] = ROLE_STAFF_GK
                continue
            # Outfield players / GKs — must be mostly on-pitch
            if rec.off_pitch_fraction >= _OFF_PITCH_STAFF_FRACTION:
                labels[tid] = ROLE_STAFF_GK

    # ── Pass C: Goalkeeper Detection ─────────────────────────────────────────

    def _apply_goalkeeper_detection(self, labels: Dict[int, str]) -> None:
        """
        Find the most likely goalkeeper for each team.

        Algorithm
        ---------
        1. Determine which pitch half each team defends by comparing the
           median X of all Team A vs Team B detections.
           Team defending the left goal (X≈0)  → their GK is at small X.
           Team defending the right goal (X≈L) → their GK is at large X.
        2. Among tracks labelled as that team, select the one with:
           - n_frames >= _MIN_GK_TRACK_FRAMES
           - x_range  <= _GK_MAX_X_RANGE     (stays near the goal line)
           - fraction of frames in the goal-box area >= _GK_BOX_FRACTION
        3. If multiple candidates exist, pick the one closest to the goal line
           in median pitch_x.
        """

        # Collect Team A and Team B track records
        team_tracks: Dict[str, List[TrackRecord]] = {
            ROLE_TEAM_A: [],
            ROLE_TEAM_B: [],
        }

        for tid, label in labels.items():
            if label in team_tracks and tid in self._tracks:
                team_tracks[label].append(self._tracks[tid])

        # Compute aggregate median X for each team to determine defending end
        team_median_x: Dict[str, Optional[float]] = {ROLE_TEAM_A: None, ROLE_TEAM_B: None}
        for team, recs in team_tracks.items():
            all_xs = [x for r in recs for x in r.pitch_xs]
            if all_xs:
                team_median_x[team] = median(all_xs)

        mx_a = team_median_x[ROLE_TEAM_A]
        mx_b = team_median_x[ROLE_TEAM_B]

        if mx_a is None or mx_b is None:
            # Not enough data — skip GK detection
            return

        # Which team defends the left goal (X≈0)?
        if mx_a < mx_b:
            left_team, right_team = ROLE_TEAM_A, ROLE_TEAM_B
            gk_label_left,  gk_label_right = ROLE_GK_TEAM_A, ROLE_GK_TEAM_B
        else:
            left_team, right_team = ROLE_TEAM_B, ROLE_TEAM_A
            gk_label_left,  gk_label_right = ROLE_GK_TEAM_B, ROLE_GK_TEAM_A

        gk_left  = self._find_best_gk(team_tracks[left_team],  goal_x=0.0)
        gk_right = self._find_best_gk(team_tracks[right_team], goal_x=self._pitch_length)

        if gk_left is not None:
            labels[gk_left.track_id] = gk_label_left

        if gk_right is not None:
            labels[gk_right.track_id] = gk_label_right

    def _find_best_gk(
        self,
        candidates: List[TrackRecord],
        goal_x: float,
    ) -> Optional[TrackRecord]:
        """
        From a list of same-team track records, select the one most likely to
        be the goalkeeper defending the goal at `goal_x`.

        Parameters
        ----------
        candidates : list of TrackRecord
        goal_x : 0.0 (left goal) or pitch_length (right goal)

        Returns
        -------
        The best-matching TrackRecord, or None if no suitable candidate found.
        """
        scored: List[Tuple[float, TrackRecord]] = []

        # Width of penalty box (40.32 m). GK must be within this Y-band centred
        # on the pitch centre-line.
        half_box_y = 40.32 / 2.0
        centre_y   = self._pitch_width / 2.0
        box_y_min  = centre_y - half_box_y
        box_y_max  = centre_y + half_box_y

        # X depth of penalty area from goal line
        penalty_depth = 16.5

        for rec in candidates:
            if rec.n_frames < _MIN_GK_TRACK_FRAMES:
                continue
            if rec.x_range > _GK_MAX_X_RANGE:
                continue
            if rec.median_pitch_x is None:
                continue

            # Fraction of frames inside the penalty box on the correct side
            box_frames = 0
            for x, y in zip(rec.pitch_xs, rec.pitch_ys):
                in_y = box_y_min <= y <= box_y_max
                if goal_x == 0.0:
                    in_x = x <= penalty_depth
                else:
                    in_x = x >= (self._pitch_length - penalty_depth)
                if in_x and in_y:
                    box_frames += 1

            box_frac = box_frames / rec.n_frames
            if box_frac < _GK_BOX_FRACTION:
                continue

            # Score = proximity to goal line (closer = better GK candidate)
            dist_to_goal = abs(rec.median_pitch_x - goal_x)
            # Lower distance → higher (negative) score for min-sort
            scored.append((dist_to_goal, rec))

        if not scored:
            return None

        scored.sort(key=lambda t: t[0])
        return scored[0][1]


# ── Convenience accessor exported for downstream modules ─────────────────────

def is_goalkeeper_label(label: str) -> bool:
    """Returns True if the label is a goalkeeper role."""
    return label in (ROLE_GK_TEAM_A, ROLE_GK_TEAM_B)


def is_team_player(label: str) -> bool:
    """True for Team A, Team B, GK Team A, GK Team B — i.e. active football participants."""
    return label in (ROLE_TEAM_A, ROLE_TEAM_B, ROLE_GK_TEAM_A, ROLE_GK_TEAM_B)
