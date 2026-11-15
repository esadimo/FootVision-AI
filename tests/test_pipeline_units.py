import numpy as np
import pandas as pd
import pytest

from src.analytics.match_analytics import rts_smooth
from src.calibration.camera_model import CameraParams, decompose, rot_from_ptr
from src.calibration.line_tracker import template_points
from src.teams.team_clusterer import assign_teams
from src.tracking.ball_trajectory import solve_ball_trajectory
from src.tracking.stitching import StitchParams, stitch_tracklets

import cv2


def _camera():
    R = rot_from_ptr(np.deg2rad(20), np.deg2rad(14))
    return CameraParams(cv2.Rodrigues(R)[0].ravel(), 3000.0, np.array([52.5, 100.0, -14.0]))


def test_camera_roundtrip_and_decompose():
    cam = _camera()
    H = cam.H()
    # a pitch point projects to the image and back
    G = cam.G()
    p = np.array([80.0, 30.0, 1.0])
    q = G @ p; q /= q[2]
    back = H @ q; back /= back[2]
    assert np.allclose(back[:2], p[:2], atol=1e-6)
    d = decompose(H)
    assert d.f == pytest.approx(3000.0, rel=1e-3)
    assert np.allclose(d.C, cam.C, atol=1e-2)


def test_template_linear_in_dimensions():
    a, b, c = template_points(0.5, 105, 68), template_points(0.5, 106, 68), template_points(0.5, 107, 69)
    assert a.shape == b.shape == c.shape
    assert np.abs((c - b) - (b - a) - (template_points(0.5, 105, 69) - a)).max() < 1e-9


def test_ball_trajectory_rejects_decoys_and_fills_gaps():
    n = 100
    rows = []
    for f in range(1, n + 1):
        if f % 7 == 0:          # missed detections
            continue
        rows.append((f, 100 + 5 * f, 500.0, 0.6))
        rows.append((f, 1500.0, 200.0 + (f % 3) * 300, 0.3))   # jumping decoy
    c = pd.DataFrame(rows, columns=["frame", "x", "y", "conf"])
    t = solve_ball_trajectory(c, n)
    good = t.dropna(subset=["x"])
    assert len(good) >= 98
    assert np.abs(good.x - (100 + 5 * good.frame)).max() < 1e-6
    assert set(t.loc[t.frame % 7 == 0, "source"]) <= {"interp", ""}


def test_rts_smoother_reduces_noise():
    rng = np.random.default_rng(0)
    tt = np.arange(200) / 25
    truth = np.stack([10 + 3 * tt, 20 + 1 * tt], 1)
    z = truth + rng.normal(0, 0.4, truth.shape)
    xs = rts_smooth(z, 1 / 25)
    assert np.abs(xs[:, :2] - truth).mean() < 0.5 * np.abs(z - truth).mean()
    assert np.median(xs[20:-20, 2]) == pytest.approx(3.0, abs=0.3)


def _frag(tid, frames, x0, vx):
    f = np.array(frames)
    return pd.DataFrame({"frame": f, "track_id": tid, "x1": 0, "y1": 0, "x2": 10, "y2": 30,
                         "pitch_x": x0 + vx * (f - f[0]) / 25, "pitch_y": 30.0})


def test_stitching_joins_reachable_same_team_only():
    a = _frag(1, range(1, 50), 40.0, 4.0)                 # ends near x = 47.7
    b = _frag(2, range(70, 120), 51.0, 4.0)               # starts 0.8 s later, ~3 m ahead
    c = _frag(3, range(70, 120), 51.2, 4.0).assign(track_id=3)
    t = pd.concat([a, b])
    out = stitch_tracklets(t, {}, labels={1: "player:A", 2: "player:A"}, p=StitchParams())
    assert out.track_id.nunique() == 1
    out2 = stitch_tracklets(pd.concat([a, c]), {}, labels={1: "player:A", 3: "player:B"}, p=StitchParams())
    assert out2.track_id.nunique() == 2


def test_team_clusterer_ignores_officials_cluster():
    rng = np.random.default_rng(1)
    rows = []
    dim = 18 * 6 + 6
    base = {k: rng.random(dim) for k in ("A", "B", "ref")}
    base["B"] = base["A"] + 0.35 * rng.standard_normal(dim) / np.sqrt(dim) * 4  # teams fairly close
    truth = {}
    tid = 0
    for k, n_tracks in (("A", 10), ("B", 10), ("ref", 4)):
        for _ in range(n_tracks):
            tid += 1; truth[tid] = k
            for _ in range(8):
                rows.append({"track_id": tid, "feat": base[k] + 0.02 * rng.standard_normal(dim)})
    out = assign_teams(pd.DataFrame(rows))
    a_ids = [t for t, k in truth.items() if k == "A"]; b_ids = [t for t, k in truth.items() if k == "B"]
    ta = {out[t]["team"] for t in a_ids}; tb = {out[t]["team"] for t in b_ids}
    assert len(ta) == 1 and len(tb) == 1 and ta != tb and "" not in ta | tb
    assert all(out[t]["team"] == "" for t, k in truth.items() if k == "ref")


def test_grass_mask_survives_lines_cutting_the_pitch():
    from src.calibration.line_tracker import grass_mask
    img = np.zeros((1080, 1920, 3), np.uint8)
    img[300:] = (40, 140, 40)                     # pitch (BGR green) below the stands
    img[300:700, :] = (30, 95, 30)                # darker, shaded part of the pitch
    for x in (500, 1100, 1500):                   # painted lines crossing the whole pitch
        cv2.line(img, (x, 300), (x - 200, 1079), (255, 255, 255), 4)
    cv2.line(img, (0, 700), (1919, 700), (255, 255, 255), 4)
    g = grass_mask(img) > 0
    assert g[320:].mean() > 0.95 and g[:280].mean() < 0.02


def test_camera_centre_bounds():
    from src.calibration.camera_model import plausible_centre
    assert plausible_centre([52.5, 100.0, -14.0])
    assert not plausible_centre([64.8, 111.4, 0.0])          # at ground level (degenerate)
    assert not plausible_centre([89.9, 14746.1, -659.4])      # kilometres away


def test_person_height_score_separates_right_and_wrong_cameras():
    from src.calibration.camera_model import person_height_score
    cam = _camera()
    R, K = cam.R(), cam.K()
    boxes = []
    for x, y in [(45, 20), (60, 30), (52, 45), (70, 15), (40, 35)]:       # people on the pitch
        img = []
        for z in (0.0, -1.8):
            q = K @ (R @ (np.array([x, y, z]) - cam.C)); img.append(q[:2] / q[2])
        (fx, fy), (_, hy) = img
        bh = fy - hy
        boxes.append([fx - 0.2 * bh, hy, fx + 0.2 * bh, fy])
    boxes = np.array(boxes)
    assert person_height_score(cam, boxes) == 1.0
    wrong = cam.copy(); wrong.C = np.array([52.5, 100.0, -5.0])         # camera far too low
    assert person_height_score(wrong, boxes) < 0.5
    assert np.isnan(person_height_score(cam, boxes[:2]))                 # too few boxes


def _track(tid, frames, x, y=34.0, cls=0, team=""):
    return pd.DataFrame({"frame": frames, "track_id": tid, "pitch_x": x, "pitch_y": y, "cls": cls,
                         "conf": 0.8, "team": team})


def test_refine_goalkeepers_promotes_teamless_track_near_goal():
    from src.pipeline.roles_teams import goalkeeper_teams, refine_goalkeepers
    f = np.arange(1, 101)
    t = pd.concat([_track(1, f, 4.0),                       # unassigned kit, at the left goal → keeper
                   _track(2, f, 50.0),                      # unassigned kit, midfield → not a keeper
                   _track(3, f, 30.0, team="A"), _track(4, f, 40.0, team="A"),
                   _track(5, f, 60.0, team="B"), _track(6, f, 70.0, team="B")])
    roles = {i: "player" for i in range(1, 7)}
    team_of = {3: "A", 4: "A", 5: "B", 6: "B", 1: "", 2: ""}
    roles = refine_goalkeepers(t, roles, team_of)
    assert roles[1] == "goalkeeper" and roles[2] == "player"
    teams = goalkeeper_teams(t, roles, {k: v for k, v in team_of.items() if roles[k] == "player"})
    assert teams[1] == "A"   # team A sits deeper toward x = 0


def test_refine_goalkeepers_one_keeper_per_goal_at_a_time():
    from src.pipeline.roles_teams import refine_goalkeepers
    f = np.arange(1, 101)
    t = pd.concat([_track(1, f, 3.0), _track(2, f, 10.0)])   # two simultaneous unassigned tracks, same goal
    roles = refine_goalkeepers(t, {1: "player", 2: "player"}, {1: "", 2: ""})
    assert sorted(roles.values()) == ["goalkeeper", "player"]


def test_split_points_cuts_sustained_team_flip_only():
    from src.teams.track_splitter import SplitParams, split_points
    rng = np.random.default_rng(0)
    f = np.arange(0, 300, 3)
    flip = np.r_[np.full(50, -0.3), np.full(50, 0.3)] + rng.normal(0, 0.1, 100)
    cuts = split_points(f, flip, SplitParams())
    assert len(cuts) == 1 and abs(cuts[0] - 150) <= 6
    blip = np.full(100, -0.3); blip[40:44] = 0.4          # 4-sample blip (occlusion) → no cut
    assert split_points(f, blip, SplitParams()) == []


def test_possession_lapses_after_max_hold():
    from src.analytics.match_analytics import EventParams, possession_and_events
    ctrl = pd.DataFrame({"frame": np.arange(1, 11), "controller": 7, "ctrl_team": "A", "method": "image"})
    ball = pd.DataFrame({"frame": np.arange(1, 201), "x": 1.0, "y": 1.0})
    poss, _ = possession_and_events(ctrl, ball, 200, EventParams(max_hold=50))
    assert (poss.possession.iloc[:60] == "A").all() and (poss.possession.iloc[60:] == "").all()
    poss, _ = possession_and_events(ctrl, ball, 200, EventParams(max_hold=0))
    assert (poss.possession == "A").all()


def test_refine_goalkeepers_uses_kit_distance_when_clustered_into_a_team():
    from src.pipeline.roles_teams import refine_goalkeepers
    f = np.arange(1, 101)
    t = pd.concat([_track(1, f, 4.0, team="A"), _track(2, f, 8.0, team="A")])
    team_of = {1: "A", 2: "A"}
    roles = refine_goalkeepers(t, {1: "player", 2: "player"}, team_of, kit_dist={1: 3.5, 2: 0.8})
    assert roles == {1: "goalkeeper", 2: "player"}
    assert refine_goalkeepers(t, {1: "player", 2: "player"}, team_of) == {1: "player", 2: "player"}


def test_ball_trajectory_keeps_segments_before_a_long_gap():
    # ball visible in frames 1–40, out of view 41–120, visible again 121–160
    f = np.r_[np.arange(1, 41), np.arange(121, 161)]
    cands = pd.DataFrame({"frame": f, "x": 100.0 + 2 * f, "y": 300.0, "conf": 0.6})
    out = solve_ball_trajectory(cands, 160)
    assert out.x.iloc[:40].notna().all() and out.x.iloc[120:].notna().all()
    assert out.x.iloc[45:115].isna().all()           # no interpolation across the long gap
