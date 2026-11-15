import os

import numpy as np
import pandas as pd
import pytest

from src.data.soccernet import Sequence, parse_role
from src.evaluation.metrics import (evaluate_ball, evaluate_detection, evaluate_teams,
                                    evaluate_tracking)


@pytest.fixture
def seq(tmp_path):
    d = tmp_path / "SNMOT-999"
    (d / "img1").mkdir(parents=True)
    (d / "gt").mkdir()
    for f in range(1, 11):
        (d / "img1" / f"{f:06d}.jpg").write_bytes(b"")
    (d / "seqinfo.ini").write_text("[Sequence]\nframeRate=25\nseqLength=10\nimWidth=1920\nimHeight=1080\n")
    (d / "gameinfo.ini").write_text(
        "[Sequence]\nnum_tracklets=3\n"
        "trackletID_1= player team left;7\ntrackletID_2= player team right;9\ntrackletID_3= ball;1\n")
    rows = []
    for f in range(1, 11):
        rows.append([f, 1, 100 + f, 100, 40, 80, 1, -1, -1, -1])
        rows.append([f, 2, 500 - f, 300, 40, 80, 1, -1, -1, -1])
        rows.append([f, 3, 300, 300, 10, 10, 1, -1, -1, -1])
    np.savetxt(d / "gt" / "gt.txt", np.array(rows), delimiter=",", fmt="%d")
    return Sequence(str(d))


def _perfect(seq):
    g = pd.DataFrame(seq.gt)
    g = g[g.role != "ball"].rename(columns={"id": "track_id"})
    g["conf"] = 0.9
    g["team"] = g.team.map({"left": "B", "right": "A"})
    return g


def test_parse_role():
    assert parse_role(" goalkeepers team left;1")["role"] == "goalkeeper"
    assert parse_role("player team right;10")["team"] == "right"
    assert parse_role("ball;1")["role"] == "ball"


def test_perfect_scores(seq):
    p = _perfect(seq)
    assert evaluate_detection(seq, p)["f1"] == pytest.approx(1.0)
    t = evaluate_tracking(seq, p)
    assert t["HOTA"] == pytest.approx(1.0) and t["IDSW"] == 0 and t["IDF1"] == pytest.approx(1.0)
    assert evaluate_teams(seq, p)["team_acc"] == pytest.approx(1.0)  # swapped mapping allowed


def test_id_switch_penalised(seq):
    p = _perfect(seq)
    p.loc[(p.frame > 5) & (p.track_id == 1), "track_id"] = 77
    t = evaluate_tracking(seq, p)
    assert t["IDSW"] == 1 and t["HOTA"] < 1.0


def test_ball(seq):
    b = pd.DataFrame({"frame": range(1, 11), "x": 305.0, "y": 305.0})
    r = evaluate_ball(seq, b)
    assert r["ball_recall"] == 1.0 and r["ball_precision"] == 1.0
    b.loc[b.frame > 5, "x"] = 900
    assert evaluate_ball(seq, b)["ball_recall"] == pytest.approx(0.5)
