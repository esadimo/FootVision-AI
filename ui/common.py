"""
Shared helpers for the FootVision web UI: finding clips and pipeline runs, background jobs,
and plain-language names for the evaluation metrics.

Background jobs run as detached processes (so a 10-minute analysis survives page reloads and
browser closes).  Each job lives in runs/ui_jobs/<id>/ with meta.json, log.txt and, once the
job ends, exit_code.
"""

from __future__ import annotations

import glob
import json
import os
import shlex
import subprocess
import sys
import time
from typing import Dict, List, Optional

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PY = sys.executable
JOBS_DIR = os.path.join(ROOT, "runs", "ui_jobs")
CLIP_ROOTS = ["data/SoccerNet/tracking/train", "data/SoccerNet/tracking/test", "data/raw"]
VAL_GAME = {f"SNMOT-{i:03d}" for i in range(60, 78)}


def rel(p: str) -> str:
    return os.path.relpath(p, ROOT)


# ── clips (inputs) ────────────────────────────────────────────────────────────

def discover_clips() -> Dict[str, str]:
    """Clip name → directory, for every folder with an img1/ frame directory."""
    out = {}
    for r in CLIP_ROOTS:
        for d in sorted(glob.glob(os.path.join(ROOT, r, "*"))):
            if os.path.isdir(os.path.join(d, "img1")):
                out[os.path.basename(d)] = d
    return out


def clip_label(name: str, path: str) -> str:
    if "/test/" in path:
        split = "test split"
    elif name in VAL_GAME:
        split = "validation game"
    elif "/train/" in path:
        split = "training games"
    else:
        split = "your video"
    gt = " · has ground truth" if os.path.exists(os.path.join(path, "gt", "gt.txt")) else ""
    return f"{name} ({split}{gt})"


def import_video(video_path: str, name: str, max_seconds: Optional[float] = None) -> str:
    """Extract a video's frames into data/raw/<name>/img1 + seqinfo.ini so the pipeline can read it."""
    import cv2
    out = os.path.join(ROOT, "data", "raw", name)
    img = os.path.join(out, "img1")
    os.makedirs(img, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n, w, h = 0, 0, 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_seconds and n >= max_seconds * fps):
            break
        n += 1
        h, w = frame.shape[:2]
        cv2.imwrite(os.path.join(img, f"{n:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cap.release()
    with open(os.path.join(out, "seqinfo.ini"), "w") as f:
        f.write(f"[Sequence]\nname={name}\nimDir=img1\nframeRate={round(fps)}\nseqLength={n}\n"
                f"imWidth={w}\nimHeight={h}\nimExt=.jpg\n")
    return out


# ── runs (outputs) ────────────────────────────────────────────────────────────

def discover_runs() -> List[str]:
    """Output folders that contain at least one analysed clip (<run>/<clip>/summary.json)."""
    cands = [d for d in glob.glob(os.path.join(ROOT, "outputs*")) if os.path.isdir(d)]
    cands += [d for d in glob.glob(os.path.join(ROOT, "runs", "tune", "*")) if os.path.isdir(d)]
    runs = [d for d in cands if glob.glob(os.path.join(d, "SNMOT-*", "summary.json")) or
            glob.glob(os.path.join(d, "*", "summary.json"))]
    return sorted(runs, key=lambda d: -os.path.getmtime(d))


def run_clips(run: str) -> List[str]:
    return sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(run, "*", "summary.json")))


def discover_evaluations() -> List[str]:
    files = glob.glob(os.path.join(ROOT, "outputs*", "evaluation.csv"))
    files += glob.glob(os.path.join(ROOT, "runs", "tune", "*", "evaluation.csv"))
    return sorted(files, key=lambda f: -os.path.getmtime(f))


def load_csv(path: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path) if os.path.exists(path) else None
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return None


def load_json(path: str) -> Optional[dict]:
    try:
        return json.load(open(path)) if os.path.exists(path) else None
    except (OSError, json.JSONDecodeError):
        return None


# ── background jobs ───────────────────────────────────────────────────────────

def start_job(title: str, commands: List[List[str]], meta: Optional[dict] = None) -> str:
    """Run `commands` one after another in a detached shell; stops at the first failure."""
    os.makedirs(JOBS_DIR, exist_ok=True)
    jid = time.strftime("%Y%m%d-%H%M%S")
    d = os.path.join(JOBS_DIR, jid)
    os.makedirs(d, exist_ok=True)
    log, code = os.path.join(d, "log.txt"), os.path.join(d, "exit_code")
    chain = " && ".join(" ".join(shlex.quote(c) for c in cmd) for cmd in commands)
    script = f"cd {shlex.quote(ROOT)} && ({chain}) > {shlex.quote(log)} 2>&1; echo $? > {shlex.quote(code)}"
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    p = subprocess.Popen(["/bin/sh", "-c", script], env=env, start_new_session=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    json.dump({"id": jid, "title": title, "pid": p.pid, "started": time.time(), "commands": commands,
               **(meta or {})}, open(os.path.join(d, "meta.json"), "w"), indent=2)
    return jid


def list_jobs() -> List[dict]:
    jobs = []
    for d in sorted(glob.glob(os.path.join(JOBS_DIR, "*")), reverse=True):
        m = load_json(os.path.join(d, "meta.json"))
        if m:
            jobs.append({**m, **job_status(m["id"])})
    return jobs


def job_status(jid: str) -> dict:
    d = os.path.join(JOBS_DIR, jid)
    code_f = os.path.join(d, "exit_code")
    if os.path.exists(code_f):
        try:
            code = int(open(code_f).read().strip() or "1")
        except ValueError:
            code = 1
        return {"state": "finished" if code == 0 else "failed", "exit_code": code}
    m = load_json(os.path.join(d, "meta.json")) or {}
    try:
        os.kill(int(m.get("pid", -1)), 0)
        return {"state": "running", "exit_code": None}
    except (OSError, ValueError):
        return {"state": "stopped", "exit_code": None}


def read_log(jid: str, tail: int = 400) -> str:
    p = os.path.join(JOBS_DIR, jid, "log.txt")
    if not os.path.exists(p):
        return ""
    lines = open(p, errors="replace").read().splitlines()
    return "\n".join(lines[-tail:])


def stop_job(jid: str):
    m = load_json(os.path.join(JOBS_DIR, jid, "meta.json")) or {}
    try:
        os.killpg(int(m["pid"]), 15)
    except (OSError, KeyError, ValueError):
        pass


STAGES = [("] detect", "Detecting players and ball"), ("] track", "Tracking players"),
          ("] calibrate", "Calibrating the camera (slowest step)"), ("] roles/teams", "Teams, roles and goalkeepers"),
          ("] ball", "Ball trajectory"), ("] analytics", "Possession, passes and statistics"),
          ("] done in", "Analysis finished"), ("phase 14", "Rendering the video"), ("evaluated", "Scoring against ground truth")]


def job_progress(log: str) -> tuple:
    """(fraction, label) from the pipeline log lines."""
    low = log.lower()
    last, label = -1, "Starting…"
    for k, (key, text) in enumerate(STAGES):
        if key.lower() in low:
            last, label = k, text
    return max(0.02, (last + 1) / len(STAGES)), label


# ── metrics in plain language ─────────────────────────────────────────────────

# column → (label, kind, higher_is_better); kind: pct | num | int
METRICS = {
    "trk_HOTA": ("Tracking quality (HOTA)", "num", True),
    "trk_IDF1": ("Identity kept (IDF1)", "num", True),
    "trk_IDSW": ("Identity switches per clip", "int", False),
    "team_acc": ("Team assignment", "pct", True),
    "role_goalkeeper_recall": ("Goalkeepers found", "pct", True),
    "role_goalkeeper_precision": ("Goalkeeper labels correct", "pct", True),
    "role_referee_recall": ("Referees found", "pct", True),
    "ball_recall@10": ("Ball found (within 10 px)", "pct", True),
    "ball_precision@10": ("Ball positions correct", "pct", True),
    "calib_height_ok": ("Calibration: players at right size", "pct", True),
    "calib_keeper_ok": ("Calibration: keepers near their goal", "pct", True),
    "possession_frame_acc": ("Possession: right team per frame", "pct", True),
    "possession_abs_err_pts": ("Possession % error", "pts", False),
    "pass_f1": ("Pass detection (F1)", "num", True),
    "turnover_f1": ("Turnover detection (F1)", "num", True),
    "distance_rel_err_median": ("Distance covered error", "pct", False),
}


def fmt(v, kind: str) -> str:
    if v is None or pd.isna(v):
        return "—"
    if kind == "pct":
        return f"{100 * v:.1f}%"
    if kind == "pts":
        return f"{v:.1f} pts"
    return f"{v:.0f}" if kind == "int" else f"{v:.3f}"


def browser_video(path: str) -> Optional[str]:
    """Path of a browser-playable (H.264) version of `path`; older MPEG-4 Part 2 videos need web_copy()."""
    import cv2
    if not os.path.exists(path):
        return None
    v = cv2.VideoCapture(path)
    cc = int(v.get(cv2.CAP_PROP_FOURCC)); v.release()
    tag = "".join(chr((cc >> 8 * i) & 255) for i in range(4)).lower()
    if tag in ("avc1", "h264"):
        return path
    web = path[:-4] + "_web.mp4"
    return web if os.path.exists(web) and os.path.getmtime(web) >= os.path.getmtime(path) else None


def web_copy(path: str, progress=None) -> str:
    """Re-encode a video to H.264 next to the original (<name>_web.mp4)."""
    import cv2
    web = path[:-4] + "_web.mp4"
    v = cv2.VideoCapture(path)
    n = int(v.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps = v.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(v.get(cv2.CAP_PROP_FRAME_WIDTH)), int(v.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(web, cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))
    k = 0
    while True:
        ok, f = v.read()
        if not ok:
            break
        out.write(f); k += 1
        if progress and k % 25 == 0:
            progress(min(1.0, k / n))
    v.release(); out.release()
    return web
