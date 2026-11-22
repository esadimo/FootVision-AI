"""Calibration: scrub through a clip and see the projected pitch on top of the video frame."""

import os

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from ui.common import discover_clips, discover_runs, load_json, rel, run_clips

st.title("Calibration")
st.caption("The pipeline estimates the broadcast camera for every frame, which turns pixels into pitch metres. "
           "Here the pitch template is drawn back onto the frame: red should sit on the real white lines.")

runs = [r for r in discover_runs() if any(os.path.exists(os.path.join(r, c, "calibration.json")) for c in run_clips(r))]
if not runs:
    st.info("No calibrations yet. Run a clip on **Analyse a clip**.")
    st.stop()
c1, c2 = st.columns(2)
run = c1.selectbox("Results folder", runs, format_func=rel)
clip = c2.selectbox("Clip", [c for c in run_clips(run) if os.path.exists(os.path.join(run, c, "calibration.json"))])
cal = load_json(os.path.join(run, clip, "calibration.json")) or {}
src = discover_clips().get(clip)
if not cal.get("H") or not src:
    st.warning("Calibration or the clip's frames are missing.")
    st.stop()

H = cal["H"]
L, W = cal.get("pitch", [105.0, 68.0])
frames = sorted(os.listdir(os.path.join(src, "img1")))
n = min(len(H), len(frames))

sc = cal.get("self_check")
m = st.columns(5)
C = cal.get("camera_centre") or [np.nan] * 3
m[0].metric("Camera height", f"{-C[2]:.0f} m", help="Estimated height of the main camera above the grass.")
m[1].metric("Behind touchline", f"{C[1] - W:.0f} m", help="How far the camera stands back from the near touchline.")
if sc:
    s = np.array([np.nan if v is None else v for v in sc], float)
    m[2].metric("Confidence", f"{100 * np.nanmean(s):.0f}%",
                help="Mean share of detected players projected at a plausible size, per frame.")
    m[3].metric("Frames flagged", f"{100 * np.mean(np.nan_to_num(s, nan=1) < 0.5):.0f}%",
                help="Frames where fewer than half the players look the right size after repair.")
m[4].metric("Frames repaired", cal.get("repaired_frames", "—"),
            help="Frames whose camera was replaced by the self-check-and-repair stage.")

fr = st.slider("Frame", 1, n, 1)
show_lines = st.toggle("Highlight detected pitch lines (yellow)", value=True)


@st.cache_data(show_spinner=False, max_entries=64)
def overlay(img_path, Hl, L, W, lines):
    from src.calibration.camera_model import get_template
    from src.calibration.line_tracker import _project, line_mask
    f = cv2.imread(img_path)
    v = f.copy()
    if lines:
        v[line_mask(f) > 0] = (0, 255, 255)
    for x, y in _project(np.linalg.inv(np.array(Hl)), get_template(L, W)).astype(int):
        if 0 <= x < f.shape[1] and 0 <= y < f.shape[0]:
            cv2.circle(v, (x, y), 3, (0, 0, 255), -1)
    return cv2.cvtColor(v, cv2.COLOR_BGR2RGB)


st.image(overlay(os.path.join(src, "img1", frames[fr - 1]), H[fr - 1], L, W, show_lines),
         use_container_width=True, caption=f"Frame {fr}: red = where the calibration puts the pitch lines")

if sc:
    s = pd.Series([np.nan if v is None else 100 * v for v in sc[:n]], index=range(1, n + 1), name="Confidence %")
    st.markdown("**Confidence per frame**: dips mark frames where players come out at the wrong size")
    st.line_chart(s, height=200)
else:
    st.caption("This calibration was made before the self-check existed; re-run the clip to get per-frame "
               "confidence.")
if cal.get("focal"):
    st.markdown("**Zoom (focal length in pixels)**: rises when the camera zooms in")
    st.line_chart(pd.Series(cal["focal"][:n], index=range(1, n + 1), name="Focal length"), height=160)
st.caption(f"Clip frames: {rel(src)}")
