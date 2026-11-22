"""Analyse a clip: pick a SoccerNet clip or upload a video, run the pipeline in the background, follow progress."""

import os
import re
import tempfile

import streamlit as st

from ui.common import (PY, ROOT, clip_label, discover_clips, import_video, job_progress, job_status, list_jobs,
                       read_log, rel, start_job, stop_job)

st.title("Analyse a clip")
st.caption("Runs the full pipeline: detection, tracking, camera calibration, teams, ball, possession, "
           "passes and player statistics. It keeps running in the background if you switch pages or close the tab.")

clips = discover_clips()

src = st.radio("Input", ["A clip already on this computer", "Upload a video file"], horizontal=True)
clip_dir = None
if src == "A clip already on this computer":
    if not clips:
        st.warning("No clips found in data/SoccerNet/tracking or data/raw.")
    else:
        names = list(clips)
        default = names.index("SNMOT-062") if "SNMOT-062" in names else 0
        name = st.selectbox("Clip", names, index=default, format_func=lambda n: clip_label(n, clips[n]))
        clip_dir = clips[name]
        n_frames = len(os.listdir(os.path.join(clip_dir, "img1")))
        st.caption(f"{n_frames} frames · {rel(clip_dir)}")
else:
    up = st.file_uploader("Video file", type=["mp4", "mov", "avi", "mkv"])
    c1, c2 = st.columns(2)
    vname = c1.text_input("Name for this clip", value="my_clip")
    max_s = c2.number_input("Use only the first … seconds (0 = whole video)", min_value=0, value=30, step=10)
    st.caption("A 30-second clip takes roughly 10–15 minutes on this Mac. Broadcast-style footage from a "
               "single main camera works best.")
    if up is not None and st.button("Import video"):
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", vname) or "my_clip"
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(up.name)[1], delete=False) as tf:
            tf.write(up.getbuffer())
        with st.spinner("Extracting frames…"):
            d = import_video(tf.name, safe, max_s or None)
        os.unlink(tf.name)
        st.session_state["imported"] = d
        st.success(f"Imported {len(os.listdir(os.path.join(d, 'img1')))} frames as '{safe}'.")
    clip_dir = st.session_state.get("imported")
    if clip_dir:
        st.caption(f"Ready: {rel(clip_dir)}")

st.subheader("Options")
c1, c2 = st.columns(2)
out_dir = c1.text_input("Save results in folder", value="outputs_ui",
                        help="A folder in the project. The Match, Accuracy and Calibration pages list it.")
video = c2.checkbox("Also render the annotated video", value=True, help="Adds a few minutes.")
has_gt = bool(clip_dir) and os.path.exists(os.path.join(clip_dir, "gt", "gt.txt"))
score = c2.checkbox("Score against ground truth", value=has_gt, disabled=not has_gt,
                    help="Only for SoccerNet clips, which come with hand-labelled players and ball.")

running = [j for j in list_jobs() if j["state"] == "running"]
if running:
    st.info(f"Another analysis is running ({running[0]['title']}). Starting a second one at the same time "
            "works but makes both slower.")

if st.button("Run analysis", type="primary", disabled=not clip_dir):
    name = os.path.basename(clip_dir)
    cmds = [[PY, "scripts/run_pipeline.py", "--seq_dir", rel(clip_dir), "--output_dir", out_dir]]
    if video:
        cmds[0].append("--video")
    if score and has_gt:
        cmds.append([PY, "scripts/evaluate_pipeline.py", "--runs", out_dir, "--seqs",
                     rel(clip_dir), "--out", os.path.join(out_dir, name, "evaluation.csv")])
    jid = start_job(f"{name} → {out_dir}", cmds, {"clip": name, "out_dir": out_dir})
    st.session_state["job"] = jid
    st.rerun()

st.divider()
st.subheader("Analyses")
jobs = list_jobs()
if not jobs:
    st.caption("Nothing run from here yet.")


def show_job(j):
    s = job_status(j["id"])
    log = read_log(j["id"])
    frac, label = job_progress(log)
    icon = {"running": "⏳", "finished": "✅", "failed": "❌", "stopped": "⏹️"}[s["state"]]
    st.markdown(f"**{icon} {j['title']}** · started {j['id'][9:11]}:{j['id'][11:13]} on "
                f"{j['id'][6:8]}.{j['id'][4:6]}")
    if s["state"] == "running":
        st.progress(frac, text=label)
        if st.button("Stop", key=f"stop_{j['id']}"):
            stop_job(j["id"]); st.rerun()
    elif s["state"] == "finished":
        st.success("Done. Open the **Match** page to see the results"
                   + (", or **Accuracy** for the scores." if any("evaluate" in " ".join(c) for c in j["commands"]) else "."))
    elif s["state"] == "failed":
        st.error("The analysis stopped with an error; the log below shows where.")
    with st.expander("Log", expanded=s["state"] in ("running", "failed")):
        st.code(log or "(no output yet)", language=None)


live_job = st.fragment(run_every=3)(show_job)   # refreshes itself every 3 s while running

for j in jobs[:5]:
    with st.container(border=True):
        (live_job if j["state"] == "running" else show_job)(j)
