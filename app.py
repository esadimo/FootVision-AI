"""
FootVision AI — web app.

Start it by double-clicking FootVision.command, or with:
    .venv/bin/streamlit run app.py

Pages:
    Analyse a clip  — run the pipeline on a SoccerNet clip or your own video, with live progress
    Match           — video, pitch view, possession, events and player stats of an analysed clip
    Accuracy        — scores against ground truth; compare two runs
    Calibration     — the projected pitch drawn on each frame, per-frame confidence
"""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="FootVision AI", page_icon="⚽", layout="wide")

pages = [
    st.Page("ui/page_match.py", title="Match", icon="⚽", default=True),
    st.Page("ui/page_analyse.py", title="Analyse a clip", icon="▶️"),
    st.Page("ui/page_accuracy.py", title="Accuracy", icon="🎯"),
    st.Page("ui/page_calibration.py", title="Calibration", icon="📐"),
]
st.navigation(pages).run()
