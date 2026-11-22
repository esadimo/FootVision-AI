"""Accuracy: scores against SoccerNet ground truth, for one run or two runs side by side."""

import glob
import os

import pandas as pd
import streamlit as st

from ui.common import METRICS, ROOT, discover_evaluations, fmt, load_csv, rel

st.title("Accuracy")
st.caption("How close the pipeline gets to the hand-labelled ground truth of SoccerNet clips. "
           "Possession and passes have no human labels, so they are scored against the same analytics run on the "
           "true players and ball.")

files = discover_evaluations()
# single-clip scores written by the Analyse page (<folder>/<clip>/evaluation.csv), grouped per folder
for folder in {os.path.dirname(os.path.dirname(p)) for p in glob.glob(os.path.join(ROOT, "outputs*", "*", "evaluation.csv"))}:
    if os.path.join(folder, "evaluation.csv") not in files:
        files.append(os.path.join(folder, "*", "evaluation.csv"))
if not files:
    st.info("No scores yet. Run a SoccerNet clip on **Analyse a clip** with “Score against ground truth” ticked.")
    st.stop()


def load(f):
    if "*" in f:
        parts = [load_csv(p) for p in sorted(glob.glob(f))]
        parts = [p for p in parts if p is not None]
        return pd.concat(parts, ignore_index=True) if parts else None
    return load_csv(f)


def name(f):
    return rel(os.path.dirname(f).replace(os.sep + "*", "")) + (" (single clips)" if "*" in f else "")


default_a = next((i for i, f in enumerate(files) if "outputs_test_v5" in f), 0)
c1, c2 = st.columns(2)
fa = c1.selectbox("Results", files, index=default_a, format_func=name)
fb = c2.selectbox("Compare with (optional)", [None] + files, format_func=lambda f: "—" if f is None else name(f))
A, B = load(fa), (load(fb) if fb else None)
if A is None or A.empty:
    st.warning("This results file is empty.")
    st.stop()

common = sorted(set(A.seq) & set(B.seq)) if B is not None else None
if B is not None:
    if not common:
        st.warning("The two result sets have no clips in common, so they cannot be compared fairly.")
        B = None
    elif len(common) < max(len(A), len(B)):
        st.caption(f"Comparing on the {len(common)} clips both runs share.")
        A, B = A[A.seq.isin(common)], B[B.seq.isin(common)]
st.caption(f"{len(A)} clip(s)")

# ── headline cards ────────────────────────────────────────────────────────────
cols = [c for c in METRICS if c in A.columns]
for row in range(0, len(cols), 4):
    cc = st.columns(4)
    for k, c in enumerate(cols[row:row + 4]):
        label, kind, hib = METRICS[c]
        a = A[c].mean()
        delta = None
        if B is not None and c in B.columns:
            dv = a - B[c].mean()
            delta = (f"{100 * dv:+.1f} pts" if kind == "pct" else f"{dv:+.1f} pts" if kind == "pts"
                     else f"{dv:+.0f}" if kind == "int" else f"{dv:+.3f}")
        cc[k].metric(label, fmt(a, kind), delta=delta, delta_color="normal" if hib else "inverse")
if B is not None:
    st.caption(f"Arrows compare with {name(fb)}; green means better.")

# ── per-clip table ────────────────────────────────────────────────────────────
st.subheader("Per clip")
show = st.multiselect("Columns", cols, default=[c for c in ("trk_HOTA", "trk_IDF1", "team_acc", "ball_recall@10",
                                                            "calib_height_ok", "possession_abs_err_pts") if c in cols],
                      format_func=lambda c: METRICS[c][0])
t = A[["seq"] + show].copy()
cfg = {"seq": st.column_config.TextColumn("Clip")}
for c in show:
    label, kind, _ = METRICS[c]
    if kind == "pct":
        t[c] = t[c] * 100
        cfg[c] = st.column_config.ProgressColumn(label, format="%.1f%%", min_value=0, max_value=100)
    else:
        cfg[c] = st.column_config.NumberColumn(label, format="%.0f" if kind == "int" else "%.1f" if kind == "pts"
                                               else "%.3f")
st.dataframe(t, column_config=cfg, use_container_width=True, hide_index=True, height=min(600, 38 + 35 * len(t)))
st.caption("Click a column header to sort, e.g. to find the weakest clips. Open a clip on the **Match** or "
           "**Calibration** page to see why.")

with st.expander("What the numbers mean"):
    st.markdown("""
- **Tracking quality (HOTA)** and **Identity kept (IDF1)**: 0–1, how well players are found and keep one ID.
- **Identity switches**: how often a player's ID jumps to someone else in a 30 s clip.
- **Ball found**: share of frames where the ball is placed within 10 px of the true ball.
- **Calibration: players at right size**: share of players whose projected height matches their real height
  within ±25% — a wrong camera estimate distorts sizes. **Keepers near their goal** catches a shifted pitch.
- **Possession % error**: difference between predicted and true possession share, in percentage points.
- **Pass / turnover detection (F1)**: 0–1, matching events within ±10 frames.
- **Distance covered error**: median relative error of distance run per player.
""")
