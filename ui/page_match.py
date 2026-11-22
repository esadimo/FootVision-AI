"""Match: everything the pipeline produced for one clip — video, possession, pitch view, events, player stats."""

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.common import browser_video, discover_runs, load_csv, load_json, rel, run_clips, web_copy

TEAM_COL = {"A": "#2a78d6", "B": "#eb6834", "": "#898781"}

st.title("Match")
runs = discover_runs()
if not runs:
    st.info("No analysed clips yet. Run one on the **Analyse a clip** page.")
    st.stop()

import glob
with_video = [i for i, r in enumerate(runs) if glob.glob(os.path.join(r, "*_phase14_ultimate.mp4"))]
c1, c2 = st.columns(2)
run = c1.selectbox("Results folder", runs, index=with_video[0] if with_video else 0, format_func=rel,
                   help="Folders that hold analysed clips. Folders with a rendered video are listed first by default.")
clips = run_clips(run)
clip = c2.selectbox("Clip", clips)
d = os.path.join(run, clip)
summary = load_json(os.path.join(d, "summary.json")) or {}
L, W = (summary.get("pitch_dims") or [105.0, 68.0])

# ── headline ──────────────────────────────────────────────────────────────────
pos = summary.get("possession_pct") or {}
m = st.columns(5)
m[0].metric("Possession A", f"{pos.get('A', float('nan')):.0f}%")
m[1].metric("Possession B", f"{pos.get('B', float('nan')):.0f}%")
m[2].metric("Passes", summary.get("n_passes", "—"))
m[3].metric("Turnovers", summary.get("n_turnovers", "—"))
conf = summary.get("calibration_self_check")
m[4].metric("Calibration", "—" if conf is None else f"{100 * conf:.0f}%",
            help="Share of detected players the calibration projects at a plausible size. "
                 "Below ~60% the pitch positions, distances and speeds are less reliable.")
if summary.get("calibration_low_confidence"):
    st.warning("Calibration confidence is low for this clip: pitch positions, distances and speeds may be off.")

tab_v, tab_p, tab_pos, tab_ev, tab_pl = st.tabs(["Video", "Pitch view", "Possession", "Events", "Players"])

# ── video ─────────────────────────────────────────────────────────────────────
with tab_v:
    vid = os.path.join(run, f"{clip}_phase14_ultimate.mp4")
    if not os.path.exists(vid):
        st.info("No video was rendered for this clip. Re-run it on **Analyse a clip** with "
                "“Also render the annotated video” ticked.")
    else:
        playable = browser_video(vid)
        if playable:
            st.video(playable)
        else:
            st.caption("This video was saved in an older format that browsers cannot play.")
            if st.button("Convert for the browser (about a minute)"):
                bar = st.progress(0.0, text="Converting…")
                web_copy(vid, lambda f: bar.progress(f, text="Converting…"))
                st.rerun()
        st.caption(f"File: {rel(vid)}")

# ── pitch view ────────────────────────────────────────────────────────────────
tracks = load_csv(os.path.join(d, "tracks.csv"))
ball = load_csv(os.path.join(d, "ball.csv"))


def pitch_fig(L, W):
    f = go.Figure()
    line = dict(color="rgba(255,255,255,0.85)", width=2)
    shapes = [dict(type="rect", x0=0, y0=0, x1=L, y1=W, line=line),
              dict(type="line", x0=L / 2, y0=0, x1=L / 2, y1=W, line=line),
              dict(type="circle", x0=L / 2 - 9.15, y0=W / 2 - 9.15, x1=L / 2 + 9.15, y1=W / 2 + 9.15, line=line)]
    for x0, x1 in ((0, 16.5), (L - 16.5, L)):
        shapes.append(dict(type="rect", x0=x0, y0=W / 2 - 20.16, x1=x1, y1=W / 2 + 20.16, line=line))
    for x0, x1 in ((0, 5.5), (L - 5.5, L)):
        shapes.append(dict(type="rect", x0=x0, y0=W / 2 - 9.16, x1=x1, y1=W / 2 + 9.16, line=line))
    f.update_layout(shapes=shapes, plot_bgcolor="#3a7d3a", height=470, margin=dict(l=10, r=10, t=10, b=10),
                    xaxis=dict(range=[-4, L + 4], showgrid=False, zeroline=False, visible=False),
                    yaxis=dict(range=[W + 4, -4], showgrid=False, zeroline=False, visible=False,
                               scaleanchor="x"),
                    legend=dict(orientation="h", y=-0.02))
    return f


with tab_p:
    if tracks is None:
        st.info("No tracks for this clip.")
    else:
        tracks["team"] = tracks.team.fillna("")
        fr = st.slider("Frame", int(tracks.frame.min()), int(tracks.frame.max()), int(tracks.frame.min()),
                       help="Positions on the pitch in metres, from the camera calibration.")
        g = tracks[tracks.frame == fr].dropna(subset=["sx"])
        fig = pitch_fig(L, W)
        for (role, team), q in g.groupby(["role", "team"]):
            name = {"referee": "Referee"}.get(role, (("GK " if role == "goalkeeper" else "") + f"Team {team}")
                                              if team else role.title())
            col = "#eda100" if role == "referee" else TEAM_COL.get(team, "#898781")
            fig.add_trace(go.Scatter(x=q.sx, y=q.sy, mode="markers+text", name=name,
                                     text=q.track_id.astype(str), textposition="top center",
                                     textfont=dict(color="white", size=10),
                                     marker=dict(size=14 if role == "goalkeeper" else 11, color=col,
                                                 symbol="diamond" if role == "goalkeeper" else "circle",
                                                 line=dict(color="white", width=1))))
        if ball is not None:
            b = ball[ball.frame == fr].dropna(subset=["bx"])
            if len(b):
                fig.add_trace(go.Scatter(x=b.bx, y=b.by, mode="markers", name="Ball",
                                         marker=dict(size=9, color="white", line=dict(color="black", width=1))))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"{len(g)} people on screen at frame {fr} · numbers are track IDs (the same as in the Players tab)")

# ── possession ────────────────────────────────────────────────────────────────
with tab_pos:
    poss = load_csv(os.path.join(d, "possession.csv"))
    if poss is None:
        st.info("No possession file.")
    else:
        poss["possession"] = poss.possession.fillna("").astype(str)
        fps = 25.0
        sec = ((poss.frame - 1) // fps).astype(int)
        share = pd.DataFrame({"A": (poss.possession == "A").groupby(sec).mean(),
                              "B": (poss.possession == "B").groupby(sec).mean(),
                              "Loose ball": (poss.possession == "").groupby(sec).mean()}) * 100
        fig = go.Figure()
        for k, col in (("A", TEAM_COL["A"]), ("B", TEAM_COL["B"]), ("Loose ball", "#b4b2a9")):
            fig.add_trace(go.Bar(x=share.index, y=share[k], name=k if k == "Loose ball" else f"Team {k}",
                                 marker_color=col))
        fig.update_layout(barmode="stack", height=320, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="Second", yaxis_title="% of the second", legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Possession goes to a team once one of its players controls the ball for a few frames, and "
                   "lapses to “loose ball” 2 s after the last touch.")

# ── events ────────────────────────────────────────────────────────────────────
with tab_ev:
    ev = load_csv(os.path.join(d, "events.csv"))
    if ev is None or ev.empty:
        st.info("No passes or turnovers detected in this clip.")
    else:
        ev = ev.assign(time_s=((ev.start_frame - 1) / 25).round(1))
        st.dataframe(ev[["time_s", "event_type", "from_team", "from_player", "to_team", "to_player", "ball_travel_m"]]
                     .rename(columns={"time_s": "Time (s)", "event_type": "Event", "from_team": "From team",
                                      "from_player": "From player", "to_team": "To team", "to_player": "To player",
                                      "ball_travel_m": "Ball travel (m)"}),
                     use_container_width=True, hide_index=True)

# ── players ───────────────────────────────────────────────────────────────────
with tab_pl:
    ps = load_csv(os.path.join(d, "player_stats.csv"))
    if ps is None or ps.empty:
        st.info("No player statistics.")
    else:
        ps["team_label"] = ps.team_label.fillna("").map(lambda t: f"Team {t}" if t else "—")
        st.dataframe(ps.rename(columns={"track_id": "Track ID", "role": "Role", "team_label": "Team",
                                        "time_visible_s": "Visible (s)", "distance_covered_m": "Distance (m)",
                                        "avg_speed_kmh": "Avg speed (km/h)", "max_speed_kmh": "Top speed (km/h)",
                                        "passes_made": "Passes"}),
                     use_container_width=True, hide_index=True)
        st.caption("Distance and speed only count the time a player is visible on camera.")
    tm = load_csv(os.path.join(d, "team_metrics.csv"))
    if tm is not None and not tm.empty:
        st.markdown("**Team shape over time**")
        cols = [c for c in tm.columns if c.endswith(("_width_m", "_depth_m"))]
        st.line_chart(tm.set_index("frame_number")[cols])
