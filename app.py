"""
FootVision AI -- Phase 14: Interactive Dashboard
Usage: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="FootVision AI", page_icon="⚽", layout="wide")

st.title("⚽ FootVision AI Dashboard")
st.markdown("Automated football match analysis from recorded video.")

# Sidebar Configuration
st.sidebar.header("Data Source")
seq_name = st.sidebar.text_input("Sequence Name", value="SNMOT-062")
output_dir = st.sidebar.text_input("Outputs Directory", value="outputs")

# Helper function to load data safely
@st.cache_data
def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

# Load Data
possession_df = load_csv(os.path.join(output_dir, f"{seq_name}_phase11_possession.csv"))
player_stats_df = load_csv(os.path.join(output_dir, f"{seq_name}_phase13_player_stats.csv"))
team_metrics_df = load_csv(os.path.join(output_dir, f"{seq_name}_phase13_team_metrics.csv"))

# Define Tabs
tab1, tab2, tab3 = st.tabs(["📺 Match Overview", "🏃 Player Stats", "📋 Tactical Analysis"])

# --- TAB 1: Match Overview ---
with tab1:
    st.header("Match Overview")
    
    if possession_df is not None and not possession_df.empty:
        last_row = possession_df.iloc[-1]
        col1, col2 = st.columns(2)
        # Using delta colors inversely or just off to show neutral stats
        col1.metric("Team A Possession", f"{last_row['team_A_pct']:.1f}%")
        col2.metric("Team B Possession", f"{last_row['team_B_pct']:.1f}%")
        
    st.markdown("### Match Replay (Passes & Turnovers)")
    video_path = os.path.join(output_dir, f"{seq_name}_phase12_passes.mp4")
    if os.path.exists(video_path):
        st.video(video_path)
    else:
        st.warning(f"Annotated video not found at: {video_path}")


# --- TAB 2: Player Stats ---
with tab2:
    st.header("Player Physical & Technical Stats")
    if player_stats_df is not None:
        
        # We can format the track_id as string so it doesn't get comma separators
        player_stats_df['track_id'] = player_stats_df['track_id'].astype(str)
        
        st.dataframe(
            player_stats_df.style.highlight_max(
                subset=['distance_covered_m', 'max_speed_kmh', 'passes_made'], 
                color='#2e7d32' # Dark green highlight
            ),
            use_container_width=True
        )
        
        st.markdown("### Top Distance Coverers")
        top_dist = player_stats_df.sort_values('distance_covered_m', ascending=False).head(15)
        # Create a combined label for the chart
        top_dist['Label'] = top_dist['team_label'] + " #" + top_dist['track_id']
        chart_data = top_dist.set_index('Label')[['distance_covered_m']]
        st.bar_chart(chart_data)
        
    else:
        st.info("Player stats CSV not found.")


# --- TAB 3: Tactical Analysis ---
with tab3:
    st.header("Tactical Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Team Heatmaps")
        hm_path = os.path.join(output_dir, f"{seq_name}_phase13_heatmaps.jpg")
        if os.path.exists(hm_path):
            st.image(hm_path, use_container_width=True, caption="Spatial dominance based on KDE.")
        else:
            st.info("Heatmaps image not found.")
            
    with col2:
        st.subheader("Pass Network")
        pn_path = os.path.join(output_dir, f"{seq_name}_phase12_pass_network.jpg")
        if os.path.exists(pn_path):
            st.image(pn_path, use_container_width=True, caption="Nodes scaled by average position. Edges by pass volume.")
        else:
            st.info("Pass network image not found.")

    st.markdown("---")
    st.subheader("Team Shape Dynamics")
    st.markdown("Tracks how compact or expansive the teams are over time.")
    
    if team_metrics_df is not None:
        team_metrics_df.set_index('frame_number', inplace=True)
        
        st.markdown("**Team Width (Lateral stretch in metres)**")
        st.line_chart(team_metrics_df[['Team A_width_m', 'Team B_width_m']])
        
        st.markdown("**Team Depth (Longitudinal stretch in metres)**")
        st.line_chart(team_metrics_df[['Team A_depth_m', 'Team B_depth_m']])
    else:
        st.info("Team metrics CSV not found.")
