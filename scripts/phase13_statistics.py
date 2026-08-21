"""
FootVision AI -- Phase 13: Player & Team Statistics

WORKFLOW:
    Reads pitch coordinates and pass events.
    Calculates player physical metrics (Distance, Speed).
    Calculates team spatial metrics (Width, Depth).
    Generates Team Heatmaps.

Usage:
    python scripts/phase13_statistics.py [--seq_dir data/raw/SNMOT-062]
"""

import os
import sys
import argparse
import csv

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.analytics.statistics import calculate_player_stats, calculate_team_metrics, generate_team_heatmaps

def main():
    parser = argparse.ArgumentParser(description="Phase 13: Player & Team Statistics")
    parser.add_argument("--seq_dir", default="data/raw/SNMOT-062")
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()

    seq_name = os.path.basename(args.seq_dir)
    coords_csv = os.path.join(args.output_dir, f"{seq_name}_phase9_pitch_coords.csv")
    events_csv = os.path.join(args.output_dir, f"{seq_name}_phase12_events.csv")
    
    if not os.path.exists(coords_csv):
        print(f"[ERROR] Required file {coords_csv} not found.")
        sys.exit(1)

    print(f"\n  Phase 13 -- Statistics & Heatmaps")
    print(f"  Sequence : {seq_name}")
    
    # 1. Player Stats
    print("\n  Calculating Player Distance & Speed Metrics...")
    player_stats = calculate_player_stats(coords_csv, events_csv, fps=25.0)
    
    player_csv_out = os.path.join(args.output_dir, f"{seq_name}_phase13_player_stats.csv")
    with open(player_csv_out, "w", newline="", encoding="utf-8") as f:
        if player_stats:
            writer = csv.DictWriter(f, fieldnames=player_stats[0].keys())
            writer.writeheader()
            writer.writerows(player_stats)
    print(f"  [Output] Player Stats saved to: {player_csv_out}")

    # Print Top 5 Distance Coverers
    print("\n  Top 5 Players by Distance Covered:")
    for i, p in enumerate(player_stats[:5]):
        print(f"    {i+1}. #{p['track_id']} ({p['team_label']}) - {p['distance_covered_m']}m | Max: {p['max_speed_kmh']}km/h | Passes: {p['passes_made']}")

    # 2. Team Spatial Metrics
    print("\n  Calculating Team Spatial Metrics (Width/Depth)...")
    team_metrics = calculate_team_metrics(coords_csv)
    
    team_csv_out = os.path.join(args.output_dir, f"{seq_name}_phase13_team_metrics.csv")
    with open(team_csv_out, "w", newline="", encoding="utf-8") as f:
        if team_metrics:
            writer = csv.DictWriter(f, fieldnames=team_metrics[0].keys())
            writer.writeheader()
            writer.writerows(team_metrics)
    print(f"  [Output] Team Spatial Metrics saved to: {team_csv_out}")

    # 3. Heatmaps
    print("\n  Generating Spatial Heatmaps...")
    heatmap_out = os.path.join(args.output_dir, f"{seq_name}_phase13_heatmaps.jpg")
    generate_team_heatmaps(coords_csv, heatmap_out)
    print(f"  [Output] Team Heatmaps saved to: {heatmap_out}")
    
    print("\n  Phase 13 Complete!\n")

if __name__ == "__main__":
    main()
