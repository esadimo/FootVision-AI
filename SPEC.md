# FootVision AI — Phase 0 Project Specification

## Project Objective

Build a computer-vision system that processes a **recorded, continuous, wide-angle football clip** (20–30 seconds) and automatically produces structured output showing player positions, team separation, and basic movement data visualized on a top-down pitch.

---

## Supported Input

| Property | Requirement |
|---|---|
| Source | Recorded video file (`.mp4`, `.avi`, `.mov`) |
| View type | Wide-angle / tactical camera shot — full or near-full pitch visible |
| Duration | 20–30 seconds of continuous footage |
| Resolution | Minimum 720p recommended (1280 × 720) |
| Frame rate | 25–30 fps |
| Teams | Two teams with visually distinct jersey colours |
| Players | At least 6 outfield players clearly visible |

---

## Expected Output

| Output | Format | Description |
|---|---|---|
| Annotated video | `.mp4` | Original clip with bounding boxes, track IDs, team colours, and frame counter |
| Detection CSV | `.csv` | One row per detection per frame with bbox, class, confidence, and coordinates |
| Track CSV | `.csv` | One row per track per frame with track ID and pitch coordinates |
| Top-down pitch image | `.png` | Static image showing projected player positions colour-coded by team |
| Possession summary | `.json` | Team A %, Team B %, Unknown %, frame-level timeline |
| Processing report | `.txt` | FPS, total frames processed, missing detections, calibration method used |

---

## Explicitly Excluded Features

The following will **not** be attempted in version 1:

- Full 90-minute match processing
- Live or real-time processing
- Player name or jersey number recognition
- Referee or goalkeeper separation (both labeled as "unknown" initially)
- Ball tracking
- Pass detection
- Foul, offside, or tactical event detection
- Automatic camera calibration (manual homography only)
- Handling camera cuts, replays, or close-up shots
- Formation estimation
- Distance covered calculation
- Heatmaps
- Interactive dashboard (static outputs only)

---

## Evaluation Criteria

| Criterion | Target |
|---|---|
| Player detection recall | ≥ 70 % of visible players detected |
| Player detection precision | ≥ 80 % (few false detections) |
| Team classification accuracy | ≥ 85 % of detections correctly assigned to team |
| Track stability | Most players keep the same ID for ≥ 5 consecutive seconds |
| Pitch projection | Projected positions visually plausible on top-down view |
| Processing completeness | Every frame in the clip is processed and recorded |

---

## Completion Criterion

Version 1 is complete when a user can:

1. Provide a supported clip
2. Run a single command
3. Receive an annotated output video and a CSV
4. Visually verify that most visible players are detected, tracked with stable IDs, and colour-coded by team on a top-down pitch view

---

*Version 1.0 — FootVision AI Phase 0*
