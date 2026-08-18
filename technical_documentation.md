# FootVision AI — Exhaustive Technical Documentation

This document serves as a living, highly detailed technical ledger for the **FootVision AI** system. As we advance through each phase of development, this document will be updated to explain precisely:
1. Which files do what (module breakdown).
2. How different pieces of code interact (architecture and data flow).
3. The underlying mathematics, APIs, and design decisions.

---

## Phase 1: Scaffolding, Core Config, & Environment

Phase 1 established a modular, clean, and reproducible architecture designed to grow from simple scripts into a production-ready system.

### 1.1 Folder Structure & Modular Architecture
The repository is segmented into four primary zones:
- `/config`: Houses project-wide parameters in a YAML schema, preventing hardcoding.
- `/src`: Contains reusable, decoupled libraries separated by mathematical or computational responsibility.
- `/scripts`: Contains procedural execution scripts corresponding to each developmental phase.
- `/data` and `/outputs`: Segmented storage zones that are omitted from version control (via `.gitignore`) to protect against pushing heavy media assets.

### 1.2 Core Config Schema (`config/config.yaml`)
To maintain consistency, all parameters (thresholds, model types, dimensions) are managed by `config/config.yaml`.
- **Paths**: Organizes absolute-relative resolving paths for inputs, frames, models, and analytics.
- **Video Config**: Manages resizing bounds and target frame-rates for preprocessing.
- **Detector Config**: Pre-sets confidence thresholds, model variants (`yolov8n.pt`), image dimensions, and filters (e.g. only matching the `"person"` class).
- **Teams Config**: Houses cluster targets ($K$-Means configuration) and vertical cropping percentages (excluding head/grass to focus on jersey colors).
- **Analytics Config**: Specifies proximity boundaries ($d \le 2.0\text{ meters}$) and temporal smoothing bounds.

---

## Phase 2: Video & Image Fundamentals (No-AI Processing)

Phase 2 focuses on decoding, handling, annotating, and encoding video data using raw computer vision principles before introducing neural network models.

```text
               [ Raw Source: Directory of JPEGs OR Video File ]
                                     │
                                     ▼
                      [ scripts/phase2_video_metadata.py ]
                        (Extract: W, H, FPS, Frames, Codec)
                                     │
                                     ▼
                      [ scripts/phase2_frame_extractor.py ]
                     (Sequential Reading & Array Inspection)
                                     │
                                     ▼
                   [ scripts/phase2_write_annotated_video.py ]
                     (Canvas drawing + VideoWriter Encoding)
                                     │
                                     ▼
                       [ Output Video / Captured Frames ]
```

### 2.1 Video Metadata Inspector (`scripts/phase2_video_metadata.py`)
This script isolates the parsing of video container formats. It is designed to handle two distinct input structures:
1. **Video Container Files (`.mp4`, `.avi`, etc.)**: Leverages OpenCV's `VideoCapture` API, querying the video header table using properties:
   - `cv2.CAP_PROP_FRAME_WIDTH`
   - `cv2.CAP_PROP_FRAME_HEIGHT`
   - `cv2.CAP_PROP_FPS`
   - `cv2.CAP_PROP_FRAME_COUNT`
   - `cv2.CAP_PROP_FOURCC` (converted bitwise into a string representation using `chr((fourcc >> shift) & 0xFF)`).
2. **Image Sequence Directories (e.g., MOT datasets)**: Performs manual directory parsing:
   - Filters files matching image extension tuples.
   - Reads the resolution of the first frame (`cv2.imread(path).shape[:2]`).
   - Resolves target playback frame rate dynamically by parsing `seqinfo.ini` via text parsing (`frameRate=X`), falling back to a base rate of `25.0` FPS.

### 2.2 Frame Extraction & Array Inspector (`scripts/phase2_frame_extractor.py`)
This script acts as the verification utility for memory layouts and sequential decoding.
- **Sequential Flow Control**: Integrates directory sorting via `sorted()` to guarantee strict chronological frame reads when processing image folders, matching the frame indexing of video containers.
- **Array Mechanics (NumPy Integration)**:
  - OpenCV decodes image frames directly into standard NumPy arrays (`np.ndarray`).
  - **Memory Layout**: A decoded frame has the shape `(H, W, 3)`, standing for `(height, width, channels)`.
  - **Color Representation**: OpenCV defaults to the **BGR** (Blue, Green, Red) format.
  - **Data Type**: Represented as `np.uint8` (unsigned 8-bit integers) bounding individual pixel values to $[0, 255]$.
  - The script prints out memory statistics using `frame.shape`, `frame.dtype`, `frame.min()`, and `frame.max()` to verify frame integrity.

### 2.3 Annotated Video Writer (`scripts/phase2_write_annotated_video.py`)
Implements manual canvas modification and multi-frame compilation.
- **Drawing Mechanics (In-place Array Modification)**:
  - Canvas mutations are executed in-place on individual frame arrays.
  - **Frame Counter**: Uses `cv2.putText` with `cv2.FONT_HERSHEY_SIMPLEX` and `cv2.LINE_AA` (Anti-Aliased) to prevent text jaggedness.
  - **Geometric Overlays**: Employs `cv2.rectangle` (bounding box demo), `cv2.circle` (center point detection demo), and `cv2.line` (trajectory baseline demo). Coordinates are mapped relative to the `(H, W)` canvas shape.
- **Encoder Setup (`cv2.VideoWriter`)**:
  - Requires four static inputs: output path, FourCC codec indicator, FPS, and frame dimensions `(W, H)`.
  - **FourCC Codec**: Uses `cv2.VideoWriter_fourcc(*"mp4v")` to output to a standard `.mp4` container.
  - **Frame Alignment**: OpenCV's encoder requires incoming frame dimensions to match the `(width, height)` dimension specified in the constructor. The script ensures no resizing mismatches occur during compilation.
  - **Resource Lifecycle**: Utilizes `.release()` on `cv2.VideoCapture` and `cv2.VideoWriter` to flush memory-mapped files and write headers.

---

## Phase 3: Manual Bounding-Box Exercise

Phase 3 builds coordinate math fundamentals for representing objects in image space, serving as the bridge to automated detectors and homography mapping.

### 3.1 Bounding Box Coordinate System
A standard 2D bounding box represents a rectangular region on a flat canvas, parameterized by top-left and bottom-right corners:
\[\text{Box} = [x_1, y_1, x_2, y_2]\]
where $x$ represents the column index (horizontal coordinate from left to right) and $y$ represents the row index (vertical coordinate from top to bottom) in the range:
\[0 \le x < W \quad \text{and} \quad 0 \le y < H\]

### 3.2 Coordinate Calculations (`scripts/phase3_manual_bbox.py`)
For any bounding box, key geometric attributes are derived as follows:
- **Dimensions**:
  \[\text{width} = x_2 - x_1\]
  \[\text{height} = y_2 - y_1\]
- **Center Point** $(x_c, y_c)$: Represents the centroid of the bounding box, useful for spatial tracking:
  \[x_c = \frac{x_1 + x_2}{2}, \quad y_c = \frac{y_1 + y_2}{2}\]
- **Bottom-Center Point** $(x_b, y_b)$: Represents the point where the player's feet touch the pitch. This coordinate is critical because it is projected onto the 2D pitch during camera calibration:
  \[x_b = \frac{x_1 + x_2}{2}, \quad y_b = y_2\]

### 3.3 Visual Overlays & Code Mechanics
The script loads `data/frames/frame_0000_first.jpg` and applies:
1. `cv2.rectangle`: yellow border outline `(0, 255, 255)` around player bounds.
2. `cv2.circle`: red dot `(0, 0, 255)` marking the centroid center.
3. `cv2.circle`: green dot `(0, 255, 0)` marking the feet point.
4. Outputs the calculations to stdout and saves the annotated frame to `outputs/phase3_manual_bbox.jpg` for validation.

---

## Phase 4: Pretrained Person Detector

Phase 4 moves beyond manual bounding boxes to automate player identification using deep-learning models trained on standard COCO labels.

### 4.1 YOLOv8 Inference Mechanics
We employ the **YOLOv8 nano** model (`yolov8n.pt`), an anchor-free single-stage detector optimized for rapid inference:
- **Weights**: Loaded dynamically from Ultralytics assets.
- **Data Flow**: Image array data of shape `(H, W, 3)` is loaded by OpenCV, then fed to YOLOv8. The model performs internal multi-scale scaling and outputs prediction tensors containing bounding box bounding coordinates, confidences, and class IDs.
- **COCO Class Filtering**: Pretrained YOLOv8 weights contain 80 classes. Outfield players, referees, and goalkeepers are all categorized under the `"person"` class (COCO index `0`). The script parses model prediction outputs and discards all other indexes.

### 4.2 Multi-Threshold Experiment Output
Evaluating `data/frames/frame_0000_first.jpg` with varying confidence thresholds illustrates the trade-off between Precision and Recall:
- **Threshold 0.20**: 19 detections (highest recall, captures distant/occluded objects, higher risk of false positives).
- **Threshold 0.40**: 16 detections (recommended pipeline baseline; clean separation of active outfield players).
- **Threshold 0.60**: 11 detections (misses smaller/occluded players).
- **Threshold 0.80**: 2 detections (retains only the top two prominent foreground players).

These bounding box outputs and associated scores are drawn using `cv2.rectangle` and `cv2.putText` and saved as custom validation image assets in the `/outputs` folder.

---

## Phase 5: Detection Evaluation Against Ground Truth

Phase 5 introduces quantitative evaluation of the detector by comparing its outputs against the MOT Challenge ground-truth annotations that ship with the SNMOT-062 SoccerNet sequence.

### 5.1 Data Flow Architecture

```text
data/raw/SNMOT-062/gt/gt.txt  →  load_ground_truth()  →  {frame_id: [[x1,y1,x2,y2], ...]}
data/raw/SNMOT-062/img1/      →  YOLO inference        →  det_boxes, det_confs
                                          ↓
                          match_detections_to_gt()   (IoU greedy matching)
                                          ↓
                            per-frame: TP, FP, FN, Precision, Recall
                                          ↓
                 outputs/phase5_evaluation_thresh20.csv  +  phase5_report_thresh20.txt
```

### 5.2 Ground-Truth Format: MOT Challenge
The file `gt/gt.txt` uses the standard MOT Challenge CSV format:
```
frame_id, track_id, x, y, w, h, conf, class_id, visibility, _
```
Coordinates are stored as `[x, y, w, h]` (top-left corner and dimensions) and converted to `[x1, y1, x2, y2]` during loading:
$$x_1 = x, \quad y_1 = y, \quad x_2 = x + w, \quad y_2 = y + h$$

### 5.3 Intersection over Union (IoU) Matching
A detection is classified as a True Positive only when it overlaps sufficiently with a ground-truth box. The overlap metric is **Intersection over Union (IoU)**:
$$\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|}$$
We use a greedy matching algorithm that iteratively assigns the highest-IoU detection-GT pair and removes both from the pool, preventing double-counting. Threshold: $\text{IoU} \ge 0.50$.

**Classification outcomes per box:**
- **True Positive (TP)**: Detection matched to a GT box (IoU ≥ 0.50)
- **False Positive (FP)**: Detection with no matching GT box (spectator, duplicate, artefact)
- **False Negative (FN)**: GT box with no matching detection (missed player)

### 5.4 Precision and Recall
$$\text{Precision} = \frac{TP}{TP + FP} \quad \text{Recall} = \frac{TP}{TP + FN}$$

### 5.5 Quantitative Results — SNMOT-062, Threshold=0.20, IoU≥0.50
Evaluated on 30 frames sampled uniformly across the 750-frame sequence:

| Metric | Value |
|---|---|
| Total TP | 442 |
| Total FP | 91 |
| Total FN | 95 |
| **Precision** | **82.9%** |
| **Recall** | **82.3%** |

**Recommended threshold: 0.20** — lower values maximize recall and ensure distant/small players are not missed, which is critical before tracking assigns IDs to every visible person.

### 5.6 Failure Pattern Analysis
- **Frame 26 — Precision 0.62, FP=8**: Highest false-positive frame. Likely caused by a camera pan or crowd elements entering the frame during a wide-zoom moment.
- **Frame 569 — Recall 0.63, FN=7**: Worst recall frame. Probable cause: players tightly clustered in the penalty area causing NMS to merge overlapping detections into single boxes.
- **Frame 595 — Recall 0.65, FN=6**: Similar pattern. Distant or off-axis players fall below the resolution threshold for reliable detection even at 0.20 confidence.
- **Frames 414, 517, 620, 672 — Precision 1.00**: Zero false positives on these frames indicate clear, well-separated players in a standard wide-angle tactical view.

### 5.7 Key Limitations
- General COCO model has no football-specific fine-tuning (no goalkeeper/referee class separation).
- NMS merges closely overlapping players into a single detection.
- Motion blur in fast-action frames reduces box quality.
- Distant players near the image boundary are consistently harder to detect.

---

## Phase 6: Full-Sequence Detection Pipeline

Phase 6 integrates all previous foundational modules into a unified, end-to-end processing pipeline that processes complete continuous video clips frame-by-frame and exports both annotated media and tabular dataset artifacts.

### 6.1 End-to-End Pipeline Architecture

```text
[ data/raw/SNMOT-062/img1 (750 frames) ]
                   │
                   ▼ (cv2.imread loop + seqinfo.ini fps)
   [ Phase 4 YOLOv8 Detector (conf >= 0.20) ] ──> Filter 'person'
                   │
                   ▼
     [ Phase 3 Coordinate Calculations ]
       - (xc, yc) centroid: ((x1+x2)/2, (y1+y2)/2)
       - (xb, yb) pitch foot contact: ((x1+x2)/2, y2)
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
[ Canvas Visual Overlay ]  [ Tabular CSV Serializer ]
- Bounding box + banner    - Frame number & timestamp
- Feet green circle        - Detection ID per frame
- HUD stats bar            - Bounding box [x1, y1, x2, y2]
         │                 - Center & bottom-center coords
         ▼                   │
[ cv2.VideoWriter (.mp4) ]   ▼
                     [ outputs/SNMOT-062_phase6_detections.csv ]
```

### 6.2 Data Schema: Tabular Detections (`.csv`)
For every detected player in each video frame, a structured row is written:

| Column Header | Data Type | Description |
|---|---|---|
| `frame_number` | `int` | 1-indexed video frame counter |
| `timestamp` | `float` | Elapsed time in seconds ($t = \frac{\text{frame\_idx}}{\text{fps}}$) |
| `detection_id` | `int` | Per-frame detection counter ($1 \dots N_{\text{dets}}$) |
| `class_name` | `string` | Object class (`"person"`) |
| `confidence` | `float` | Model confidence score ($0.20 \le c \le 1.0$) |
| `x1`, `y1` | `int` | Top-left corner coordinates |
| `x2`, `y2` | `int` | Bottom-right corner coordinates |
| `center_x`, `center_y` | `float` | Calculated bounding box centroid |
| `bottom_center_x`, `bottom_center_y` | `float` | Estimated foot-ground contact coordinate |

### 6.3 Performance & Throughput Metrics (SNMOT-062 on GTX 1650)
- **Total Frames Processed**: 750 (30.0 seconds of footage @ 25.0 FPS)
- **Total Detections Logged**: 13,183 players
- **Average Detections per Frame**: 17.58 players
- **Total Processing Wall Time**: 79.84 seconds
- **Average Pipeline Throughput**: **9.39 FPS**
- **Inference Time**: ~72.82 ms/frame
- **Drawing & IO Overhead**: ~4.76 ms/frame

### 6.4 Generated Deliverables
1. **Annotated MP4 Video**: `outputs/SNMOT-062_phase6_detections.mp4`
2. **Detection Records**: `outputs/SNMOT-062_phase6_detections.csv`
3. **Execution Summary Report**: `outputs/SNMOT-062_phase6_report.txt`

---

## Phase 7: Multi-Object Player Tracking (ByteTrack)

Phase 7 transitions the system from isolated frame-by-frame detection to temporal tracking by maintaining persistent identities (`track_id`) for players over continuous sequences.

### 7.1 Detection vs. Tracking Paradigm
- **Detection**: Solves spatial localization within isolated images without temporal memory:
  $$\text{Frame } t \implies \{\text{Box}_1, \text{Box}_2, \dots, \text{Box}_N\}$$
- **Tracking**: Solves data association across consecutive frames by correlating movement and appearance:
  $$\text{Frame } t \implies \{(\text{Track ID } k_1, \text{Box}_{k1}), (\text{Track ID } k_2, \text{Box}_{k2}), \dots\}$$

### 7.2 ByteTrack Mathematical Architecture
ByteTrack solves data association by splitting detections into two confidence pools rather than discarding low-confidence detections:

```text
Incoming Detections (Frame t)
        │
        ├──> High-Score Pool (conf >= 0.50) ──> Matched with existing tracks via Kalman Filter & IoU
        │                                         │ (Unmatched tracks remain active)
        │                                         ▼
        └──> Low-Score Pool (0.10 <= conf < 0.50) ──> Matched with remaining unmatched tracks
                                                   (Recovers occluded/blurred players)
```

1. **Kalman Filter Motion Prediction**:
   Each track maintains an 8-dimensional state vector estimating bounding box geometry and velocity:
   $$\mathbf{x} = [x_c, y_c, a, h, \dot{x}_c, \dot{y}_c, \dot{a}, \dot{h}]^T$$
   where $(x_c, y_c)$ is center position, $a = w/h$ is aspect ratio, and $h$ is height.
2. **Linear Assignment Problem**:
   Matched using the Hungarian Algorithm (LAP) with IoU cost matrices.
3. **Trajectory Trail Buffering**:
   Historical $(x_b, y_b)$ foot coordinates are stored in a ring buffer (`deque(maxlen=30)`), rendering anti-aliased fading trail lines that indicate player motion trajectories.

### 7.3 Data Schema: Tabular Track Records (`.csv`)
Saved to `outputs/SNMOT-062_phase7_tracks.csv`:

| Column Header | Data Type | Description |
|---|---|---|
| `frame_number` | `int` | 1-indexed video frame index |
| `timestamp` | `float` | Sequence playback time in seconds |
| `track_id` | `int` | Persistent object identifier assigned by ByteTrack |
| `class_name` | `string` | Object class (`"person"`) |
| `confidence` | `float` | Detection confidence |
| `x1, y1, x2, y2` | `int` | Tracked bounding box coordinates |
| `center_x, center_y` | `float` | Centroid coordinates |
| `bottom_center_x, bottom_center_y` | `float` | Ground-contact coordinates for pitch mapping |

### 7.4 Continuity & Performance Metrics (SNMOT-062)
- **Total Track Detections**: 11,778 player positions
- **Average Active Players**: 15.70 per frame
- **Average Track Lifetime**: 2.49 seconds (62.3 frames)
- **Median Track Lifetime**: 1.04 seconds
- **Stable Tracks ($\ge 5$ s)**: 36 of 209 (17.2%)
- **Pipeline Throughput**: **7.94 FPS** (including YOLOv8 inference, ByteTrack association, trajectory trail rendering, video encoding, and CSV logging)

### 7.5 Output Deliverables
1. **Annotated Track Video**: `outputs/SNMOT-062_phase7_tracking.mp4`
2. **Track Dataset Table**: `outputs/SNMOT-062_phase7_tracks.csv`
3. **Tracking Continuity Report**: `outputs/SNMOT-062_phase7_report.txt`

---

## Phase 8: Team, Referee & Staff/Goalkeeper Kit Classification

Phase 8 implements multi-class kit and role classification to separate:
- **Team A**: White / Navy outfield players
- **Team B**: Green / White outfield players
- **Referee**: Yellow / Gold uniforms
- **Staff / Goalkeeper**: Black / Dark attire

### 8.1 Modular Subsystem Architecture

```text
[ Detected Player Bounding Box (x1, y1, x2, y2) ]
                      │
                      ▼
        [ src/teams/crop_extractor.py ]
        - Spatial chest cropping (top: 15%, bottom: 50%, side margins: 20%)
        - Eliminates background pitch & shorts without deleting green jerseys
                      │
                      ▼ Chest image patch
       [ src/teams/colour_features.py ]
        - Extracts LAB (L, a, b) and HSV (H, S, V) median color metrics
                      │
                      ▼ Metric Dict {L, a, b, H, S, V}
         [ src/teams/classifier.py ]
        - Multi-domain boundary decisions:
          1. Black/Dark Staff/GK check (L < 0.35 or V < 0.30)
          2. Yellow Referee check (H in [18, 36], b > -a, S >= 0.35)
          3. Green Team B check (H in [36, 85], -a > b, S >= 0.16)
          4. White Team A check (L >= 0.55, S < 0.28)
        - Temporal sliding-window majority voting per track_id
                      │
         ┌────────────┴────────────┐
         ▼                         ▼
[ Role-Colored Visual Overlay ]  [ Tabular Dataset Serializer ]
- Team A: White / Light Gray     - frame_number, timestamp, track_id
- Team B: Bright Green           - team_label ("Team A", "Team B", "Referee", "Staff/GK")
- Referee: Bright Yellow/Gold    - confidence, bounding box & feet coordinates
- Staff/GK: Dark Black/Gray      - CSV output
         │                         │
         ▼                         ▼
[ outputs/SNMOT-062_phase8_teams.mp4 ]  [ outputs/SNMOT-062_phase8_teams.csv ]
```

### 8.2 Spatial Chest Extraction (Green Kit Preservation)
To prevent deleting green jerseys (Team B) while isolating the kit from background pitch grass and dark shorts:
- **Vertical bounds**: $[y_1 + 0.15h, y_1 + 0.50h]$ (pure upper-chest area, avoiding head and shorts).
- **Horizontal bounds**: $[x_1 + 0.20w, x_2 - 0.20w]$ (inner 60% chest width, omitting outer background grass).

### 8.3 Chromaticity & Luminance Decision Mechanics
1. **Staff & Goalkeepers (Black/Dark)**:
   $$L < 0.35 \quad \lor \quad V < 0.30$$
2. **Referee (Yellow)**:
   $$H \in [18, 36], \quad S \ge 0.35, \quad b > -a$$
   The positive $b$-axis (Yellow) strictly dominates the negative $a$-axis (Green).
3. **Team B (Green / White)**:
   $$(H \in [36, 85] \land S \ge 0.16) \quad \lor \quad (a < -0.04 \land S \ge 0.14) \quad \lor \quad (-a > b \land S \ge 0.16)$$
   The negative $a$-axis (Green) dominates the positive $b$-axis (Yellow).
4. **Team A (White / Navy)**:
   $$L \ge 0.55 \quad \land \quad S < 0.28$$
   Neutral luminance and low saturation.

### 8.4 Tabular Dataset Schema (`outputs/SNMOT-062_phase8_teams.csv`)

| Column Header | Data Type | Description |
|---|---|---|
| `frame_number` | `int` | 1-indexed video frame counter |
| `timestamp` | `float` | Playback time in seconds |
| `track_id` | `int` | Persistent object identifier from ByteTrack |
| `team_label` | `string` | Role label (`"Team A"`, `"Team B"`, `"Referee"`, `"Staff/GK"`) |
| `class_name` | `string` | Object class (`"person"`) |
| `confidence` | `float` | Detection confidence |
| `x1, y1, x2, y2` | `int` | Player bounding box coordinates |
| `center_x, center_y` | `float` | Bounding box center coordinates |
| `bottom_center_x, bottom_center_y` | `float` | Ground-contact coordinates for pitch projection |

### 8.5 Deliverables
- `outputs/SNMOT-062_phase8_teams.mp4`
- `outputs/SNMOT-062_phase8_teams.csv`
- `outputs/SNMOT-062_phase8_report.txt`


---

## Phase 9: Pitch Calibration & 2D Tactical Radar Mapping

Phase 9 is the geometric intelligence layer of FootVision AI. It lifts every tracked player from pixel space into real-world pitch metric space (in metres) using a projective homography calibrated from manually-clicked pitch-line landmarks. The output is a side-by-side composite video — broadcast frame on the left, top-down 2D tactical radar on the right — along with a CSV of all player positions expressed in pitch coordinates across every frame.

---

### 9.1 Files Introduced

| File | Role |
|---|---|
| `src/calibration/pitch_model.py` | FIFA pitch geometry constants and top-down pitch canvas drawing engine |
| `src/calibration/homography.py` | Homography computation, projection, and save/load utilities |
| `src/visualization/pitch_plots.py` | 2D radar renderer and side-by-side composite frame builder |
| `scripts/phase9_pick_landmarks.py` | One-time interactive calibration tool (mouse-click landmark picker) |
| `scripts/phase9_pitch_radar.py` | Full Phase 9 pipeline: YOLOv8 + ByteTrack + kit classification + homography + composite video |

---

### 9.2 The Core Concept: Planar Homography

#### Why a Homography?

A football pitch is a planar surface. Under the assumption that all players' feet touch that plane (which is valid for a broadcast wide-angle shot of a flat pitch), the relationship between a pixel coordinate $(x_{px}, y_{px})$ in the camera image and a real-world coordinate $(X_m, Y_m)$ in pitch metric space is a **projective transformation** (homography):

$$\begin{pmatrix} w \cdot X_m \\ w \cdot Y_m \\ w \end{pmatrix} = \mathbf{H} \begin{pmatrix} x_{px} \\ y_{px} \\ 1 \end{pmatrix}$$

where $\mathbf{H}$ is a $3 \times 3$ real matrix with 8 degrees of freedom (homogeneous, so the overall scale is irrelevant), and $w$ is the projective weight. After solving for $\mathbf{H}$, any pixel on the ground plane can be converted to metres with a simple matrix–vector multiplication and a perspective divide.

#### Why the Foot Coordinate?

Players are volumetric objects, but we want to find where on the pitch plane they are standing. The **foot coordinate** is the bottom-center of the bounding box: $(x_{foot} = (x_1 + x_2)/2, \; y_{foot} = y_2)$. This is the pixel that lies closest to the ground contact point, making it the most geometrically meaningful location to pass through the homography.

---

### 9.3 `src/calibration/pitch_model.py` — Pitch Geometry Constants

This module acts as the *ground truth registry* for all pitch knowledge.

**Constants defined:**

| Symbol | Value | Meaning |
|---|---|---|
| `PITCH_LENGTH` | 105.0 m | FIFA standard pitch length (x-axis) |
| `PITCH_WIDTH` | 68.0 m | FIFA standard pitch width (y-axis) |
| `PENALTY_BOX_DEPTH` | 16.5 m | Distance from goal line to penalty area boundary |
| `PENALTY_BOX_WIDTH` | 40.32 m | Width of penalty area |
| `GOAL_AREA_DEPTH` | 5.5 m | Depth of 6-yard box |
| `CENTER_CIRCLE_RADIUS` | 9.15 m | Centre circle radius |
| `PENALTY_SPOT_DIST` | 11.0 m | Penalty spot from goal line |

**`PITCH_LANDMARKS` dict:** A registry of all pitch-marking intersection positions in metric space. Each entry maps a descriptive name (e.g. `"left_penalty_top_right"`) to `(X_meters, Y_meters)`. There are 25 named landmarks covering all standard pitch markings visible in broadcast footage. This dictionary is the bridge between the human-clickable names in the picker UI and the mathematical point correspondences used by `cv2.findHomography`.

**`draw_pitch(canvas_width, canvas_height, ...)` function:** Renders a top-down pitch diagram on an OpenCV NumPy array using the following sequence:
1. Fills the canvas with grass green background.
2. Draws the outer pitch boundary rectangle.
3. Draws the halfway line.
4. Draws the centre circle and centre spot.
5. Draws both penalty areas and goal areas as rectangles.
6. Draws penalty arcs (partial ellipses arcing out from penalty spots).
7. Draws penalty spots as 3-pixel filled circles.
8. Draws corner arcs as quarter-ellipses at each pitch corner.
9. Draws simplified goal boxes.

The `m2p(x_m, y_m)` inner function handles the coordinate system conversion: pitch metres → canvas pixels by linear scaling (`px = x_m / PITCH_LENGTH * canvas_width`).

---

### 9.4 `src/calibration/homography.py` — Homography Math

#### `compute_homography(image_points, pitch_points) -> np.ndarray`

Calls `cv2.findHomography(src, dst, cv2.RANSAC, ransacReprojThreshold=5.0)`.

- **Input**: Two parallel lists: pixel coordinates from the camera frame and their corresponding metric coordinates on the real pitch.
- **Algorithm**: The **Direct Linear Transformation (DLT)** formulation sets up a system of $2n$ linear equations for $n$ point pairs, then solves for the 8 independent elements of $\mathbf{H}$ via SVD. OpenCV's RANSAC wrapper adds robustness: it repeatedly samples 4-point subsets, computes a candidate $\mathbf{H}$, counts inliers (points with reprojection error $< 5$ pixels), and returns the $\mathbf{H}$ with the largest inlier count.
- **Minimum requirement**: 4 point correspondences (sufficient to solve the 8-DoF system). Each additional correspondence over-constrains the system and improves accuracy.

#### `project_point(H, x_px, y_px) -> (X_m, Y_m)`

Applies the homography using OpenCV's `cv2.perspectiveTransform`. Internally:
1. Forms a $1 \times 1 \times 2$ shaped float64 NumPy tensor (required input shape).
2. Multiplies by $\mathbf{H}$ and performs the perspective divide.
3. Returns `(X_m, Y_m)` as a Python tuple of floats.

#### `project_points_batch(H, points_px) -> list`

Batch-projects a list of pixel coordinates in a single `cv2.perspectiveTransform` call. This is significantly faster than looping `project_point` for every detection in every frame.

#### `compute_reprojection_error(H, image_points, pitch_points) -> float`

Validates calibration quality. Projects each clicked landmark pixel coordinate through $\mathbf{H}$ and computes the Euclidean distance to the known metric coordinate. Returns the mean error in metres. Acceptable threshold: $< 3.0$ metres.

#### Save/Load

`save_homography(H, path)` and `load_homography(path)` wrap `np.save`/`np.load` for the $3 \times 3$ float64 matrix. The matrix is stored as `outputs/homography.npy`.

---

### 9.5 `scripts/phase9_pick_landmarks.py` — Interactive Calibration Tool

This is the **one-time setup step** that must be run before the main pipeline.

**Workflow:**
1. Opens the first frame of the sequence (1920x1080) in an OpenCV named window.
2. A `LandmarkPicker` object registers a `cv2.setMouseCallback` on the window.
3. The top HUD bar displays which landmark to click next, pulling descriptions from `LANDMARK_SEQUENCE` (an ordered list of 13 pitch-marking intersections).
4. On each left click (`cv2.EVENT_LBUTTONDOWN`), the pixel coordinate is appended to `self.clicked` and a green circle is drawn at that location.
5. Key bindings:
   - **ENTER**: Compute and save homography (requires $\ge 4$ clicks).
   - **Z**: Undo last click.
   - **ESC**: Cancel without saving.
6. After confirmation, `compute_homography` is called with the collected correspondences, reprojection error is printed, and the matrix is saved to `outputs/homography.npy`.
7. A preview window shows the 2D pitch diagram with the clicked landmarks projected as cyan dots, allowing visual verification.

The quality of calibration is entirely dependent on the accuracy of the user's clicks. Clicking precisely on line intersections (where two pitch marking lines cross) gives the best results.

---

### 9.6 `scripts/phase9_pitch_radar.py` — Full Pipeline

This script orchestrates the complete Phase 9 pass over all 750 frames.

#### Data Flow

```
img1/XXXXXX.jpg
      |
      v
  YOLO model.track()          --> raw detections + ByteTrack IDs
      |
      v
  extract_torso_crop()        --> jersey region crop
      |
      v
  extract_chest_color_metrics()  --> {L, a, b, H, S, V}
      |
      v
  MatchKitClassifier.update_track()  --> team_label (with temporal smoothing)
      |
      v
  project_point(H, foot_x, foot_y)  --> (pitch_x_m, pitch_y_m)
      |
      v
  CSV row: frame_num, timestamp, track_id, label, bbox, foot_px, pitch_meters
      |
      v
  annotate_broadcast_frame()   --> bounding boxes + labels on raw frame
      |
      v
  draw_pitch_radar()           --> 2D top-down radar canvas
      |
      v
  build_composite_frame()      --> side-by-side broadcast | radar
      |
      v
  VideoWriter.write()          --> outputs/SNMOT-062_phase9_tactical_radar.mp4
```

#### `annotate_broadcast_frame(frame, tracks, labels)`

Draws coloured bounding boxes and overlaid tags for each detected track. The tag format is `#<track_id> <team_label>` (e.g. `#12 Team A`). A filled rectangle behind the text makes it readable over any background.

#### `draw_broadcast_hud(frame, frame_num, timestamp, label_counts)`

Adds a dark bar at the top of the broadcast panel showing frame number, timestamp, and live team count summary.

#### `draw_pitch_radar(player_positions, ...)`

Calls `draw_pitch()` for the green base canvas, then iterates over `player_positions` to draw each player as a filled circle (radius 6 px) with a white border. The dot color encodes team: Team A = light red, Team B = green, Referee = yellow, Staff/GK = gray. Track IDs are printed next to each dot. A legend in the top-left corner identifies the color scheme. Frame number and timestamp are shown in the bottom-left corner.

#### `build_composite_frame(broadcast_frame, radar_frame, target_height=540)`

Scales both panels independently to `target_height=540 px` maintaining aspect ratio, inserts a 4-pixel white divider between them, and horizontally concatenates them with `np.hstack`. The result is a widescreen composite video at a resolution of approximately `1530x540` pixels.

#### Output Files

| File | Contents |
|---|---|
| `outputs/SNMOT-062_phase9_tactical_radar.mp4` | Side-by-side composite video at 25 FPS |
| `outputs/SNMOT-062_phase9_pitch_coords.csv` | Columns: `frame_number, timestamp_s, track_id, team_label, bbox_x1, bbox_y1, bbox_x2, bbox_y2, foot_x_px, foot_y_px, pitch_x_m, pitch_y_m` |
| `outputs/homography.npy` | 3x3 float64 homography matrix from the calibration step |

---

### 9.7 `src/visualization/pitch_plots.py` — Radar Rendering

#### `TEAM_DOT_COLORS` and `TEAM_TEXT_COLORS`

Dictionaries mapping team label strings to BGR tuples used for radar dots and legend text respectively:
- `"Team A"`: `(200, 200, 255)` — Light blue-white (visually distinct from Team B)
- `"Team B"`: `(50, 200, 50)` — Green
- `"Referee"`: `(0, 215, 255)` — Yellow
- `"Staff/GK"`: `(128, 128, 128)` — Gray

#### `_meters_to_radar_px(x_m, y_m, w, h, margin)`

Converts pitch metre coordinates into pixel coordinates on the radar canvas. Uses `margin=20` pixels of padding on all sides so dots near the touchline are not clipped:
```
px = margin + clip(x_m / PITCH_LENGTH, 0, 1) * (canvas_w - 2*margin)
py = margin + clip(y_m / PITCH_WIDTH,  0, 1) * (canvas_h - 2*margin)
```
Players whose projected coordinates fall outside `[0, PITCH_LENGTH] x [0, PITCH_WIDTH]` are silently skipped (out-of-bounds guard).

---

### 9.8 Calibration Quality and Limitations

**Accuracy**: A well-calibrated homography with 8-10 carefully clicked landmarks typically achieves a mean reprojection error of 1-2 metres on a wide-angle broadcast frame. This is sufficient for tactical analysis (team shape, player spacing) but insufficient for millimetre-accurate offside adjudication.

**Assumptions**:
- The camera is static (fixed position, no pan/tilt/zoom). If the camera moves, the homography must be recalibrated for each camera position.
- All feet are on the pitch plane. Players jumping or airborne will have temporarily incorrect projections.
- Lens distortion is ignored. For wide-angle lenses with significant barrel distortion, pre-undistorting the frame with a calibrated camera intrinsic matrix before homography application would improve accuracy.

**Improvement path**: In a future phase, automatic landmark detection (using a pitch segmentation network or keypoint detector trained on pitch markings) could replace the manual clicking step, enabling dynamic re-calibration as the camera pans.
