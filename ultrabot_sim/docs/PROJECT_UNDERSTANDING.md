# Project understanding (written before building the simulation)

UltraBot (`UltraBot-Trajectory-Tracking`, author Eshita Gupta) is a breast
ultrasound research prototype:

**ultrasound video → YOLO lesion detection → Center/IoU tracking → LSTM
next-position prediction → 2D biopsy trajectory planning (shortest / safest /
recommended) → robot simulation.**

## What the existing code does

| File | What it does (facts the simulation relies on) |
|---|---|
| `ultrabot/pipeline.py` | `run_pipeline()` reads a video, resizes every frame to the **990 × 720 "pane"** (`VIDEO_W × OUT_H`), runs `YOLO(models/best.pt)` with `conf=0.10`, converts boxes with `_yolo_detections()`, associates them with `CenterIoUTracker`, and runs the LSTM. `FrameRecord` stores per frame: detected, conf, class, `box_xyxy`, `raw_center`, `pred_center`, `pred_velocity`, `lstm_ready`, track id/state, consecutive hits. `load_models()` loads YOLO + the LSTM checkpoint. `generate_synthetic_pipeline_result()` makes a synthetic run. |
| `LSTMModel` / `predict_next()` | 3-layer LSTM, hidden 128, dropout 0.2, Linear 128→64→2. Input: the last **20 consecutive** centres of the primary track, normalised `(x/990, y/720)`. Output: a normalised **velocity**; next centre = current + velocity, clamped to the pane. It runs only after 20 consecutive real detections on the primary track. |
| `ultrabot/tracker.py` | Greedy IoU + nearest-centre association (min IoU 0.10, max centre distance 90 px). Unmatched tracks coast for up to 8 frames, then become lost. **Centres are never invented during misses.** The primary track is the live track with the most hits. |
| `ultrabot/biopsy.py` | `plan_biopsy()` scores one straight pixel segment; `candidate_entries()` samples the **four image borders**; `candidate_targets()` = lesion, LSTM prediction and 4 nearby points; `optimize_biopsy_paths()` keeps feasible rows and ranks a min-max normalised weighted cost. Shortest = minimum pixel length; **safest = fewest "safety violations" then smallest target error**; recommended = lowest cost. |
| `ultrabot/render.py` | `OUT_W = 1280`, `SIDEBAR_W = 290`, `VIDEO_W = 990`, `OUT_H = 720`, `FPS_OUT = 10`. |
| `ultrabot/robot_sim.py` | 2D GIF: a needle tip interpolated along the recommended pixel segment. |
| `ultrabot/metrics.py` | Per-run statistics, incl. the one-step LSTM error (prediction at t vs the real centre at t+1 on the same track). |
| `app.py` | Streamlit dashboard: Overview / Detection & Tracking / Temporal Prediction / Trajectory Planning / Virtual Biopsy / Robot Simulation tabs. Planning uses the **last detection** of the run; "NAVIGATE RECOMMENDED PATH" sends the recommended row to the 2D robot GIF. |
| `robotics/*` | Older 3D attempt: three.js page, matplotlib fallback, ROS message stubs, an approximate 6-joint arm; maps pixels linearly into a box with a **fixed lesion depth**. Not reused. |
| `data.yaml` | Classes `0: benign`, `1: malignant`. |
| `models/best.pt`, `models/lstm_checkpoint.pth`, `outputs/input.mp4` | Present. The sample video is 1673 frames at 5 fps, 692 × 692, and is a **concatenation of many short patient clips** (14 lesion tracks; the longest, frames 759-917, has 153 detections). |

## Weaknesses of the old planner that the simulation fixes

1. **"Safest" ignores anatomy.** It only means "smallest target error"; no
   structure is avoided. → The new planner maximises the *minimum clearance*
   to vessels, chest wall, ribs and pleura along the whole needle path.
2. **Impossible entries.** `candidate_entries()` uses all four image
   borders, including the **bottom border, which in a B-mode image is the
   chest-wall side**, and the top border lies under the transducer. → New
   entries are only on the skin, beside the probe, on both sides.
3. **No physical units.** Everything is in pixels. → Everything is planned
   and controlled in millimetres (configured scanner depth / width).
4. **Wrong depth assumption.** `robotics/coordinate_transform.py` uses a
   fixed lesion depth "because ultrasound has no depth". In a **B-mode image
   the vertical axis IS depth**; the simulation uses it.
5. **Bug.** `generate_synthetic_pipeline_result()` uses `math` but
   `pipeline.py` never imported it, so the dashboard's synthetic demo
   crashed. Fixed with one `import math` line (the only change to
   `pipeline.py`).

## Constraints respected

* `ultrabot/pipeline.py`, `tracker.py`, `biopsy.py` are imported, not
  changed (apart from the `import math` fix). The models are not retrained.
* The new perception loop is frame-by-frame but identical to
  `run_pipeline()`; a test checks that it reproduces `run_pipeline()`
  exactly on the sample video.
* No detection, prediction or metric is fabricated; a single image never
  gets an LSTM prediction.
* `app.py` gets exactly one new button ("Open 3D Robotic Simulation").
