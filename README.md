# UltraBot: Autonomous & Interactive 3D Robotic Ultrasound-Guided Breast Biopsy Simulation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](https://streamlit.io/)
[![PyBullet](https://img.shields.io/badge/Physics-PyBullet-black.svg)](https://pybullet.org/)

UltraBot is an end-to-end perception-in-the-loop autonomous and interactive simulation system for **robotic ultrasound-guided core-needle breast biopsy**. 

The system pairs deep learning perception (YOLO for lesion detection, Center-IoU temporal tracking, LSTM for motion forecasting) with an anatomy-aware 3D path planner and dual-arm robotic execution (simulated KUKA LBR iiwa arms in PyBullet and pyrender).

---

## 🌟 Key Capabilities

1. **Perception Pipeline:**
   - **YOLOv8** (`models/best.pt`): Detects and classifies breast lesions (benign vs. malignant) on ultrasound B-mode video.
   - **CenterIoU Tracker**: Associates detections continuously across frames.
   - **LSTM Trajectory Predictor** (`models/lstm_checkpoint.pth`): Predicts future lesion displacement during the needle insertion window to center biopsy gun firing accurately.

2. **Anatomy-Aware Trajectory Planner:**
   - Evaluates feasible skin entry sites outside the ultrasound probe footprint.
   - Computes **Shortest** (least tissue travel), **Safest** (maximum minimum clearance from vessels/chest wall/pleura), and **Recommended** (multi-objective weighted balance) trajectories.
   - Nonlinear refinement using SLSQP optimization.

3. **Dual-Arm Robotic Simulation:**
   - **Arm 1 (Probe Arm):** Holds ultrasound probe on the skin surface with active 6-DoF inverse kinematics.
   - **Arm 2 (Needle Arm):** Executes collision-free RRT-Connect transit, straight approach, remote center of motion (RCM) skin locking, and biopsy gun firing.

4. **Interactive Interfaces:**
   - **Streamlit Web Dashboard:** Interactive browser-based analytics, video replay, metrics charts, and 3D simulation panel.
   - **Surgeon Console GUI:** Real-time OpenCV / PyBullet / OpenGL desktop application with 11 camera angles, live clearance profiles, E-stop, and side-by-side comparison.

---

## 🚀 Quick Start

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Kusumm06/simulation.git
cd simulation

# PyTorch (CPU or CUDA)
pip install torch torchvision

# Core dependencies
pip install -r requirements.txt
```

### 2. Verify Installation (Self-Test)

Run the automated 51-point verification suite:

```bash
python -m ultrabot_sim.selftest
```

### 3. Run Streamlit Dashboard

Launch the browser interface:

```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 4. Run 3D Robotic Simulation (Standalone Window)

- **Autonomous Procedure Run:**
  ```bash
  python -m ultrabot_sim.main --auto
  ```

- **Shortest vs. Safest Path Comparison:**
  ```bash
  python -m ultrabot_sim.main --compare
  ```

- **Interactive Procedure (Surgeon Controls):**
  ```bash
  python -m ultrabot_sim.main
  ```

---

## 🎮 Hotkey Controls (Standalone GUI)

| Key | Action |
|---|---|
| `1` / `2` / `3` | Select **Shortest** / **Safest** / **Recommended** path |
| `Enter` | Confirm and begin robotic biopsy procedure |
| `c` | Run comparison mode |
| `p` | Freeze / unfreeze planning target |
| `Space` | Pause / resume |
| `z` / `r` | **E-Stop** / Resume and retract needle |
| `+` / `-` | Increase / decrease insertion velocity (1–10 mm/s) |
| `v` / `b` | Cycle 3D camera angles |
| `o` | Open zoomable 3D OR viewer window |
| `q` / `Esc` | Exit |

---

## 📂 Project Structure

```
├── app.py                      # Streamlit interactive web dashboard
├── requirements.txt            # Project dependencies
├── models/
│   ├── best.pt                 # YOLO breast lesion detection model
│   └── lstm_checkpoint.pth     # LSTM trajectory forecasting model
├── outputs/
│   └── input.mp4               # Ultrasound B-mode sample dataset
├── ultrabot/                   # Core UltraBot 2D pipeline
│   ├── pipeline.py             # Inference pipeline & model loading
│   ├── tracker.py              # Center-IoU lesion tracker
│   ├── biopsy.py               # 2D biopsy path planning algorithms
│   ├── render.py               # Video overlay & annotation rendering
│   └── robot_sim.py            # Kinematics helpers
└── ultrabot_sim/               # 3D Dual-Arm Robotic Simulation
    ├── main.py                 # Simulation entry point
    ├── scene.py                # PyBullet dual-arm KUKA scene & IK
    ├── planner.py              # Anatomy-aware 3D needle trajectory planner
    ├── anatomy.py              # Parametric tissue, fascia, vessel model
    ├── control.py              # Finite-state machine & safety supervisor
    ├── display.py              # Multi-panel surgeon console HUD
    ├── or_render.py            # OpenGL 3D operating room rendering
    ├── selftest.py             # Verification test suite
    └── tests/                  # Unit and integration tests
```

---

## ⚠️ Disclaimer

This software is an academic research simulation developed for computational robotics and perception research. It is a kinematic simulation and not a certified medical device; it is not intended for clinical use, diagnostics, or human surgical procedures.
