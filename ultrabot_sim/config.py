"""
All settings of the UltraBot 3D robotic biopsy simulation.

Units: metres / seconds / radians, unless the name says otherwise
(_MM = millimetres, _DEG = degrees, _PX = pixels, _S = seconds).

The simulation imports the existing UltraBot pipeline (ultrabot/) for
YOLO detection, tracking and the LSTM. Nothing here changes that code.
"""
import os
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- paths
PKG_DIR = Path(__file__).resolve().parent
ROOT = PKG_DIR.parent                              # repository root
RUNS_DIR = PKG_DIR / "runs"
ASSET_DIR = str(PKG_DIR / "assets")
DEFAULT_VIDEO = ROOT / "outputs" / "input.mp4"
DEFAULT_BUNDLE = ROOT / "outputs" / "sim_bundle.json"

DISCLAIMER = ("Kinematic research simulation, not a medical device. "
              "Not for clinical use.")


def _pick_device():
    """CUDA if the CUDA build of torch is installed, else CPU (like the
    reference project). YOLO and the LSTM both follow this."""
    try:
        import torch
        if torch.cuda.is_available():
            return 0
    except Exception:                                       # noqa: BLE001
        pass
    return "cpu"


DEVICE = _pick_device()

# ---------------------------------------------------------------- perception
# Same values as the Streamlit dashboard / ultrabot.pipeline.run_pipeline.
CONF = 0.10                     # YOLO confidence threshold
TRACK_MAX_MISSED = 8
TRACK_MAX_CENTER_DIST = 90.0    # px
TRACK_MIN_IOU = 0.10
TRAIL_LEN = 40                  # centres drawn as the track history

# The pane every coordinate lives in (ultrabot.render: VIDEO_W x OUT_H).
PANE_W_PX = 990
PANE_H_PX = 720

# ---------------------------------------------------------------- ultrasound
# The mm scale is a CONFIGURED scanner preset, not measured from the video.
US_DEPTH_MM = 40.0              # imaging depth: top -> bottom of the pane
US_WIDTH_MM = 50.0              # linear probe footprint: left -> right
US_SLICE_MM = 1.0               # elevational slice thickness (beam width)
PROBE_LEN = 0.125               # m, flange -> transducer face
PROBE_BODY = (0.056, 0.016, 0.105)   # m, housing extents (x lateral, y, z)
PROBE_CLEARANCE_MM = 4.0        # gun body must stay this far from the probe

# ---------------------------------------------------------------- anatomy
# Parametric phantom registered to the image plane. All depths are below
# the transducer face (probe frame +z), mm. NOT detected from the image.
SKIN_THICK_MM = 2.0
FAT_THICK_MM = 6.0              # subcutaneous fat: 2 -> 8 mm below the skin
PECTORAL_DEPTH_MM = 30.0        # pectoral fascia under the probe (default)
PECTORAL_MIN_BELOW_LESION_MM = 6.0   # fascia pushed this far below the lesion
MUSCLE_THICK_MM = 8.0           # pectoralis major
RIB_RADIUS_MM = 5.0
RIB_PITCH_MM = 26.0             # rib-to-rib spacing along the probe
RIB_OFFSET_MM = 7.0             # rib centre phase along x
PLEURA_GAP_MM = 3.0             # pleura below the rib centres + radius
# Breast surface beside the probe: flat under the transducer (compressed),
# falling away outside it. depth(d) = H * (1 - cos(min(d / D, 1) * pi / 2)),
# d = lateral distance from the probe edge.
BREAST_HEIGHT_MM = 32.0         # H: breast thickness at the scan site
BREAST_FALLOFF_MM = 110.0       # D
# Superficial vessels: tubes running out of the image plane, given in the
# probe frame (x_mm, depth_mm, radius_mm). A vessel that would overlap the
# lesion (plus VESSEL_MARGIN_MM) is moved sideways and the move is logged.
VESSELS = [(31.0, 5.0, 1.0), (-29.0, 15.0, 1.2)]
# Cooper's ligaments: thin oblique bands from the fascia up to the skin,
# (x at the fascia, x at the skin) in mm.
LIGAMENTS = [(-46.0, -34.0), (-20.0, -8.0), (8.0, 19.0), (33.0, 47.0),
             (58.0, 71.0)]
LIGAMENT_HALF_WIDTH_MM = 0.4
# Nipple-areola complex (entry forbidden): centre x in the probe frame.
NIPPLE_X_MM = -48.0
NIPPLE_R_MM = 16.0
# Image-assisted chest-wall estimate (--auto-chestwall)
AUTO_CHESTWALL = False
CHESTWALL_MIN_CONFIDENCE = 0.55

# ---------------------------------------------------------------- needle
NEEDLE_GAUGE = "14G"
NEEDLE_OD_MM = 2.1              # 14 G
NEEDLE_LEN_MM = 100.0           # exposed cannula beyond the gun nose
THROW_MM = 22.0                 # biopsy-gun throw
NOTCH_MM = 19.0                 # specimen notch length
GRIP_MARGIN_MM = 10.0           # needle that must stay outside the skin
NEEDLE_MOUNT_DEG = 30.0         # gun bracket: needle axis tilted from flange z
GUN_LEN = 0.135                 # m, gun body behind the needle hub
GUN_RADIUS = 0.013              # m, gun body half-width (needle exits the nose centre)
FIRE_TIME_S = 0.10              # stylet then cannula
FIRE_SLOWMO = 10.0              # slow-motion factor for the throw animation
HOLD_S = 2.0
RETRACT_SPEED_MM_S = 5.0
PRE_ENTRY_MM = 10.0             # tip this far before the skin after approach

# ---------------------------------------------------------------- planning
LESION_ON = "pred"              # aim at the LSTM time-of-arrival prediction
ENTRY_MIN_OFFSET_MM = 5.0       # entry distance outside the probe edge
ENTRY_MAX_OFFSET_MM = 60.0
ENTRY_STEP_MM = 0.5
MAX_ANGLE_TO_CHESTWALL_DEG = 35.0
MIN_SKIN_ANGLE_DEG = 8.0        # puncture must not be tangential to the skin
MIN_INSERTION_MM = 5.0
CHESTWALL_CLEARANCE_MM = 5.0
VESSEL_MARGIN_MM = 3.0
RIB_MARGIN_MM = 5.0
PLEURA_MARGIN_MM = 5.0
SAMPLE_STEP_MM = 0.5            # clearance profile resolution
PLAN_BUFFER_MM = 0.25           # refined paths keep this extra clearance
SAFETY_TOL_MM = 0.10            # numeric tolerance of the live checks
TARGET_TOL_MM = 1.0             # refined path may pass this far from the aim
REFINE_TOP_K = 3                # candidates refined with scipy.optimize
SAFETY_MARGIN_MM = 2.0          # LSTM uncertainty fallback (< 5 errors)
LSTM_MAX_ROLLOUT = 10
# "Predicted miss" = how far the throw axis would pass OUTSIDE the central
# part of the lesion (radius CORE_FRACTION x the lesion half-size across the
# needle). 0 = the notch goes through the lesion core.
CORE_FRACTION = 0.5
MOTION_WARN_MM = 1.5
MISS_ABORT_MM = 2.5
MISS_ABORT_FRAMES = 3            # abort only if it persists (debounce)
SIDE_STABLE_FRAMES = 5           # plan side must be stable before moving
PREFIRE_REAIM_MM = 1.5          # re-adjust depth at pre-fire only beyond this
LESION_SMOOTH_FRAMES = 5        # median of the last N REAL detections
LOST_ABORT_FRAMES = 8
DRIVE_PLANNER = "anatomy"       # "anatomy" | "legacy"
PLAN_BUDGET_MS = 200.0
# Recommended-path weights (min-max normalised terms, original UltraBot
# style). Presets cycle with [ and ].
WEIGHT_PRESETS = {
    "balanced":        dict(w_len=1.0, w_clear=1.0, w_angle=1.0, w_lig=0.5,
                            w_motion=1.0, w_target=1.0),
    "clearance-first": dict(w_len=0.5, w_clear=3.0, w_angle=1.0, w_lig=1.0,
                            w_motion=1.0, w_target=1.0),
    "length-first":    dict(w_len=3.0, w_clear=0.5, w_angle=1.0, w_lig=0.5,
                            w_motion=1.0, w_target=1.0),
}
WEIGHT_PRESET = "balanced"

# ---------------------------------------------------------------- scene (world, m)
# Patient supine, head toward +x, patient's RIGHT side = +y.
TABLE_TOP_Z = 0.78
TABLE_PAD = 0.05
TABLE_LEN = 2.02
TABLE_CENTER_X = 0.0
TABLE_HALF_W = 0.27
PATIENT_HEIGHT = 1.66
PATIENT_PELVIS_X = 0.0
CHEST_Z = 1.035                  # chest skin at the scan site (body is fitted)
# Probe face centre on the right breast (the transducer touches the skin
# here); the breast mound sits on the chest so that its top is here.
PROBE_SITE = np.array([0.355, 0.105, CHEST_Z + BREAST_HEIGHT_MM / 1000.0])
# Probe frame axes in the world: x_p (image right) toward the head,
# z_p (depth) straight down. y_p = z_p x x_p.
PROBE_X_AXIS = np.array([1.0, 0.0, 0.0])
PROBE_Z_AXIS = np.array([0.0, 0.0, -1.0])
# Robot carts (KUKA iiwa base frames)
CART_PROBE = np.array([0.33, -0.56, 0.80])     # patient's left: probe arm
CART_NEEDLE = np.array([0.36, 0.64, 0.80])     # patient's right: needle arm
ARM_MIN_CLEARANCE = 0.06        # m, arm-arm and arm-patient
ARM_CLEAR_QUERY = 0.30
EE_LINK = 6
JOINT_LOWER = [-2.96, -2.09, -2.96, -2.09, -2.96, -2.09, -3.05]
JOINT_UPPER = [2.96, 2.09, 2.96, 2.09, 2.96, 2.09, 3.05]
JOINT_MARGIN_DEG = 5.0
MAX_JOINT_VEL = np.radians(60.0)   # rad/s, per joint
MAX_TCP_SPEED = 0.250           # m/s, tool point, free-space transit
APPROACH_FINAL_SPEED = 0.050    # m/s, last STANDOFF_MM straight to the skin
STANDOFF_MM = 60.0              # transit ends this far back along the axis
INSERT_SPEED_MM_S = 3.0         # 1 .. 10
INSERT_ACCEL_MM_S2 = 6.0
IK_ITERS = 80
IK_DAMPING = 0.02
IK_POSTURE_GAIN = 0.02
IK_TOL = 1e-4                   # m: 0.1 mm position accuracy
IK_AXIS_TOL_DEG = 0.05
RRT_MAX_ITERS = 4000
RRT_STEP = 0.18                 # rad
RRT_CHECK_STEP = 0.04           # rad between collision checks
RRT_LINK_MARGIN = 0.020         # m, arm links vs obstacles
RRT_TOOL_MARGIN = 0.008         # m, gun / needle vs obstacles
NEEDLE_HOME_POS = np.array([0.40, 0.42, 1.36])  # gun tip, parked
NEEDLE_READY_OFFSET = np.array([0.0, 0.10, 0.17])  # ready: above the breast
PROBE_LIFT = 0.06               # m above the skin before contact

# ---------------------------------------------------------------- control loop
CONTROL_HZ = 30.0
VIDEO_REALTIME = True
SCAN_MIN_HITS = 20              # = LSTM sequence length
PLAN_EVERY_FRAMES = 1           # replan on every new video frame

# ---------------------------------------------------------------- console
CONSOLE_SIZE = (1920, 1080)
CONSOLE_EVERY = 2              # redraw the console every N control ticks
MAX_DISPLAY_WIDTH = 1600
MAX_DISPLAY_HEIGHT = 900
WINDOW = "UltraBot 3D Robotic Biopsy Console"

# ---------------------------------------------------------------- OR panel rendering
OR_RENDERER = "pyrender"         # "pyrender" | "pybullet"
OR_QUALITY = "medium"
OR_QUALITY_PRESETS = {
    "low":    dict(supersample=1, shadows=False, ambient=[0.6] * 3,
                   lamp=13.0, room=2.1, bounce=0.16,
                   normal_maps=False, post=False),
    "medium": dict(supersample=2, shadows=True, ambient=[0.6] * 3,
                   lamp=13.0, room=2.1, bounce=0.16,
                   normal_maps=True, post=True),
    "high":   dict(supersample=3, shadows=True, ambient=[0.6] * 3,
                   lamp=13.0, room=2.1, bounce=0.16,
                   normal_maps=True, post=True),
}
OR_SIZE = (744, 440)
OR_FOV = 50.0
OR_RENDER_EVERY = 3             # control ticks between panel submissions
OR_FAST_GL = True
OR_PANEL_MODES = ("process", "inline")
OR_REALISTIC = True
OR_STERILE_SLEEVES = True
OR_CAPTIONS = True
ROOM_SIZE = (3.4, 3.2, 1.6)
MONITOR_POS = (1.40, -0.55, 1.36)
MONITOR_YAW = np.pi
US_CART_POS = (-0.72, -0.80)
RADIOLOGIST_POS = (-0.22, -0.80)
SURGEON_HEIGHT = 1.74
OR_VIEWS = [
    # name,               distance, yaw, pitch, target
    ("overview",            1.75, -150, -30, [0.30, 0.04, 1.03]),
    ("radiologist view",    1.15, -110, -38, [0.36, 0.08, 1.05]),
    ("needle close-up",     0.42,  -12, -18, [0.36, 0.10, 1.08]),
    ("probe close-up",      0.40,  -70, -22, [0.355, 0.10, 1.10]),
    ("cross-section",       0.30,    0, -2, [0.355, 0.105, 1.07]),
    ("top down",            1.10,  -90, -89, [0.36, 0.05, 1.00]),
    ("side",                1.70,    0, -14, [0.36, 0.00, 1.05]),
    ("probe arm",           1.40, -160, -24, [0.34, -0.25, 1.15]),
    ("needle arm",          1.40,   25, -24, [0.36, 0.35, 1.15]),
    ("whole room",          4.4,  -38, -28, [0.00, 0.00, 1.00]),
    ("monitors",            1.10,  -90,  -8, [1.40, -0.55, 1.36]),
]
OR_WINDOW_SIZE = (1280, 720)
OR_WINDOW_FPS = 60.0

# ---------------------------------------------------------------- misc
PERF_LOG = True
VOICE_ENABLED = False
VOSK_MODEL_DIR = str(ROOT / "models" / "vosk-small-en")
VOICE_SAMPLE_RATE = 16000
VOICE_WAKE_WORD = "ana"
VOICE_WAKE_ALIASES = ["anna"]
SEED = 0


def us_scale():
    """(mm per px lateral, mm per px axial) for the current preset."""
    return US_WIDTH_MM / PANE_W_PX, US_DEPTH_MM / PANE_H_PX


def ensure_runs_dir():
    os.makedirs(RUNS_DIR, exist_ok=True)
    return RUNS_DIR
