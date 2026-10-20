# Technical notes

Design, units, algorithms, safety rules, data formats, tests, limitations.
For setup, running and keys see [../README.md](../README.md).

> Kinematic research simulation, not a medical device. Not for clinical use.

## 1. Data flow

```
video / image / frames / bundle / demo           inputs.py
      │  every frame resized to the 990 x 720 pane
      ▼
YOLO (models/best.pt) → CenterIoUTracker → LSTM   perception.py (ultrabot/)
      │  FrameObs: box, centre, class, conf, track, lstm_ready, prediction
      ▼
lesion in mm, 5-frame median, LSTM rollout        lesion_model.py, us_geometry.py
      ▼
parametric anatomy registered to the plane        anatomy.py
      ▼
shortest / safest / recommended (+ legacy 2D)     planner.py
      ▼
state machine, insertion, safety                  control.py, safety.py
      ▼
two KUKA iiwa arms: DLS IK, RRT-Connect, RCM       scene.py, approach.py, rcm.py
      ▼
console + 3D room + logs                          display.py, us_render.py,
                                                  or_render.py, metrics.py
```

## 2. Units and frames

* World: metres, z up, patient supine, head toward +x, patient's right = +y.
* Probe frame (mm): origin at the transducer-face centre; +x lateral (image
  right, toward the head), +z depth (image down), +y = z × x (elevation).
* Pixel (u, v) of the pane → probe (x, 0, z):
  `x = (u − 495) · W/990`, `z = v · D/720`, with the **configured** scanner
  preset `US_WIDTH_MM` (50) and `US_DEPTH_MM` (40), overridable with
  `--width-mm`, `--depth-mm`. The true scale depends on the scanner
  settings used when the video was recorded; it is not measured from the
  video.
* The probe frame in the world comes from arm 1's forward kinematics
  (link-7 frame + flange offset + probe length), so the image plane is
  exactly where the robot holds the probe.
* Elevation: a point is visible if |y| ≤ slice/2 (`US_SLICE_MM` = 1). The
  needle's out-of-plane distance is measured every tick.

## 3. Perception (unchanged UltraBot code)

`perception.Perception.step()` is the per-frame body of
`ultrabot.pipeline.run_pipeline()`: same YOLO call, same
`_yolo_detections()`, same `CenterIoUTracker` settings, same LSTM gating
(20 consecutive frames on the primary track), same velocity convention,
same integer rounding of the prediction. `tests/test_perception.py`
verifies identical outputs on the sample video.

`lesion_model.LesionTracker`:

* lesion = the ellipse inscribed in the YOLO box, in mm; the elevational
  half-size is **assumed** equal to the lateral one (single-plane imaging);
* the centre used for planning is the **median of the last 5 real boxes** on
  the track (YOLO box jitter on large lesions is several mm per frame; the
  median never creates a position that was not measured);
* one-step LSTM error of this run, in px and mm (same definition as
  `ultrabot/metrics.py`);
* **LSTM rollout**: the predicted centre is fed back as the newest element
  of the 20-long window, `k = ceil(t_fire / frame_dt)` times (cap
  `LSTM_MAX_ROLLOUT` = 10); uncertainty after k steps = RMS one-step error ×
  √k (`SAFETY_MARGIN_MM` while fewer than 5 errors exist). No rollout when
  the LSTM is not ready - the plan then aims at the current centre and says
  so.

## 4. Anatomy model (parametric, NOT detected)

| Structure | Model | Role |
|---|---|---|
| skin | flat under the transducer (compressed), outside it `H(1 − cos(min(d/D,1)·π/2))`, H = 32 mm, D = 110 mm | entry surface |
| fat | skin + 2 .. skin + 8 mm | low risk |
| gland | fat .. pectoral fascia | lesions |
| pectoral fascia | line z = z0 + k·x, z0 = max(30 mm, lesion bottom + 6 mm); optional image estimate (`--auto-chestwall`: row-gradient band + RANSAC, used only above 0.55 confidence, never above the lesion) | hard, 5 mm |
| pectoralis | 8 mm below the fascia | |
| ribs | circles r 5 mm, pitch 26 mm, under the muscle | hard, 5 mm |
| pleura | line below the ribs | hard, 5 mm |
| vessels | tubes through the plane at (31, 5) r 1.0 and (−29, 15) r 1.2 mm; moved sideways (and logged) if they would overlap the lesion | hard, 3 mm |
| Cooper's ligaments | 5 oblique bands fascia → skin | soft (counted) |
| nipple-areola | x = −48 ± 16 mm | no entry |

Every structure has an exact signed distance (tubes and half-planes) and a
trimesh mesh (3D view, PyBullet collision of the breast). The vessel layout
was chosen so that, over the real lesions of the sample video, a median of
39 of 222 candidate paths are feasible and the safest path has a median
8.5 mm more clearance than the shortest one.

## 5. Planner

* **Entries**: skin points beside the probe, 5-60 mm outside its edge,
  every 0.5 mm, both sides (222 candidates). Never the image bottom.
* **Line**: entry → aim (LSTM rollout centre). Pre-fire tip = aim − 11 mm,
  fired tip = aim + 11 mm along the needle (22 mm throw, centred).
* **Hard constraints** (first failing reason is kept for display):
  nipple region; puncture ≥ 8° to the skin; angle to the chest wall ≤ 35°;
  insertion ≥ 5 mm; insertion + throw ≤ needle (100 mm) − 10 mm; gun body
  clear of the probe (4 mm) and the skin (2 mm); exact minimum clearance of
  the whole segment entry → fired tip to vessels (3 mm), chest wall (5 mm),
  ribs (5 mm), pleura (5 mm); robot reach (below).
* **Shortest**: minimum insertion (ties: smaller chest-wall angle).
  **Safest**: maximum minimum clearance (ties: angle, ligaments, motion
  sensitivity). **Recommended**: weighted min-max normalised cost
  `w_len·len + w_clear·(1 − clear) + w_angle·angle + w_lig·lig +
  w_motion·motion + w_target·err` (presets balanced / clearance-first /
  length-first, keys `[` `]`).
* **Motion sensitivity** = |predicted lesion displacement across the
  needle| + rollout uncertainty. **Predicted miss** = how far the throw axis
  passes outside the lesion core (half of the lesion's half-size across the
  needle); 0 = the notch goes through the core.
* **Refinement**: `scipy.optimize.minimize` (SLSQP, bounded) on entry
  offset and a ±4° angle change, with all hard constraints as inequality
  constraints and a 0.25 mm buffer; kept only if still feasible and better.
* **Robot reach** (`control.check_plan_ik`): DLS IK at pre-entry, skin and
  pre-fire (< 0.1 mm, < 0.05°), joint margin ≥ 5°, link-patient and
  arm-arm clearance ≥ 6 cm, tool clearances. A failing path is dropped and
  the next one of the same kind is tried.
* **Legacy**: the original `optimize_biopsy_paths()` runs on the same frame
  and is drawn in grey (key `L`). `--planner legacy` drives the robot with
  it; it is evaluated with the same hard constraints and refused when
  infeasible (its entries lie on the image border, under the transducer).
* Typical planning time 100-140 ms on a laptop CPU (median over the sample
  lesions, without the IK checks).

## 6. Robot

* Two KUKA LBR iiwa 7 (`pybullet_data`), kinematic (joints set exactly,
  no contact forces), like the reference project. Probe arm on the
  patient's left, needle arm on the right.
* **IK**: damped least squares, ported from the reference `dls_ik`, with an
  exact axis-angle orientation error, the Jacobian shifted to the tool
  point, lower damping near convergence and a null-space pull to home.
  6-DoF for the probe (image-plane orientation matters), 5-DoF for the
  needle (roll about the needle is free). Typical accuracy: 0.001-0.002 mm.
  `ik_robust` = continuity seed, then home / ready, then deterministic
  random seeds, preferring solutions with joint-limit margin.
* **Gun mount**: the gun sits on a bracket with its axis 30° from the
  flange axis. With a coaxial mount the near-horizontal needle poses put
  joint 6 at its ±120° limit; with 30° the margin is 24-31°.
* **Ready poses**: per side of the probe, chosen among many IK solutions
  for the shortest collision-free transit; solved once and cached in
  `runs/cache/arm_poses_<hash>.json` (re-verified by FK on load).
* **Approach**: RRT-Connect in joint space (collision checks on a ghost
  PyBullet world: arm links, gun and needle vs patient, table, breast,
  probe and the other arm), random shortcutting, then a C2 joint spline
  timed by tool arc length with a minimum-jerk law; peak tool-point speed
  ≤ 250 mm/s, joint speed ≤ 60°/s, verified densely. The last 60 mm to the
  pre-entry pose is a straight servo at ≤ 50 mm/s that tracks the live plan.
* **Skin RCM**: from skin contact the line is locked (`rcm.LockedLine`);
  the tip only moves along it (accelerate-limited depth servo, 3 mm/s
  default, 1-10). The RCM deviation is measured from the arm's forward
  kinematics, not from the command.
* **Biopsy gun**: stylet then cannula advance 22 mm in 0.1 s (slow motion
  ×10 with `t`), hold 2 s, retract 5 mm/s along the same axis.

## 7. State machine and safety

`IDLE → SCAN → PLAN → APPROACH → ALIGN → SKIN_CONTACT → INSERT →
PRE_FIRE_CHECK → FIRE → HOLD → RETRACT → HOME → DONE` (+ `PAUSED`,
`ABORT_RETRACT`, `E_STOP`). Every transition is logged with time and
reason (`events_<tag>.csv`).

* IDLE: probe lowered onto the skin; the video starts at contact. In
  compare mode the needle arm is pre-positioned on the next path's side.
* Plans update on every frame until the line is locked; during the
  approach the side of the probe is locked.
* After puncture only the depth may change: the throw is re-centred on the
  predicted lesion position (limited by needle length and by every margin
  along the extended path). Safety can never increase the depth.

| Check | WARN | Action |
|---|---|---|
| needle clearance to hard structures | within 1 mm of the margin | inside the margin → abort (retract) |
| out of plane | > slice/4 | > slice/2 → abort |
| lesion not detected / track switched | 1+ frames → hold | > 8 frames → abort |
| predicted miss | > 1.5 mm | > 2.5 mm → hold; persisting 3 frames → abort |
| throw end vs chest wall | | < 5 mm → firing blocked → retract |
| arms | < 6 cm apart | link < 2 cm from patient / gun on probe → hold |
| E-stop (`z`, voice "stop") | | both arms freeze; resume retracts if inside |

The 3-frame persistence for the miss rule and the 5-frame median exist
because YOLO's box on a large lesion jumps several mm between frames of
this handheld footage; a single-frame glitch must not trigger a retraction.

## 8. Plan bundle (dashboard → simulation)

`app.py` → "Open 3D Robotic Simulation" writes `outputs/sim_bundle.json`
and starts `python -m ultrabot_sim.main --bundle outputs/sim_bundle.json
--auto`. Schema (validated on load, clear errors):

| key | content |
|---|---|
| `source_video` | analysed video path (frames are read from it) |
| `planning_frame` | frame the dashboard planned on |
| `records` | all `FrameRecord`s of the run (each needs index, detected, raw_center, box_xyxy, lstm_ready, track_id) |
| `lstm_meta` | sequence_length, hidden_size, epochs |
| `class_names` | {"0": "benign", "1": "malignant"} |
| `pane_size` | [990, 720] (must match) |
| `conf_threshold`, `src_fps`, `is_synthetic` | as in the run |
| `optimizer` | the dashboard optimizer's rows without `BiopsyPlan` objects |

The bundle replays exactly what the dashboard computed (no new YOLO run);
the LSTM checkpoint is only used for multi-step rollouts.

## 9. Logs

`runs/metrics_<tag>.csv` (every tick), `events_<tag>.csv`,
`summary_<tag>.json` (procedures with plans, fire result, pass/fail
criteria; LSTM error stats; detection coverage; assumptions),
`perf_<tag>.csv`. `python -m ultrabot_sim.analyze` makes the comparison
table and figures in `runs/figures/`.

Success criteria per procedure (never SUCCESS if one fails): the notch
passes through the lesion; fired tip < 2 mm from the commanded point; min
clearance ≥ margin throughout; IK error < 0.1 mm; skin RCM < 0.5 mm;
out-of-plane < 0.5 mm; no E-stop / abort.

## 10. Tests

* `pytest ultrabot_sim/tests -q`: 35 tests (geometry round trips, anatomy
  SDF signs, perception == run_pipeline, LSTM gating, single image,
  planner constraints, shortest ≤ safest length, safest ≥ shortest
  clearance, vessel on the line refused, planning time, determinism,
  IK on 200 reachable targets, unreachable refused, RCM during insertion,
  collision-free approach within speed limits, abort on lesion lost /
  predicted miss / clearance, E-stop freeze, safety never deeper, a full
  headless run).
* `python -m ultrabot_sim.selftest`: 51 PASS / WARN / FAIL checks in groups
  A-F, ending with a headless compare run on the sample video.

## 11. Limitations and future work

* **Tissue is rigid.** No deformation when the probe presses or the needle
  advances, no needle deflection or bending, no force feedback. Future:
  FEM tissue (e.g. SOFA), needle-tissue interaction and deflection models,
  force sensing at the gun.
* **Anatomy is a parametric model** registered to the image plane, not
  segmented from the ultrasound. Future: segment fascia, vessels (Doppler)
  and ribs from the image.
* **Calibration.** The mm scale is the configured scanner preset; there is
  no probe-to-image calibration of the recorded video. Future: phantom
  calibration (N-wire), reading the depth from the scanner.
* **Single plane.** The lesion's elevational size is assumed. Future: 3D
  ultrasound / multi-plane sweeps.
* **Lesion motion** in the recorded video comes mostly from the
  sonographer's hand; with a robot-held probe the real relative motion
  would be smaller (breathing). The simulation treats image motion as
  tissue motion, which makes it conservative.
* **Real robot integration** (ROS 2 / KUKA FRI, hand-eye calibration,
  certified safety controller) is out of scope.
