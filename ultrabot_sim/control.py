"""
Procedure state machine, insertion controller, lesion-motion compensation
and guards.

IDLE -> SCAN -> PLAN -> APPROACH -> ALIGN -> SKIN_CONTACT -> INSERT ->
PRE_FIRE_CHECK -> FIRE -> HOLD -> RETRACT -> HOME -> DONE
(+ PAUSED, ABORT_RETRACT, E_STOP)

IDLE            probe arm lowers the probe onto the skin over the lesion
SCAN            perception warm-up: the LSTM needs 20 consecutive hits
PLAN            shortest / safest / recommended computed and shown
APPROACH        needle arm: collision-free transit (RRT-Connect) to a
                stand-off pose STANDOFF_MM back along the planned axis
ALIGN           straight line (<= APPROACH_FINAL_SPEED) to the pre-entry
                pose, tip PRE_ENTRY_MM before the skin; plans keep updating
                (same side only) until here
SKIN_CONTACT    line LOCKED (rigid needle cannot steer); tip to the skin
INSERT          tip along the locked line; only the depth may change
                (time-of-arrival LSTM compensation), never sideways
PRE_FIRE_CHECK  predicted miss, chest wall vs throw end, lesion visible
FIRE            biopsy gun throw (stylet then cannula)
HOLD            2 s
RETRACT         back along the same axis, 5 mm/s, out of the skin
HOME            collision-free return; gun re-cocked
Every transition is logged with its time and reason.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ultrabot_sim import config as cfg
from ultrabot_sim.anatomy import Anatomy
from ultrabot_sim.approach import Approach, JointPath
from ultrabot_sim.planner import (NeedlePlan, evaluate, segment_clearance,
                                  _jsonable)
from ultrabot_sim.rcm import LockedLine, axis_angle_deg, rcm_deviation
from ultrabot_sim.safety import CLEAR, SafetyMonitor
from ultrabot_sim.scene import joint_margin_deg

STATES = ["IDLE", "SCAN", "PLAN", "APPROACH", "ALIGN", "SKIN_CONTACT",
          "INSERT", "PRE_FIRE_CHECK", "FIRE", "HOLD", "RETRACT", "HOME",
          "DONE"]
SPECIAL = ["PAUSED", "ABORT_RETRACT", "E_STOP"]
INSIDE_STATES = ("SKIN_CONTACT", "INSERT", "PRE_FIRE_CHECK", "FIRE", "HOLD",
                 "RETRACT", "ABORT_RETRACT")
PLAN_DWELL_S = 1.0
APPROACH_EST_S = 6.0
ALIGN_MAX_ANG_VEL = math.radians(30.0)


@dataclass
class RunResult:
    kind: str
    outcome: str = "RUNNING"           # DONE | ABORTED
    abort_reason: str = ""
    plan: dict = field(default_factory=dict)
    t_start: float = 0.0
    t_end: float = 0.0
    fire: dict = field(default_factory=dict)
    max_ik_err_mm: float = 0.0
    max_axis_err_deg: float = 0.0
    max_rcm_dev_mm: float = 0.0
    max_out_of_plane_mm: float = 0.0
    min_needle_clearance_mm: float = float("inf")
    min_clearance_struct: str = ""
    min_slack_mm: float = float("inf")
    e_stop: bool = False
    holds: int = 0
    transitions: list = field(default_factory=list)
    approach: dict = field(default_factory=dict)

    def criteria(self):
        """Pass / fail lines (never SUCCESS if a hard check failed)."""
        f = self.fire
        rows = [
            ("throw passes through the lesion",
             bool(f.get("passes_through_lesion"))),
            ("fired-tip miss < 2 mm from aim",
             f.get("fired_tip_vs_aim_mm") is not None and
             f["fired_tip_vs_aim_mm"] < 2.0),
            ("min clearance >= margin throughout",
             self.min_slack_mm > -cfg.SAFETY_TOL_MM and bool(f)),
            ("IK error < 0.1 mm", self.max_ik_err_mm < 0.1),
            ("skin RCM deviation < 0.5 mm", self.max_rcm_dev_mm < 0.5),
            ("out-of-plane < 0.5 mm", self.max_out_of_plane_mm < 0.5),
            ("no E-stop / abort", self.outcome == "DONE" and not self.e_stop),
        ]
        return rows

    @property
    def success(self):
        return all(ok for _, ok in self.criteria())

    def to_dict(self):
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["criteria"] = [{"check": n, "pass": ok} for n, ok in
                         self.criteria()]
        d["success"] = self.success
        return _jsonable(d)


class Procedure:
    def __init__(self, scene, calib, lesion_tracker, planner, kind=None,
                 auto=False, image_mode=False, log=None, frame_dt=0.2,
                 preset=None, auto_chestwall=None, seed=0, side_hint=None):
        self.scene = scene
        self.calib = calib
        self.lt = lesion_tracker
        self.planner = planner
        self.kind = kind or "recommended"
        self.auto = auto
        self.image_mode = image_mode
        self.frame_dt = frame_dt
        self.preset = preset or cfg.WEIGHT_PRESET
        self.auto_chestwall = auto_chestwall
        self.seed = seed
        self.side_hint = side_hint
        self._log = log or (lambda t, kind, msg: print(f"[{t:7.2f}] {msg}"))
        self.safety = SafetyMonitor()
        self.needle = scene.needle
        self.speed = cfg.INSERT_SPEED_MM_S
        self.slowmo = False
        self.frozen = False
        self.t = 0.0
        self.state = "IDLE"
        self.t_state = 0.0
        self._paused_from = None
        self.plan_set = None
        self.plan_set_full = None
        self.plan = None
        self.anatomy = None
        self.locked = None
        self.locked_plan = None
        self.locked_track = None
        self.locked_pose = None
        self.side_lock = None
        self.s = -cfg.PRE_ENTRY_MM
        self.v = 0.0
        self.s_target = 0.0
        self.cmd_tip = None
        self.cmd_dir = None
        self.joint_path = None
        self.path_t = 0.0
        self.align = None
        self.result = None
        self.results = []
        self.report = None
        self.ik_err_mm = 0.0
        self.axis_err_deg = 0.0
        self.rcm_dev_mm = 0.0
        self.out_of_plane_mm = 0.0
        self.predicted_miss_mm = None
        self.aim_offset_mm = None
        self.core_radius_mm = None
        self.fired_tip_mm = None
        self.start_requested = False
        self.awaiting_confirm = False
        self._approach_failures = 0
        self._arm_clear = {}
        self._arm_clear_tick = 0
        self._plan_seed = {}
        self.events = []
        self._miss_frames = 0
        self._side_seen = None
        self._side_count = 0
        self._probe_path = None
        self._prep_path = None
        self._prep_t = 0.0
        self.needle_ready = False
        self.ready_probe()

    # ================================================================ log
    def log(self, kind, msg):
        self.events.append((round(self.t, 3), kind, msg))
        self._log(self.t, kind, msg)

    def goto(self, new, reason):
        old = self.state
        self.state = new
        self.t_state = 0.0
        self.log("state", f"{old} -> {new}: {reason}")
        if self.result is not None:
            self.result.transitions.append((round(self.t, 3), old, new,
                                            reason))

    # ============================================================ commands
    def select(self, kind):
        if self.state in ("SCAN", "PLAN", "DONE", "IDLE"):
            self.kind = kind
            self.log("cmd", f"path selected: {kind}")
            if self.plan_set is not None:
                self.plan = self._pick(self.plan_set)
        else:
            self.log("cmd", f"path change refused in {self.state}")

    def request_start(self):
        if self.state in ("SCAN", "PLAN"):
            if self.plan is None or not self.plan.feasible:
                self.log("cmd", "start refused: no feasible plan")
                return False
            self.start_requested = True
            self.log("cmd", f"start requested ({self.kind})")
            return True
        self.log("cmd", f"start refused in {self.state}")
        return False

    def toggle_freeze(self):
        self.frozen = not self.frozen
        self.log("cmd", "target frozen" if self.frozen else "target live")

    def e_stop(self):
        if self.state != "E_STOP":
            self._paused_from = self.state
            self.safety.e_stop = True
            if self.result is not None:
                self.result.e_stop = True
            self.v = 0.0
            self.goto("E_STOP", "E-stop pressed: both arms frozen")

    def resume(self):
        if self.state == "E_STOP":
            self.safety.e_stop = False
            prev = self._paused_from or "SCAN"
            # after an E-stop with the needle in tissue: retract, never go on
            if prev in ("SKIN_CONTACT", "INSERT", "PRE_FIRE_CHECK", "FIRE"):
                self._abort("resumed after E-stop: retracting")
            elif prev in ("APPROACH", "ALIGN"):
                self.goto("HOME", "resumed after E-stop: returning home")
                self._start_home()
            else:
                self.goto(prev, "resumed after E-stop")

    def toggle_pause(self):
        if self.state == "PAUSED":
            self.goto(self._paused_from or "SCAN", "resumed")
        elif self.state not in ("E_STOP",):
            self._paused_from = self.state
            self.v = 0.0
            self.goto("PAUSED", "paused")

    def change_speed(self, delta):
        self.speed = float(np.clip(self.speed + delta, 1.0, 10.0))
        self.log("cmd", f"insertion speed {self.speed:.0f} mm/s")

    def cycle_preset(self, step):
        names = list(cfg.WEIGHT_PRESETS)
        i = (names.index(self.preset) + step) % len(names)
        self.preset = names[i]
        self.log("cmd", f"cost weights: {self.preset}")

    # ============================================================ probe
    def ready_probe(self):
        """Lower the probe from its home onto the skin (collision-checked
        joint path) during IDLE."""
        ap = Approach(self.scene, seed=self.seed)
        jp = ap.plan("probe", self.scene.q["probe"],
                     self.scene.ready_q["probe"])
        if jp is None:
            jp = JointPath([self.scene.q["probe"],
                            self.scene.ready_q["probe"]], [2.0])
        self._probe_path = jp
        self.path_t = 0.0

    # ============================================================ planning
    def _pick(self, ps):
        if cfg.DRIVE_PLANNER == "legacy":
            return self._legacy_plan(ps)
        return ps.get(self.kind)

    def _time_before_insert(self):
        if self.state in ("IDLE", "SCAN", "PLAN"):
            return APPROACH_EST_S
        if self.state == "APPROACH" and self.joint_path is not None:
            return max(self.joint_path.T - self.path_t, 0.0) + 1.5
        if self.state == "ALIGN" and self.align is not None:
            w = self.align["plan"].world
            return float(np.linalg.norm(w["pre_entry"] - self.align["tip"])
                         ) / cfg.APPROACH_FINAL_SPEED + 0.5
        return 0.0

    def replan(self, obs=None):
        st = self.lt.state
        if st is None or self.state in INSIDE_STATES or \
                self.state in ("HOME", "DONE", "E_STOP", "PAUSED"):
            return None
        if self.frozen and self.plan_set is not None:
            return self.plan_set
        frame = obs.image_bgr if obs is not None else None
        self.anatomy = Anatomy(self.calib, st.geom, frame,
                               self.auto_chestwall)
        pose = self.scene.probe_pose()
        ps = self.planner.plan(
            st, self.anatomy, self.lt, pose=pose,
            ik_checker=self.check_plan_ik, preset=self.preset,
            t_before_insert=self._time_before_insert(),
            frame_dt=self.frame_dt, speed_mm_s=self.speed,
            side_lock=self.side_lock)
        self.plan_set = ps
        if self.side_lock is None:
            self.plan_set_full = ps          # both sides (comparison table)
        self.plan = self._pick(ps)
        return ps

    def check_plan_ik(self, plan: NeedlePlan):
        """Robot feasibility of a plan (constraint 4): IK < 0.1 mm and
        < 0.05 deg at pre-entry, skin and pre-fire, joint margin >= 5 deg,
        continuous IK along the insertion, link / tool clearances."""
        sc = self.scene
        w = plan.world
        d = w["dir"]
        seed = self._plan_seed.get(plan.side, sc.ready_side_q[plan.side])
        q0, e0, a0, m0, ok0 = sc.ik_needle_robust(w["pre_entry"], d, seed,
                                                  n_random=8)
        out = {"ok": False, "err_mm": e0 * 1000, "axis_deg": a0,
               "joint_margin_deg": m0, "reason": ""}
        if not ok0:
            out["reason"] = "pre-entry pose unreachable"
            return out
        errs, angs, margins = [e0], [a0], [m0]
        q = q0
        for tip in (w["entry"], w["prefire"]):
            q, e, a = sc.ik_needle(tip, d, q, iters=120)
            errs.append(e)
            angs.append(a)
            margins.append(joint_margin_deg(q))
        out.update(err_mm=max(errs) * 1000, axis_deg=max(angs),
                   joint_margin_deg=min(margins))
        if max(errs) >= cfg.IK_TOL or max(angs) >= cfg.IK_AXIS_TOL_DEG:
            out["reason"] = "insertion not continuously reachable"
            return out
        if min(margins) < cfg.JOINT_MARGIN_DEG:
            out["reason"] = "joint limit margin < 5 deg"
            return out
        cl = sc.clearances(needle_q=q, probe_q=sc.q["probe"])
        out["clearances_m"] = cl
        if cl["needle_links"] < cfg.ARM_MIN_CLEARANCE:
            out["reason"] = "arm link too close to the patient"
            return out
        if cl["arm_arm"] < cfg.ARM_MIN_CLEARANCE:
            out["reason"] = "arms too close"
            return out
        if cl["gun_probe"] < cfg.PROBE_CLEARANCE_MM / 1000.0:
            out["reason"] = "gun too close to the probe"
            return out
        if cl["needle_wrist"] < 0.02 or cl["gun_patient"] < 0.002:
            out["reason"] = "tool too close to the patient"
            return out
        self._plan_seed[plan.side] = q0
        out["ok"] = True
        out["q_pre_entry"] = list(map(float, q0))
        return out

    def _legacy_plan(self, ps):
        """The original 2D planner's recommended path, mapped to the probe
        plane and evaluated with the SAME hard constraints. The robot only
        executes it if it is feasible (usually it is not: its entries lie
        on the image border, under the transducer)."""
        row = (ps.legacy or {}).get("recommended")
        if not row or self.anatomy is None:
            return None
        an = self.anatomy
        E = np.array(self.calib.px_to_mm(*row["entry"]), float)
        A = np.array(self.calib.px_to_mm(*row["target"]), float)
        d = (A - E) / max(np.linalg.norm(A - E), 1e-9)
        st = self.lt.state
        t_now = np.array(st.center_mm)
        ev = evaluate(an, E[None], d[None], A[None], t_now, t_now,
                      np.zeros(2), self.lt.uncertainty_mm(1))
        p = self.planner._make_plan("legacy", ev, 0, int(np.sign(E[0]) or 1),
                                    0.0, t_now, t_now, None, {}, an=an)
        if E[1] < float(an.skin_depth(E[0])) + 0.5 or \
                abs(E[0]) <= self.calib.half_width_mm:
            p.reasons.insert(0, "entry under the transducer (image border)")
            p.feasible = False
        p.attach_world(self.scene.probe_pose(), self.calib)
        return p

    # ============================================================ main tick
    def tick(self, dt, new_frame=False, obs=None):
        self.t += dt
        if new_frame:
            if self.predicted_miss_mm is not None and \
                    self.predicted_miss_mm > cfg.MISS_ABORT_MM:
                self._miss_frames += 1
            else:
                self._miss_frames = 0
            if self.plan is not None:
                if self.plan.side == self._side_seen:
                    self._side_count += 1
                else:
                    self._side_seen, self._side_count = self.plan.side, 1
        self.t_state += dt
        st = self.state
        if st == "E_STOP" or st == "PAUSED":
            self._measure()
            return
        if new_frame and st in ("SCAN", "PLAN", "APPROACH", "ALIGN", "DONE"):
            if st != "DONE" or self.auto:
                self.replan(obs)
        handler = getattr(self, "_st_" + st.lower())
        handler(dt, new_frame)
        self._measure()
        self._safety(dt)

    # ------------------------------------------------------------ states
    def _st_idle(self, dt, new_frame):
        """Set-up before imaging: probe onto the skin, then (if the side
        of the coming path is known, e.g. the second run of a comparison)
        the needle arm to that side's ready pose. The video does not play
        during IDLE (main.py): imaging starts at skin contact."""
        if not self._probe_path.done(self.path_t):
            self.path_t += dt
            self.scene.set_q("probe", self._probe_path.q(self.path_t))
            if self._probe_path.done(self.path_t) and self.side_hint:
                self._plan_needle_prep(self.side_hint)
                self.log("state", "probe on the skin; pre-positioning the "
                         "needle arm before imaging")
            return
        if self.side_hint and not self.needle_ready:
            self._step_needle_prep(dt)
            return
        if not self.side_hint:
            self._plan_needle_prep()
        self.goto("SCAN", "probe on the skin over the lesion; imaging")

    def _plan_needle_prep(self, side=None):
        """Needle arm: parked -> ready pose on the side of the probe the
        live plan uses (pre-positioned while the LSTM warms up)."""
        sc = self.scene
        if side is None:
            side = self.plan.side if self.plan is not None else 1
        self._prep_side = side
        goal = sc.ready_side_q[side]
        if np.allclose(sc.q["needle"], goal, atol=1e-6):
            self._prep_path, self._prep_t = None, 0.0
            self.needle_ready = True
            return
        ap = Approach(sc, seed=self.seed + 3)
        jp = ap.plan("needle", sc.q["needle"], goal)
        if jp is None:
            jp = JointPath([sc.q["needle"], goal], [3.0])
        self._prep_path, self._prep_t = jp, 0.0
        self.needle_ready = False

    def _step_needle_prep(self, dt):
        # follow the live plan's side while waiting (only when settled)
        if self.plan is not None and self.needle_ready and \
                self.plan.side != getattr(self, "_prep_side", None) and \
                self._side_count >= cfg.SIDE_STABLE_FRAMES and \
                self.state == "SCAN":
            self._plan_needle_prep(self.plan.side)
        if self.needle_ready or self._prep_path is None:
            return
        self._prep_t += dt
        self.scene.set_q("needle", self._prep_path.q(self._prep_t))
        if self._prep_path.done(self._prep_t):
            self.needle_ready = True
            self.log("state", f"needle arm ready on the "
                     f"{'+x' if self._prep_side > 0 else '-x'} side")

    def _st_scan(self, dt, new_frame):
        self._step_needle_prep(dt)
        st = self.lt.state
        if st is None:
            return
        ready = st.lstm_ready or self.image_mode
        if self.start_requested and self.plan is not None and \
                self.plan.feasible:
            self.start_requested = False
            self.goto("PLAN", f"operator start ({self.kind})")
            return
        if self.auto and ready and self.plan is not None and \
                self.plan.feasible:
            why = "LSTM ready" if st.lstm_ready else \
                "single image (LSTM not available)"
            self.goto("PLAN", f"{why}; feasible {self.kind} path")

    def _st_plan(self, dt, new_frame):
        self._step_needle_prep(dt)
        if not self.needle_ready:
            return
        if self.plan is None or not self.plan.feasible:
            if self.t_state > 3.0:
                self.goto("SCAN", "no feasible plan")
            return
        if self.t_state >= PLAN_DWELL_S:
            self._begin_run()
            self._start_approach()

    def _begin_run(self):
        self.result = RunResult(kind=self.kind, t_start=self.t)
        self.result.plan = self.plan.to_dict()
        self.result.transitions.append((round(self.t, 3), "SCAN", "PLAN",
                                        "run started"))
        self.side_lock = self.plan.side

    def _start_approach(self):
        sc = self.scene
        plan = self.plan
        w = plan.world
        d = w["dir"]
        so = w["pre_entry"] - d * (cfg.STANDOFF_MM / 1000.0)
        q_so, e, a, m, ok = sc.ik_needle_robust(so, d, sc.q["needle"],
                                                n_random=12)
        if not ok:
            self._approach_fail("stand-off pose unreachable")
            return
        ap = Approach(sc, seed=self.seed + self._approach_failures)
        jp = ap.plan("needle", sc.q["needle"], q_so)
        if jp is None:
            self._approach_fail(ap.stats.get("reason", "no path"))
            return
        self.joint_path = jp
        self.path_t = 0.0
        if self.result is not None:
            self.result.approach = dict(ap.stats)
        self.goto("APPROACH", f"collision-free transit "
                  f"({ap.stats['waypoints']} waypoints, "
                  f"{jp.T:.1f} s, planned in {ap.stats['ms']:.0f} ms)")

    def _approach_fail(self, why):
        self._approach_failures += 1
        self.side_lock = None
        self.goto("SCAN", f"approach not possible: {why}")

    def _st_approach(self, dt, new_frame):
        if self._lesion_lost_outside():
            return
        self.path_t += dt
        q = self.joint_path.q(self.path_t)
        self.scene.set_q("needle", q)
        self.cmd_tip, self.cmd_dir = None, None
        if self.joint_path.done(self.path_t):
            self._start_align()

    def _start_align(self):
        if self.plan is None or not self.plan.feasible:
            self._abort_outside("plan became infeasible during approach")
            return
        tip, d = self.scene.needle_axis()
        self.align = {"tip": tip.copy(), "d": d.copy(), "plan": self.plan,
                      "settled": 0.0}
        w = self.plan.world
        dist = float(np.linalg.norm(w["pre_entry"] - tip))
        self.goto("ALIGN", f"straight approach {dist * 1000:.0f} mm at "
                  f"<= {cfg.APPROACH_FINAL_SPEED * 1000:.0f} mm/s, tracking "
                  "the live plan")

    def _lesion_lost_outside(self):
        if self.image_mode:
            return False
        if self.lt.frames_without_lesion > cfg.LOST_ABORT_FRAMES:
            self._abort_outside(f"lesion not seen for "
                                f"{self.lt.frames_without_lesion} frames")
            return True
        return False

    def _st_align(self, dt, new_frame):
        """Speed-limited servo of the commanded tip pose toward the LIVE
        plan's pre-entry pose (the plan keeps updating with the lesion until
        the line is locked). No restarts: the goal may move every frame."""
        if self._lesion_lost_outside():
            return
        al = self.align
        if self.plan is not None and self.plan.feasible:
            al["plan"] = self.plan
        w = al["plan"].world
        goal, gd = w["pre_entry"], w["dir"]
        e = goal - al["tip"]
        dist = float(np.linalg.norm(e))
        # slow down over the last 10 mm (no overshoot)
        v = min(cfg.APPROACH_FINAL_SPEED, max(0.004, dist * 4.0))
        step = min(dist, v * dt)
        if dist > 1e-12:
            al["tip"] = al["tip"] + e / dist * step
        ang = math.radians(axis_angle_deg(al["d"], gd))
        if ang > 1e-12:
            u = min(1.0, ALIGN_MAX_ANG_VEL * dt / ang)
            al["d"] = _slerp_dir(al["d"], gd, u)
        self._move_tip(al["tip"], al["d"])
        close = dist < 2e-5 and axis_angle_deg(al["d"], gd) < 0.005
        al["settled"] = al["settled"] + dt if close else 0.0
        if al["settled"] >= 0.2:
            if self.plan is None or not self.plan.feasible:
                self._abort_outside("no feasible plan at the skin")
                return
            self._lock_line()

    def _lock_line(self):
        p = self.align["plan"]
        w = p.world
        self.locked = LockedLine(w["entry"], w["dir"])
        self.locked_plan = p
        self.locked_pose = self.scene.probe_pose()
        st = self.lt.state
        self.locked_track = st.track_id if st is not None else None
        self.s = -cfg.PRE_ENTRY_MM
        self.v = 0.0
        self.s_target = 0.0
        if self.result is not None:
            self.result.plan = p.to_dict()
        self.goto("SKIN_CONTACT", f"line LOCKED ({p.kind}: entry "
                  f"{p.entry_mm[0]:+.1f}/{p.entry_mm[1]:.1f} mm, "
                  f"{p.angle_chestwall_deg:.1f} deg to chest wall)")

    def _st_skin_contact(self, dt, new_frame):
        self._servo_depth(dt, 0.0)
        if self.s >= -0.01:
            self.s = 0.0
            self.s_target = self.locked_plan.insertion_mm
            self.goto("INSERT", "skin punctured; inserting along the locked "
                      "line")

    def _st_insert(self, dt, new_frame):
        if new_frame or self.fired_tip_mm is None:
            self._update_depth_target()
        tgt = self.s_target
        if self.report is not None and self.report.action in ("hold",):
            tgt = self.s                    # hold position
        self._servo_depth(dt, tgt)
        if abs(self.s - self.s_target) < 0.02 and abs(self.v) < 0.05 and \
                (self.report is None or self.report.action != "hold"):
            self.goto("PRE_FIRE_CHECK", f"pre-fire depth "
                      f"{self.s:.1f} mm reached")

    def _st_pre_fire_check(self, dt, new_frame):
        if new_frame:
            old = self.s_target
            self._update_depth_target()
            if abs(self.s_target - old) > cfg.PREFIRE_REAIM_MM:
                self.goto("INSERT", "lesion moved along the axis: adjusting "
                          "depth")
                return
            # small axial changes are inside the 22 mm throw: keep depth
            self.s_target = old
        self._servo_depth(dt, self.s)
        r = self.report
        if r is not None and r.action == "no_fire":
            self._abort("firing blocked: throw would reach the chest wall")
            return
        if r is not None and r.action == "hold":
            return
        if self.t_state >= 0.4:
            self._fire()

    def _fire(self):
        self.needle.fire()
        self.goto("FIRE", "biopsy gun fired")

    def _st_fire(self, dt, new_frame):
        done = self.needle.update(dt, cfg.FIRE_SLOWMO if self.slowmo else 1.0)
        self.scene.refresh_tools()
        if done:
            self._record_fire()
            self.goto("HOLD", "throw complete; holding")

    def _st_hold(self, dt, new_frame):
        if self.t_state >= cfg.HOLD_S:
            self.goto("RETRACT", "sample taken; retracting along the axis")

    def _st_retract(self, dt, new_frame):
        goal = -(cfg.PRE_ENTRY_MM + self.needle.stylet_ext)
        self._servo_depth(dt, goal, vmax=cfg.RETRACT_SPEED_MM_S)
        if self.s <= goal + 0.02:
            self.locked = None
            self.goto("HOME", "needle out of the skin")
            self._start_home()

    def _st_abort_retract(self, dt, new_frame):
        goal = -(cfg.PRE_ENTRY_MM + self.needle.stylet_ext)
        self._servo_depth(dt, goal, vmax=cfg.RETRACT_SPEED_MM_S)
        if self.s <= goal + 0.02:
            self.locked = None
            self.goto("HOME", "aborted: needle out of the skin")
            self._start_home()

    def _start_home(self):
        ap = Approach(self.scene, seed=self.seed + 7)
        jp = ap.plan("needle", self.scene.q["needle"],
                     self.scene.home_q["needle"])
        if jp is None:
            jp = JointPath([self.scene.q["needle"],
                            self.scene.home_q["needle"]], [4.0])
            self.log("warn", "home path: RRT failed, direct joint move")
        self.joint_path = jp
        self.path_t = 0.0

    def _st_home(self, dt, new_frame):
        self.path_t += dt
        self.scene.set_q("needle", self.joint_path.q(self.path_t))
        if self.joint_path.done(self.path_t):
            self.needle.reset()
            self.scene.refresh_tools()
            self._finish()

    def _st_done(self, dt, new_frame):
        pass

    def _finish(self):
        r = self.result
        if r is not None:
            if r.outcome == "RUNNING":
                r.outcome = "DONE"
            r.t_end = self.t
            self.results.append(r)
        self.side_lock = None
        self.goto("DONE", "needle arm home" +
                  ("" if r is None else f" ({r.outcome})"))

    def next_run(self, kind=None):
        """Back to SCAN for another procedure (compare mode / retry)."""
        if kind:
            self.kind = kind
        self.result = None
        self.locked = None
        self.locked_plan = None
        self.fired_tip_mm = None
        self.predicted_miss_mm = None
        self.side_lock = None
        self.plan = self._pick(self.plan_set) if self.plan_set else None
        self.goto("SCAN", f"ready for the next run ({self.kind})")

    # ------------------------------------------------------------ aborts
    def _abort(self, why):
        if self.result is not None:
            self.result.outcome = "ABORTED"
            self.result.abort_reason = why
        self.v = 0.0
        if self.locked is None:
            self._abort_outside(why)
            return
        self.goto("ABORT_RETRACT", why)

    def _abort_outside(self, why):
        if self.result is not None:
            self.result.outcome = "ABORTED"
            self.result.abort_reason = why
        self.goto("HOME", f"aborted before skin: {why}")
        self._start_home()

    # ------------------------------------------------------------ motion
    def _move_tip(self, tip, d):
        """IK for the needle arm (continuity seed, robust fallback)."""
        sc = self.scene
        q, e, a = sc.ik_needle(tip, d, sc.q["needle"], iters=60)
        if e >= cfg.IK_TOL or a >= cfg.IK_AXIS_TOL_DEG:
            q2, e2, a2, m2, ok = sc.ik_needle_robust(tip, d, sc.q["needle"],
                                                     n_random=6)
            if e2 < e:
                q, e, a = q2, e2, a2
        # joint speed limit (never exceeded silently)
        dq = np.abs(np.asarray(q) - np.asarray(sc.q["needle"]))
        lim = cfg.MAX_JOINT_VEL / cfg.CONTROL_HZ * 1.5
        if float(dq.max()) > lim and self.state in INSIDE_STATES:
            self.log("warn", f"joint step {math.degrees(dq.max()):.2f} deg "
                     "exceeds the speed limit")
        sc.set_q("needle", q)
        self.cmd_tip, self.cmd_dir = np.asarray(tip), np.asarray(d)

    def _servo_depth(self, dt, target, vmax=None):
        """Accelerate-limited 1-D tracking of the insertion depth (mm)."""
        vmax = self.speed if vmax is None else vmax
        a = cfg.INSERT_ACCEL_MM_S2
        e = target - self.s
        v_des = math.copysign(min(vmax, math.sqrt(2 * a * abs(e))), e)
        if abs(e) < 1e-4:
            v_des = 0.0
        self.v += float(np.clip(v_des - self.v, -a * dt, a * dt))
        step = self.v * dt
        if (e > 0 and step > e) or (e < 0 and step < e):
            step, self.v = e, 0.0
        self.s += step
        self._move_tip(self.locked.tip(self.s / 1000.0), self.locked.d)

    def _update_depth_target(self):
        """Only the DEPTH may change once the line is locked: centre the
        throw on the predicted lesion position at firing time, limited by
        the needle length and the hard-structure margins along the
        extended path. Safety never lets this go deeper."""
        p = self.locked_plan
        E = np.array(p.entry_mm)
        d = np.array(p.direction)
        T = cfg.THROW_MM
        t_left = max(self.s_target - self.s, 0.0) / max(self.speed, 1e-3)
        k = int(math.ceil((t_left + 0.4) / max(self.frame_dt, 1e-3)))
        ro = self.lt.rollout(k)
        st = self.lt.state
        if ro.k > 0:
            c = np.array(ro.points_mm[-1])
        elif st is not None:
            c = np.array(st.center_mm)
        else:
            c = np.array(p.lesion_pred_mm)
        a_ax = float((c - E) @ d)
        s_des = a_ax - T / 2.0
        s_max = cfg.NEEDLE_LEN_MM - cfg.GRIP_MARGIN_MM - T
        lo, hi = cfg.MIN_INSERTION_MM, min(s_des, s_max)
        # deepest depth that keeps every margin along E -> fired tip
        hi = self._deepest_safe(E, d, max(hi, lo))
        s_new = float(np.clip(s_des, lo, max(hi, lo)))
        if self.report is not None and self.report.level != CLEAR:
            s_new = min(s_new, max(self.s, self.s_target))
        self.s_target = s_new
        nperp = np.array([-d[1], d[0]])
        perp = abs(float((c - E) @ nperp))
        cen = s_new + T / 2.0
        over = max(abs(a_ax - cen) - T / 2.0, 0.0)
        self.aim_offset_mm = math.hypot(perp, over)
        self.core_radius_mm = self.lt.core_radius_mm(d)
        self.predicted_miss_mm = max(self.aim_offset_mm -
                                     self.core_radius_mm, 0.0)
        self.fired_tip_mm = E + d * (s_new + T)

    def _deepest_safe(self, E, d, s_hi):
        an = self.anatomy
        T = cfg.THROW_MM

        def ok(s):
            fam = segment_clearance(an, E[None], d[None],
                                    np.array([s + T]))
            return all(float(v[0]) - an.margins[k] >= 0.0
                       for k, v in fam.items())
        if ok(s_hi):
            return s_hi
        lo = 0.0
        hi = s_hi
        for _ in range(30):
            mid = (lo + hi) / 2
            if ok(mid):
                lo = mid
            else:
                hi = mid
        return lo

    # ------------------------------------------------------------ measure
    def _measure(self):
        sc = self.scene
        tip, d = sc.needle_axis()
        if self.cmd_tip is not None and self.state in (
                "ALIGN", "SKIN_CONTACT", "INSERT", "PRE_FIRE_CHECK", "FIRE",
                "HOLD", "RETRACT", "ABORT_RETRACT"):
            self.ik_err_mm = float(np.linalg.norm(tip - self.cmd_tip)) * 1000
            self.axis_err_deg = axis_angle_deg(d, self.cmd_dir)
        else:
            self.ik_err_mm = 0.0
            self.axis_err_deg = 0.0
        pose = self.locked_pose or sc.probe_pose()
        self.out_of_plane_mm = float(abs(pose.world_to_probe(tip)[1]))
        if self.locked is not None and self.s + self.needle.stylet_ext > 0:
            self.rcm_dev_mm = rcm_deviation(self.locked.entry, tip, d) * 1000
        else:
            self.rcm_dev_mm = 0.0
        r = self.result
        if r is not None and self.state not in ("DONE",):
            r.max_ik_err_mm = max(r.max_ik_err_mm, self.ik_err_mm)
            r.max_axis_err_deg = max(r.max_axis_err_deg, self.axis_err_deg)
            r.max_rcm_dev_mm = max(r.max_rcm_dev_mm, self.rcm_dev_mm)
            if self.state in INSIDE_STATES or self.state == "ALIGN":
                r.max_out_of_plane_mm = max(r.max_out_of_plane_mm,
                                            self.out_of_plane_mm)

    def _safety(self, dt):
        sc = self.scene
        self._arm_clear_tick += 1
        if self._arm_clear_tick % 3 == 0 or not self._arm_clear:
            self._arm_clear = sc.clearances()
        tip, d = sc.needle_axis()
        entry = self.locked.entry if self.locked is not None else None
        an = self.anatomy
        if an is None:
            self.report = None
            return
        pose = self.locked_pose or sc.probe_pose()
        r = self.safety.evaluate(
            self.state, an, pose, tip, self.locked.d if self.locked else d,
            entry, self.s, self.needle.stylet_ext, abs(self.v), self.lt,
            self.locked_track,
            self.predicted_miss_mm if self.state in (
                "INSERT", "PRE_FIRE_CHECK", "SKIN_CONTACT") else None,
            self.fired_tip_mm if self.state in ("INSERT", "PRE_FIRE_CHECK")
            else None, self._arm_clear, self._miss_frames)
        prev = self.report
        self.report = r
        res = self.result
        if res is not None and r.needle_clearance_mm is not None:
            slack = r.needle_clearance_mm - an.margins[r.clearance_struct]
            if r.needle_clearance_mm < res.min_needle_clearance_mm:
                res.min_needle_clearance_mm = r.needle_clearance_mm
                res.min_clearance_struct = r.clearance_struct
            res.min_slack_mm = min(res.min_slack_mm, slack)
        if r.level != CLEAR and (prev is None or prev.messages != r.messages):
            self.log("safety", f"{r.level}: " + "; ".join(r.messages))
        if r.action == "hold" and res is not None and \
                (prev is None or prev.action != "hold"):
            res.holds += 1
        if r.action == "abort" and self.state in (
                "SKIN_CONTACT", "INSERT", "PRE_FIRE_CHECK"):
            self._abort(r.messages[-1] if r.messages else "safety abort")
        elif r.action == "abort" and self.state in ("APPROACH", "ALIGN"):
            self._abort_outside(r.messages[-1])

    def _record_fire(self):
        """Throw result vs the lesion AT FIRING TIME (from perception)."""
        sc = self.scene
        f, d, _ = sc.needle_frame()
        stylet, cannula = self.needle.tips(f, d)
        pose = self.locked_pose
        st = self.lt.state
        E = np.array(self.locked_plan.entry_mm)
        dd = np.array(self.locked_plan.direction)
        cmd_fired = E + dd * (self.s_target + cfg.THROW_MM)
        tip_mm = pose.world_to_probe(stylet)
        fired_xz = np.array([tip_mm[0], tip_mm[2]])
        rec = {"t": round(self.t, 3), "depth_mm": self.s,
               "fired_tip_mm": fired_xz.tolist(),
               "fired_tip_world": stylet.tolist(),
               "fired_tip_vs_aim_mm": float(np.linalg.norm(fired_xz -
                                                           cmd_fired)),
               "aim_mm": (E + dd * (self.s_target + cfg.THROW_MM / 2)
                          ).tolist(),
               "predicted_miss_mm": self.predicted_miss_mm}
        if st is not None:
            c = np.array(st.center_mm)
            nperp = np.array([-dd[1], dd[0]])
            rec["lesion_center_at_fire_mm"] = c.tolist()
            rec["lesion_frame"] = st.frame
            rec["throw_miss_mm"] = abs(float((c - E) @ nperp))
            # does the specimen notch pass through the lesion ellipse?
            a_n, b_n = self.needle.notch(f, d)
            pa, pb = pose.world_to_probe(a_n), pose.world_to_probe(b_n)
            ts = np.linspace(0, 1, 60)
            xs = pa[0] + (pb[0] - pa[0]) * ts
            zs = pa[2] + (pb[2] - pa[2]) * ts
            inside = st.geom.contains(xs, zs)
            rec["notch_in_lesion_frac"] = float(inside.mean())
            rec["passes_through_lesion"] = bool(inside.any())
            rec["lesion_class"] = st.class_name
        else:
            rec["passes_through_lesion"] = False
            rec["lesion_center_at_fire_mm"] = None
        if self.result is not None:
            self.result.fire = rec
        self.log("fire", "throw: notch " + (
            "THROUGH the lesion" if rec["passes_through_lesion"] else
            "MISSED the lesion") + f", tip {rec['fired_tip_vs_aim_mm']:.3f} mm "
            "from the commanded point")

    # ------------------------------------------------------------ telemetry
    def telemetry(self):
        tip, d = self.scene.needle_axis()
        pose = self.locked_pose or self.scene.probe_pose()
        tp = pose.world_to_probe(tip)
        r = self.report
        return {
            "t": round(self.t, 4), "state": self.state, "kind": self.kind,
            "s_mm": round(self.s, 4), "v_mm_s": round(self.v, 4),
            "s_target_mm": round(self.s_target, 4),
            "tip_world": tip.round(6).tolist(),
            "tip_probe_mm": tp.round(4).tolist(),
            "ik_err_mm": self.ik_err_mm, "axis_err_deg": self.axis_err_deg,
            "rcm_dev_mm": self.rcm_dev_mm,
            "out_of_plane_mm": self.out_of_plane_mm,
            "predicted_miss_mm": self.predicted_miss_mm,
            "safety": r.level if r else CLEAR,
            "clearance_mm": r.needle_clearance_mm if r else None,
            "clearance_struct": r.clearance_struct if r else None,
            "clearance_by_family": r.clearance_by_family if r else {},
            "arm": self._arm_clear,
            "q_needle": list(map(float, self.scene.q["needle"])),
            "q_probe": list(map(float, self.scene.q["probe"])),
            "stylet_ext": self.needle.stylet_ext,
        }


def _slerp_dir(a, b, u):
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    c = float(np.clip(a @ b, -1, 1))
    th = math.acos(c)
    if th < 1e-9:
        return b
    return (math.sin((1 - u) * th) * a + math.sin(u * th) * b) / math.sin(th)
