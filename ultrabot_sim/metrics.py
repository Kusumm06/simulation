"""
Per-run logs in ultrabot_sim/runs/:

  metrics_<tag>.csv   one row per control tick (state, lesion, prediction,
                      tip in world / probe mm / pixels, aim, clearances,
                      out-of-plane, IK error, RCM deviation, joints, safety)
  events_<tag>.csv    state transitions, warnings, commands
  summary_<tag>.json  procedures (plans, fire result, criteria), LSTM error
                      stats of this run, detection coverage, perf
Everything written is computed during the run; nothing is invented.
"""
from __future__ import annotations

import csv
import json
import time

import numpy as np

from ultrabot_sim import config as cfg
from ultrabot_sim.planner import _jsonable

TICK_COLS = [
    "t", "state", "kind", "frame", "detected", "track_id", "lesion_u",
    "lesion_v", "lesion_x_mm", "lesion_z_mm", "pred_u", "pred_v",
    "lstm_ready", "tip_x_w", "tip_y_w", "tip_z_w", "tip_x_mm", "tip_y_mm",
    "tip_z_mm", "tip_u", "tip_v", "depth_mm", "target_depth_mm",
    "tip_to_aim_mm", "pred_miss_mm", "aim_offset_mm", "clear_vessel_mm",
    "clear_chestwall_mm", "clear_rib_mm", "clear_pleura_mm",
    "out_of_plane_mm", "ik_err_mm", "axis_err_deg", "rcm_dev_mm",
    "arm_arm_m", "safety", "q0", "q1", "q2", "q3", "q4", "q5", "q6"]


class MetricsLogger:
    def __init__(self, tag, calib):
        cfg.ensure_runs_dir()
        self.tag = tag
        self.calib = calib
        self.path_ticks = cfg.RUNS_DIR / f"metrics_{tag}.csv"
        self.path_events = cfg.RUNS_DIR / f"events_{tag}.csv"
        self.path_summary = cfg.RUNS_DIR / f"summary_{tag}.json"
        self._ft = open(self.path_ticks, "w", newline="")
        self._wt = csv.writer(self._ft)
        self._wt.writerow(TICK_COLS)
        self._fe = open(self.path_events, "w", newline="")
        self._we = csv.writer(self._fe)
        self._we.writerow(["t", "kind", "message"])
        self.frames = 0
        self.detected_frames = 0
        self.t0 = time.time()
        self.marks = 0

    def event(self, t, kind, msg):
        self._we.writerow([round(t, 3), kind, msg])
        self._fe.flush()

    def count_frame(self, obs):
        self.frames += 1
        self.detected_frames += int(bool(obs.detected))

    def tick(self, proc, obs):
        tel = proc.telemetry()
        st = proc.lt.state
        tp = tel["tip_probe_mm"]
        u, v = self.calib.mm_to_px(tp[0], tp[2])
        cb = tel["clearance_by_family"] or {}
        q = tel["q_needle"]
        row = [
            tel["t"], tel["state"], tel["kind"],
            obs.index if obs else "", int(obs.detected) if obs else "",
            obs.track_id if obs else "",
            *(_r(st.center_px) if st else ("", "")),
            *(_r(st.center_mm) if st else ("", "")),
            *(_r(obs.pred_center_px) if obs and obs.pred_center_px
              else ("", "")),
            int(obs.lstm_ready) if obs else "",
            *_r(tel["tip_world"], 6), *_r(tp, 3), round(float(u), 1),
            round(float(v), 1),
            round(tel["s_mm"], 3) if proc.locked else "",
            round(tel["s_target_mm"], 3) if proc.locked else "",
            round(tel["s_target_mm"] - tel["s_mm"], 3) if proc.locked else "",
            _o(tel["predicted_miss_mm"]), _o(proc.aim_offset_mm),
            _o(cb.get("vessel")), _o(cb.get("chest wall")), _o(cb.get("rib")),
            _o(cb.get("pleura")), round(tel["out_of_plane_mm"], 4),
            round(tel["ik_err_mm"], 5), round(tel["axis_err_deg"], 5),
            round(tel["rcm_dev_mm"], 5),
            _o((tel["arm"] or {}).get("arm_arm"), 4), tel["safety"],
            *[round(x, 5) for x in q]]
        self._wt.writerow(row)

    def mark(self, t):
        self.marks += 1
        self.event(t, "mark", f"mark {self.marks}")

    def close(self, procs_results, extra):
        self._ft.close()
        self._fe.close()
        res = [r.to_dict() for r in procs_results]
        summary = {
            "tag": self.tag, "disclaimer": cfg.DISCLAIMER,
            "wall_time_s": round(time.time() - self.t0, 2),
            "procedures": res,
            "frames_processed": self.frames,
            "detection_coverage": (self.detected_frames / self.frames
                                   if self.frames else None),
            "assumptions": {
                "us_depth_mm": cfg.US_DEPTH_MM, "us_width_mm": cfg.US_WIDTH_MM,
                "slice_mm": cfg.US_SLICE_MM,
                "anatomy": "parametric model registered to the image plane "
                           "(not detected)",
                "elevational_lesion_size": "assumed equal to lateral",
                "tissue": "rigid; no deformation, needle deflection or force",
            },
        }
        summary.update(extra)
        self.path_summary.write_text(json.dumps(_jsonable(summary), indent=2))
        return summary


def _r(v, nd=2):
    return tuple(round(float(x), nd) for x in v)


def _o(v, nd=3):
    if v is None:
        return ""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return ""
    return round(f, nd) if np.isfinite(f) else ""


def success_table(results):
    """Pass / fail lines for the console and the terminal."""
    lines = []
    for r in results:
        lines.append(f"=== {r.kind.upper()} path: {r.outcome}"
                     + (f" ({r.abort_reason})" if r.abort_reason else ""))
        for name, ok in r.criteria():
            lines.append(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        f = r.fire
        if f:
            lines.append(f"  notch in lesion {100 * f.get('notch_in_lesion_frac', 0):.0f} %"
                         f", throw axis {f.get('throw_miss_mm', float('nan')):.2f} mm "
                         f"from the lesion centre, fired tip "
                         f"{f['fired_tip_vs_aim_mm']:.4f} mm from command")
        lines.append(f"  max IK error {r.max_ik_err_mm:.4f} mm, axis "
                     f"{r.max_axis_err_deg:.4f} deg, skin RCM "
                     f"{r.max_rcm_dev_mm:.4f} mm, out-of-plane "
                     f"{r.max_out_of_plane_mm:.4f} mm")
        if np.isfinite(r.min_needle_clearance_mm):
            lines.append(f"  min needle clearance {r.min_needle_clearance_mm:.2f}"
                         f" mm ({r.min_clearance_struct})")
        p = r.plan or {}
        if p:
            lines.append(f"  path: insertion {p.get('insertion_mm', 0):.1f} mm,"
                         f" {p.get('angle_chestwall_deg', 0):.1f} deg to chest "
                         f"wall, planned min clearance "
                         f"{p.get('min_clearance_mm', 0):.1f} mm "
                         f"({p.get('min_clearance_struct')})")
        lines.append("  RESULT: " + ("SUCCESS" if r.success else "NOT SUCCESSFUL"))
    return lines
