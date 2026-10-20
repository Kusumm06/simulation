"""
Safety layer, evaluated on every control tick.

Levels CLEAR / WARN / DANGER (colours like the reference project). The
safety layer can only SLOW, STOP (hold) or RETRACT the needle, or forbid
firing. It never moves the needle deeper - that rule is enforced again in
control.py, which clamps the depth target whenever safety is not CLEAR.

Checks
  * clearance of the needle tip and in-tissue shaft to every hard
    structure (vessels, chest wall, ribs, pleura) and time-to-contact;
  * out-of-plane deviation of the needle vs the slice thickness
    (the needle must stay visible in the image);
  * predicted miss of the throw vs the lesion (LSTM rollout): WARN at
    MOTION_WARN_MM, abort at MISS_ABORT_MM;
  * lesion lost (tracker coasting / lost / switched to another track)
    while inserting: hold, then abort after LOST_ABORT_FRAMES frames;
  * fired-tip clearance to the chest wall: never fire below
    CHESTWALL_CLEARANCE_MM;
  * arm-arm and arm-patient clearance from PyBullet distance queries.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ultrabot_sim import config as cfg

CLEAR, WARN, DANGER = "CLEAR", "WARN", "DANGER"
NEAR_STATES = ("ALIGN", "SKIN_CONTACT", "INSERT", "PRE_FIRE_CHECK", "FIRE",
               "HOLD", "RETRACT", "ABORT_RETRACT")
_RANK = {CLEAR: 0, WARN: 1, DANGER: 2}
LEVEL_BGR = {CLEAR: (90, 200, 90), WARN: (40, 190, 250), DANGER: (60, 60, 235)}


@dataclass
class SafetyReport:
    level: str = CLEAR
    action: str | None = None          # None | "hold" | "abort" | "no_fire"
    messages: list = field(default_factory=list)
    needle_clearance_mm: float | None = None
    clearance_struct: str | None = None
    clearance_by_family: dict = field(default_factory=dict)
    time_to_contact_s: float | None = None
    out_of_plane_mm: float = 0.0
    predicted_miss_mm: float | None = None
    fired_tip_chestwall_mm: float | None = None
    lost_frames: int = 0
    arm: dict = field(default_factory=dict)

    def raise_to(self, level, msg, action=None):
        if _RANK[level] > _RANK[self.level]:
            self.level = level
        self.messages.append(msg)
        order = {None: 0, "no_fire": 1, "hold": 2, "abort": 3}
        if action and order[action] > order.get(self.action, 0):
            self.action = action


class SafetyMonitor:
    def __init__(self):
        self.e_stop = False

    def evaluate(self, state, anatomy, pose, tip_w, axis_w, entry_w,
                 inserted_mm, ext_mm, speed_mm_s, lesion_tracker, locked_track,
                 predicted_miss_mm, fired_tip_mm=None, arm_clear=None,
                 miss_frames=0):
        """
        tip_w / axis_w: actual needle tip (world) and direction from the
        arm's FK; entry_w: locked skin entry (world) or None; inserted_mm:
        tip depth along the line past the skin; ext_mm: throw extension.
        """
        r = SafetyReport()
        if self.e_stop:
            r.raise_to(DANGER, "E-STOP engaged: both arms frozen", "hold")
        inside = entry_w is not None and inserted_mm + ext_mm > 0.0
        # ---- needle clearance (in-tissue part only)
        if inside:
            L = inserted_mm + ext_mm
            s = np.linspace(0.0, L, max(2, int(L / 0.5) + 1))
            pts = entry_w[None, :] + axis_w[None, :] * (s[:, None] / 1000.0)
            P = pose.world_to_probe(pts)
            c = anatomy.clearance(P)
            fam_min = {k: float(np.min(v)) for k, v in c.items()}
            slack = {k: fam_min[k] - anatomy.margins[k] for k in fam_min}
            worst = min(slack, key=slack.get)
            r.needle_clearance_mm = fam_min[worst]
            r.clearance_struct = worst
            r.clearance_by_family = fam_min
            if speed_mm_s > 1e-3:
                r.time_to_contact_s = max(slack[worst], 0.0) / speed_mm_s
            if slack[worst] < -cfg.SAFETY_TOL_MM:
                r.raise_to(DANGER, f"needle inside the {worst} margin "
                           f"({fam_min[worst]:.1f} mm)", "abort")
            elif slack[worst] < 1.0:
                r.raise_to(WARN, f"needle within 1 mm of the {worst} margin")
        # ---- out of plane (needle visibility): only once the needle is
        # lined up with the image plane
        if tip_w is not None and state in NEAR_STATES:
            y = float(pose.world_to_probe(tip_w)[1])
            r.out_of_plane_mm = abs(y)
            if abs(y) > cfg.US_SLICE_MM / 2:
                r.raise_to(DANGER, f"needle tip {abs(y):.2f} mm out of the "
                           "image plane (not visible)",
                           "abort" if inside else "hold")
            elif abs(y) > cfg.US_SLICE_MM / 4:
                r.raise_to(WARN, f"needle {abs(y):.2f} mm out of plane")
        # ---- lesion tracking while the needle is in the tissue
        if state in ("SKIN_CONTACT", "INSERT", "PRE_FIRE_CHECK"):
            lt = lesion_tracker
            lost = lt.frames_without_lesion
            if lt.state is not None and locked_track is not None and \
                    lt.state.track_id != locked_track:
                lost = max(lost, 1)
                r.raise_to(WARN, "tracker switched to another lesion track")
            r.lost_frames = lost
            if lost > cfg.LOST_ABORT_FRAMES:
                r.raise_to(DANGER, f"lesion lost for {lost} frames", "abort")
            elif lost > 0:
                r.raise_to(WARN, f"lesion not detected ({lost} frames): "
                           "holding", "hold")
            # predicted miss of the throw
            if predicted_miss_mm is not None:
                r.predicted_miss_mm = predicted_miss_mm
                if predicted_miss_mm > cfg.MISS_ABORT_MM and \
                        miss_frames >= cfg.MISS_ABORT_FRAMES:
                    r.raise_to(DANGER, f"predicted throw miss "
                               f"{predicted_miss_mm:.1f} mm > "
                               f"{cfg.MISS_ABORT_MM} mm for {miss_frames} "
                               "frames", "abort")
                elif predicted_miss_mm > cfg.MISS_ABORT_MM:
                    r.raise_to(DANGER, f"predicted throw miss "
                               f"{predicted_miss_mm:.1f} mm: holding "
                               f"({miss_frames}/{cfg.MISS_ABORT_FRAMES})",
                               "hold")
                elif predicted_miss_mm > cfg.MOTION_WARN_MM:
                    r.raise_to(WARN, f"predicted throw miss "
                               f"{predicted_miss_mm:.1f} mm")
            # chest wall vs the throw end point
            if fired_tip_mm is not None:
                cw = float(anatomy.clearance(np.array([fired_tip_mm]))
                           ["chest wall"][0])
                r.fired_tip_chestwall_mm = cw
                if cw < cfg.CHESTWALL_CLEARANCE_MM - cfg.SAFETY_TOL_MM:
                    r.raise_to(DANGER, f"throw would end {cw:.1f} mm from "
                               "the chest wall: firing blocked", "no_fire")
        # ---- arms
        if arm_clear:
            r.arm = arm_clear
            if arm_clear.get("arm_arm", 1.0) < cfg.ARM_MIN_CLEARANCE:
                r.raise_to(WARN, f"arms {arm_clear['arm_arm'] * 100:.1f} cm "
                           "apart")
            if arm_clear.get("needle_links", 1.0) < 0.02:
                r.raise_to(DANGER, "needle-arm link near the patient", "hold")
            if arm_clear.get("gun_probe", 1.0) < cfg.PROBE_CLEARANCE_MM / 2000:
                r.raise_to(DANGER, "gun touching the probe", "hold")
        return r
