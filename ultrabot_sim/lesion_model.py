"""
Lesion state in millimetres + LSTM time-of-arrival rollout.

From each FrameObs (pixels, from YOLO + tracker + LSTM) this keeps:
  * the lesion centre and ellipse in the probe frame (mm), via USCalib;
  * the measured one-step LSTM error of THIS run (prediction made at frame
    t vs the real associated centre at t+1 on the same track) - the same
    definition as ultrabot/metrics.py;
  * an autoregressive LSTM rollout: the predicted centre is fed back as the
    newest element of the 20-long window, k times, to estimate where the
    lesion will be when the needle fires. k is capped at LSTM_MAX_ROLLOUT.
    Uncertainty radius after k steps = RMS one-step error x sqrt(k)
    (SAFETY_MARGIN_MM while fewer than 5 errors have been measured).

Nothing is invented: with no detection there is no lesion state, and with
no LSTM window there is no rollout (the plan then aims at the current
centre and says so).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ultrabot_sim import config as cfg
from ultrabot_sim.anatomy import LesionGeom
from ultrabot_sim.us_geometry import USCalib


@dataclass
class Rollout:
    points_px: list            # [(u, v)] for steps 1..k
    points_mm: list            # [(x, z)]
    radii_mm: list             # uncertainty radius per step
    k: int
    source: str                # "lstm" | "none"


@dataclass
class LesionState:
    frame: int
    track_id: Optional[int]
    center_px: tuple
    center_mm: tuple
    geom: LesionGeom
    class_name: Optional[str]
    conf: Optional[float]
    lstm_ready: bool
    pred_px: Optional[tuple] = None
    pred_mm: Optional[tuple] = None
    window: Optional[list] = None


class LesionTracker:
    def __init__(self, calib: USCalib, model_lstm=None, seq_len=20):
        self.calib = calib
        self.model = model_lstm
        self.seq_len = seq_len
        self.state: Optional[LesionState] = None
        self.last_obs = None
        self.errors_px: list = []
        self.errors_mm: list = []
        self._prev_pred = None          # (frame, track, (u, v))
        self.frames_without_lesion = 0
        self.track_changes = 0

    # ------------------------------------------------------------ update
    def update(self, obs):
        self.last_obs = obs
        # one-step LSTM error (same definition as ultrabot.metrics)
        if self._prev_pred is not None and obs.detected:
            f, tid, (pu, pv) = self._prev_pred
            if obs.index == f + 1 and obs.track_id == tid:
                du = obs.center_px[0] - pu
                dv = obs.center_px[1] - pv
                self.errors_px.append(math.hypot(du, dv))
                mx, mz = self.calib.len_px_to_mm(du, dv)
                self.errors_mm.append(float(math.hypot(mx, mz)))
        self._prev_pred = None
        if obs.lstm_ready and obs.pred_center_px is not None:
            self._prev_pred = (obs.index, obs.track_id, obs.pred_center_px)
        if not obs.detected or obs.box_xyxy is None:
            self.frames_without_lesion += 1
            return self.state
        if self.state is not None and obs.track_id != self.state.track_id:
            self.track_changes += 1
            self._recent = []
        self.frames_without_lesion = 0
        # median of the last N real detections on this track (smooths the
        # YOLO box jitter; never creates a position that was not measured
        # in at least one frame's box)
        self._recent = (getattr(self, "_recent", []) +
                        [tuple(obs.box_xyxy)])[-cfg.LESION_SMOOTH_FRAMES:]
        box = tuple(np.median(np.array(self._recent), axis=0))
        g = LesionGeom.from_box_px(box, self.calib)
        cx, cz = g.x, g.z
        st = LesionState(frame=obs.index, track_id=obs.track_id,
                         center_px=tuple(map(float, self.calib.mm_to_px(cx, cz))),
                         center_mm=(float(cx), float(cz)), geom=g,
                         class_name=obs.class_name, conf=obs.conf,
                         lstm_ready=obs.lstm_ready, window=obs.lstm_window)
        if obs.pred_center_px is not None:
            px, pz = self.calib.px_to_mm(*obs.pred_center_px)
            st.pred_px = tuple(obs.pred_center_px)
            st.pred_mm = (float(px), float(pz))
        self.state = st
        return st

    # ------------------------------------------------------------ errors
    def sigma_mm(self):
        """RMS one-step LSTM error of this run (mm), or None (< 5)."""
        if len(self.errors_mm) < 5:
            return None
        e = np.asarray(self.errors_mm[-200:], float)
        return float(np.sqrt(np.mean(e ** 2)))

    def error_stats(self):
        e = np.asarray(self.errors_mm, float)
        ep = np.asarray(self.errors_px, float)
        if e.size == 0:
            return {"n": 0}
        return {"n": int(e.size), "mean_mm": float(e.mean()),
                "median_mm": float(np.median(e)),
                "p90_mm": float(np.percentile(e, 90)),
                "rms_mm": float(np.sqrt(np.mean(e ** 2))),
                "mean_px": float(ep.mean()), "median_px": float(np.median(ep))}

    # ----------------------------------------------------------- rollout
    def core_radius_mm(self, d):
        """Half the lesion extent across a needle direction d (x, z), times
        CORE_FRACTION: the throw should pass inside this radius."""
        st = self.state
        if st is None:
            return 0.0
        n = np.array([-d[1], d[0]], float)
        n /= np.linalg.norm(n)
        r = 1.0 / math.sqrt((n[0] / st.geom.a) ** 2 + (n[1] / st.geom.b) ** 2)
        return cfg.CORE_FRACTION * r

    def rollout(self, k):
        """Autoregressive rollout of the LSTM from the current window."""
        st = self.state
        k = int(max(0, min(k, cfg.LSTM_MAX_ROLLOUT)))
        if st is None or not st.lstm_ready or st.window is None or \
                self.model is None or k == 0:
            return Rollout([], [], [], 0, "none")
        from ultrabot.pipeline import predict_next
        W, H = self.calib.pane_w, self.calib.pane_h
        win = [list(map(float, c)) for c in st.window]
        # start from the (smoothed) current centre, keep the LSTM's motion
        W0 = np.array(win[-1])
        c0 = np.array([st.center_px[0] / W, st.center_px[1] / H])
        shift = c0 - W0
        pts_px = []
        for _ in range(k):
            vx, vy = predict_next(self.model, win)
            cx = min(max(win[-1][0] + float(vx), 0.0), (W - 1) / W)
            cy = min(max(win[-1][1] + float(vy), 0.0), (H - 1) / H)
            win = win[1:] + [[cx, cy]]
            pts_px.append(((cx + shift[0]) * W, (cy + shift[1]) * H))
        sig = self.sigma_mm()
        radii = [(sig if sig is not None else cfg.SAFETY_MARGIN_MM)
                 * math.sqrt(i + 1) for i in range(k)]
        pts_mm = [tuple(map(float, self.calib.px_to_mm(u, v)))
                  for u, v in pts_px]
        return Rollout(pts_px, pts_mm, radii, k, "lstm")

    def uncertainty_mm(self, k):
        sig = self.sigma_mm()
        base = sig if sig is not None else cfg.SAFETY_MARGIN_MM
        return base * math.sqrt(max(k, 1))
