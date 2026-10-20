"""
Perception, frame by frame, using the EXISTING UltraBot pipeline code.

This reproduces the per-frame body of ultrabot.pipeline.run_pipeline()
(same YOLO call, same _yolo_detections(), same CenterIoUTracker, same
LSTM input / output convention) but one frame at a time, so the robot can
react while the video plays. Nothing is re-trained and nothing is copied:
the functions and classes are imported from ultrabot/.

Honesty rules kept from the original code:
  * no detection -> no lesion centre (never interpolated);
  * the LSTM runs only after LSTM sequence length consecutive real hits
    on the primary track, exactly like run_pipeline();
  * --drop / --jitter are explicit robustness experiments and are logged.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ultrabot_sim import config as cfg


@dataclass
class FrameObs:
    index: int
    t: float
    image_bgr: Optional[np.ndarray]
    detected: bool = False
    conf: Optional[float] = None
    class_id: Optional[int] = None
    class_name: Optional[str] = None
    box_xyxy: Optional[tuple] = None
    center_px: Optional[tuple] = None
    pred_center_px: Optional[tuple] = None
    pred_velocity: Optional[tuple] = None
    lstm_ready: bool = False
    track_id: Optional[int] = None
    track_state: Optional[str] = None
    consecutive_hits: int = 0
    detections_this_frame: int = 0
    synthetic: bool = False
    # extras used by the simulation
    track_history: list = field(default_factory=list)   # [(frame, (x, y))]
    lstm_window: Optional[list] = None   # 20 normalised centres used
    infer_ms: float = 0.0
    lstm_status: str = ""
    source: str = "video"


def lstm_status_text(consecutive_hits, seq_len, ready, mode):
    if mode == "image":
        return "not available (single image)"
    if ready:
        return "ready"
    return f"warming {min(consecutive_hits, seq_len)}/{seq_len}"


class Perception:
    """YOLO + CenterIoUTracker + LSTM on a stream of 990x720 panes."""

    def __init__(self, models=None, conf=None, weights=None, lstm=None,
                 drop=0.0, jitter=0.0, seed=0, mode="video"):
        from ultrabot.pipeline import load_models
        from ultrabot.tracker import CenterIoUTracker
        if models is None:
            models = load_models(weights, lstm)
        self.model_yolo, self.model_lstm, self.lstm_meta, self.class_names = \
            models
        self.seq_len = int(self.lstm_meta.get("sequence_length", 20))
        self.conf = cfg.CONF if conf is None else float(conf)
        self.tracker = CenterIoUTracker(max_missed=cfg.TRACK_MAX_MISSED,
                                        max_center_dist=cfg.TRACK_MAX_CENTER_DIST,
                                        min_iou=cfg.TRACK_MIN_IOU)
        self.drop = float(drop)
        self.jitter = float(jitter)
        self.rng = np.random.default_rng(seed)
        self.mode = mode
        self.frames = 0
        self.detections = 0
        self.dropped = 0

    def step(self, pane_bgr, index, t=0.0):
        from ultrabot.pipeline import _yolo_detections, predict_next
        from ultrabot.render import clamp
        from ultrabot.tracker import Detection
        vh, vw = pane_bgr.shape[:2]
        t0 = time.perf_counter()
        results = self.model_yolo(pane_bgr, conf=self.conf, verbose=False)
        infer_ms = (time.perf_counter() - t0) * 1000.0
        dets = _yolo_detections(results, self.class_names)
        # robustness experiments (reference project's --drop / --jitter)
        if self.drop > 0 and dets:
            keep = []
            for d in dets:
                if self.rng.random() < self.drop:
                    self.dropped += 1
                else:
                    keep.append(d)
            dets = keep
        if self.jitter > 0 and dets:
            jd = []
            for d in dets:
                ox, oy = self.rng.normal(0.0, self.jitter, 2)
                x1, y1, x2, y2 = d.box
                jd.append(Detection(box=(x1 + ox, y1 + oy, x2 + ox, y2 + oy),
                                    center=(d.center[0] + ox,
                                            d.center[1] + oy),
                                    conf=d.conf, class_id=d.class_id,
                                    class_name=d.class_name))
            dets = jd
        self.frames += 1
        frame_count = int(index)
        matches = self.tracker.update(frame_count, dets)
        primary = self.tracker.primary_track()
        primary_match = None
        if primary is not None:
            for item in matches:
                if item.track.track_id == primary.track_id:
                    primary_match = item
                    break
        matched = primary_match.detection if primary_match is not None \
            else None
        obs = FrameObs(index=frame_count, t=t, image_bgr=pane_bgr,
                       detections_this_frame=len(dets), infer_ms=infer_ms,
                       source=self.mode)
        if matched is not None:
            self.detections += 1
            obs.detected = True
            obs.center_px = (float(matched.center[0]),
                             float(matched.center[1]))
            obs.box_xyxy = tuple(float(v) for v in matched.box)
            obs.conf = float(matched.conf)
            obs.class_id = matched.class_id
            obs.class_name = matched.class_name
        if primary is not None:
            obs.track_id = primary.track_id
            obs.track_state = primary.state
            obs.consecutive_hits = primary.consecutive_hits
            obs.track_history = [(h["frame"], tuple(h["center"]))
                                 for h in primary.history[-cfg.TRAIL_LEN:]]
        # LSTM: identical gating and maths to run_pipeline()
        if self.mode != "image" and obs.detected and primary is not None \
                and primary.consecutive_hits >= self.seq_len:
            run = primary.history[-self.seq_len:]
            frames = [h["frame"] for h in run]
            consecutive = all(frames[i] == frames[i - 1] + 1
                              for i in range(1, len(frames)))
            if consecutive:
                seq = [[h["center"][0] / vw, h["center"][1] / vh]
                       for h in run]
                vx, vy = predict_next(self.model_lstm, seq)
                raw_cx, raw_cy = obs.center_px
                px = clamp((raw_cx / vw + float(vx)) * vw, 0, vw - 1)
                py = clamp((raw_cy / vh + float(vy)) * vh, 0, vh - 1)
                # run_pipeline stores int() of the predicted point
                obs.pred_center_px = (float(int(px)), float(int(py)))
                obs.pred_velocity = (float(vx), float(vy))
                obs.lstm_ready = True
                obs.lstm_window = seq
        obs.lstm_status = lstm_status_text(obs.consecutive_hits, self.seq_len,
                                           obs.lstm_ready, self.mode)
        return obs


class RecordReplay:
    """
    Observations from stored FrameRecords (dashboard bundle or the
    synthetic demo): exactly what was computed there, no new YOLO run.
    The LSTM checkpoint (if available) is only used for multi-step
    rollouts on the stored centres.
    """

    def __init__(self, records, seq_len=20, mode="bundle", synthetic=False):
        self.records = {int(r["index"]): r for r in records}
        self.seq_len = int(seq_len)
        self.mode = mode
        self.synthetic = synthetic
        self.history = {}           # track_id -> [(frame, center)]

    def step(self, pane_bgr, index, t=0.0):
        r = self.records.get(int(index))
        obs = FrameObs(index=int(index), t=t, image_bgr=pane_bgr,
                       synthetic=self.synthetic, source=self.mode)
        if r is None:
            obs.lstm_status = "no record for this frame"
            return obs
        obs.detected = bool(r.get("detected"))
        obs.track_id = r.get("track_id")
        obs.track_state = r.get("track_state")
        obs.consecutive_hits = int(r.get("consecutive_hits") or 0)
        obs.detections_this_frame = int(r.get("detections_this_frame") or 0)
        if obs.detected:
            c = r.get("raw_center")
            obs.center_px = (float(c[0]), float(c[1]))
            obs.box_xyxy = tuple(float(v) for v in r["box_xyxy"])
            obs.conf = r.get("conf")
            obs.class_id = r.get("class_id")
            obs.class_name = r.get("class_name")
            if obs.track_id is not None:
                h = self.history.setdefault(obs.track_id, [])
                h.append((obs.index, obs.center_px))
                del h[:-max(cfg.TRAIL_LEN, self.seq_len)]
        if obs.track_id is not None:
            obs.track_history = list(self.history.get(obs.track_id, []))[
                -cfg.TRAIL_LEN:]
        if r.get("lstm_ready") and r.get("pred_center"):
            obs.lstm_ready = True
            obs.pred_center_px = tuple(float(v) for v in r["pred_center"])
            if r.get("pred_velocity"):
                obs.pred_velocity = tuple(float(v)
                                          for v in r["pred_velocity"])
            hist = self.history.get(obs.track_id, [])
            run = hist[-self.seq_len:]
            if len(run) == self.seq_len and all(
                    run[i][0] == run[i - 1][0] + 1
                    for i in range(1, len(run))):
                W, H = cfg.PANE_W_PX, cfg.PANE_H_PX
                obs.lstm_window = [[c[0] / W, c[1] / H] for _, c in run]
        obs.lstm_status = lstm_status_text(obs.consecutive_hits, self.seq_len,
                                           obs.lstm_ready, self.mode)
        return obs
