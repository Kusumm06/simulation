"""Run-level statistics computed only from this clip's detections and tracks."""

from __future__ import annotations

import math
from typing import Any


def _stat(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "median": median,
        "min": min(values),
        "max": max(values),
    }


def compute_run_metrics(result: dict[str, Any]) -> dict[str, Any]:
    """Return only quantities computed from this run or stored checkpoint metadata."""
    records = result.get("records") or []
    n = len(records)
    detected = [r for r in records if r.get("detected") and r.get("raw_center")]
    primary_id = result.get("primary_track_id")
    tracks = result.get("tracks") or []
    primary_row = next((t for t in tracks if t.get("track_id") == primary_id), None)
    primary_recs = [r for r in records if primary_id is not None and r.get("track_id") == primary_id]
    tracked = [r for r in primary_recs if r.get("detected") and r.get("raw_center")]
    coasting = [r for r in primary_recs if r.get("track_state") == "coasting"]
    missed_n = int(primary_row["misses"]) if primary_row else sum(1 for r in primary_recs if not r.get("detected"))

    confs = [float(r["conf"]) for r in detected if r.get("conf") is not None]
    classes: dict[str, int] = {}
    for r in detected:
        name = r.get("class_name")
        if name is None and r.get("class_id") is not None:
            name = str(r["class_id"])
        if name is not None:
            classes[str(name)] = classes.get(str(name), 0) + 1

    step_disp = []
    for prev, cur in zip(tracked, tracked[1:]):
        if int(cur["index"]) != int(prev["index"]) + 1:
            continue
        x0, y0 = prev["raw_center"]
        x1, y1 = cur["raw_center"]
        step_disp.append(math.hypot(float(x1) - float(x0), float(y1) - float(y0)))

    pred_errors = []
    for rec, nxt in zip(records, records[1:]):
        if not rec.get("lstm_ready") or not rec.get("pred_center"):
            continue
        if not nxt.get("detected") or not nxt.get("raw_center"):
            continue
        if rec.get("track_id") is None or rec.get("track_id") != nxt.get("track_id"):
            continue
        px, py = rec["pred_center"]
        ax, ay = nxt["raw_center"]
        pred_errors.append(math.hypot(float(px) - float(ax), float(py) - float(ay)))

    last_det = tracked[-1] if tracked else (detected[-1] if detected else None)
    last_pred = None
    for rec in reversed(records):
        if rec.get("lstm_ready") and rec.get("pred_center"):
            last_pred = rec
            break

    return {
        "frames_processed": n,
        "frames_with_detection": len(detected),
        "tracked_frames": len(tracked),
        "missed_detections": missed_n,
        "coasting_frames": len(coasting),
        "detection_coverage": (len(detected) / n) if n else None,
        "primary_track_id": primary_id,
        "confidence": _stat(confs),
        "class_counts": classes,
        "consecutive_center_displacement_px": _stat(step_disp),
        "lstm_one_step_error_px": _stat(pred_errors),
        "lstm_ready_frames": sum(1 for r in records if r.get("lstm_ready")),
        "last_detection": last_det,
        "last_prediction": last_pred,
        "tracks": result.get("tracks") or [],
        "unavailable": [
            "Official validation mAP / precision / recall are not stored with models/best.pt in this workspace.",
            "No held-out tracking ADE/FDE report is present in the repository.",
        ],
        "notes": [
            "Coverage is frames with a YOLO detection / processed frames. It is not mAP.",
            "Tracked frames count associated detections on the primary track only. Centers are raw box midpoints.",
            "Missed detections are frames after a track exists where no box was associated. No interpolated centers are inserted.",
            "Average displacement uses consecutive-frame pairs on the same track.",
            "LSTM one-step error compares a prediction at frame t with the next real associated center at t+1 on the same track.",
            "LSTM runs only after 20 consecutive real detections on that track.",
        ],
    }
