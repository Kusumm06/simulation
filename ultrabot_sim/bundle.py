"""
Plan bundle: what the Streamlit dashboard hands to the 3D simulation.

Schema (JSON, validated by inputs.validate_bundle on load):

    source_video     path of the analysed video
    planning_frame   frame index the dashboard planned on
    records          all FrameRecords of the run (ultrabot.pipeline)
    lstm_meta        {"sequence_length", "hidden_size", "epochs"}
    class_names      {"0": "benign", "1": "malignant"}
    pane_size        [990, 720]
    conf_threshold   YOLO confidence used
    src_fps          video frame rate
    is_synthetic     true for the dashboard's synthetic demo
    optimizer        the dashboard's optimize_biopsy_paths() rows without the
                     BiopsyPlan objects (recommended / shortest / safest /
                     top3 / counts / weights / formula)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from ultrabot_sim import config as cfg


def _strip_row(row):
    if row is None:
        return None
    return {k: v for k, v in row.items() if k != "plan"}


def make_bundle(result: dict, opt: dict | None, planning_frame: int | None):
    """result = PipelineResult.to_dict() (as stored by the dashboard)."""
    recs = []
    for r in result.get("records", []):
        recs.append({k: (list(v) if isinstance(v, tuple) else v)
                     for k, v in r.items()})
    out = {
        "source_video": str(result.get("source_video", "")),
        "planning_frame": int(planning_frame) if planning_frame else None,
        "records": recs,
        "lstm_meta": result.get("lstm_meta", {}),
        "class_names": {str(k): v for k, v in
                        (result.get("class_names") or {}).items()},
        "pane_size": [cfg.PANE_W_PX, cfg.PANE_H_PX],
        "conf_threshold": result.get("conf_threshold", cfg.CONF),
        "src_fps": result.get("src_fps", 5.0),
        "is_synthetic": bool(result.get("is_synthetic", False)),
        "optimizer": None,
    }
    if opt:
        out["optimizer"] = {
            "recommended": _strip_row(opt.get("recommended")),
            "shortest": _strip_row(opt.get("shortest")),
            "safest": _strip_row(opt.get("safest")),
            "top3": [_strip_row(r) for r in opt.get("top3", [])],
            "n_candidates": opt.get("n_candidates"),
            "n_feasible": opt.get("n_feasible"),
            "n_rejected": opt.get("n_rejected"),
            "weights": opt.get("weights"), "formula": opt.get("formula"),
        }
    return out


def write_bundle(bundle, path=None):
    path = Path(path or cfg.DEFAULT_BUNDLE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle, default=_default))
    return path


def _default(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:                                       # noqa: BLE001
        pass
    if isinstance(o, (bytes, bytearray)):
        return None
    return str(o)


def launch(bundle_path, extra_args=("--auto",)):
    """Start the simulation in its own window (non-blocking)."""
    cmd = [sys.executable, "-m", "ultrabot_sim.main", "--bundle",
           str(bundle_path), *extra_args]
    return subprocess.Popen(cmd, cwd=str(cfg.ROOT))
