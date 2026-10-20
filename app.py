"""Streamlit dashboard for the existing UltraBot detection, tracking, and planning pipeline."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ultrabot.pipeline import (
    DEFAULT_OUTPUT,
    DEFAULT_VIDEO,
    LSTM_CHECKPOINT,
    YOLO_WEIGHTS,
    check_model_availability,
    generate_synthetic_pipeline_result,
    get_lstm_checkpoint_path,
    get_project_root,
    get_yolo_weights_path,
    load_models,
    run_pipeline,
    safe_load_models,
    validate_lstm_checkpoint,
    validate_yolo_weights,
)

ROOT = get_project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultrabot.biopsy import (
    DISCLAIMER,
    load_ultrasound_pane,
    optimize_biopsy_paths,
    plan_biopsy,
    render_biopsy_plan,
    render_planning_workspace,
    suggested_defaults,
)
from ultrabot.metrics import compute_run_metrics
from ultrabot.render import OUT_H, VIDEO_W
from ultrabot.robot_sim import ROBOT_DISCLAIMER, render_robot_frame, run_robot_simulation

try:
    from robotics.streamlit_3d_panel import render_3d_simulation_panel
except ImportError:
    render_3d_simulation_panel = None

UPLOAD_DIR = ROOT / "outputs" / "uploads"
DASH_DIR = ROOT / "outputs" / "dashboard"
SAMPLE_EXCLUDE = {"output.mp4", "ultrabot_live.mp4", "dashboard_run.mp4"}
FOOTER = "ULTRABOT AI | Research Prototype · Virtual simulation only — not for clinical use or robotic control."
PLOT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0b1218",
    font=dict(color="#c5d0da", size=11, family="Segoe UI, sans-serif"),
    margin=dict(l=42, r=12, t=36, b=36),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
)


def _list_sample_videos():
    videos = []
    out = ROOT / "outputs"
    if out.exists():
        for path in sorted(out.glob("*.mp4")):
            if path.name in SAMPLE_EXCLUDE or path.name.startswith("dashboard_"):
                continue
            videos.append(path)
    if UPLOAD_DIR.exists():
        videos.extend(sorted(UPLOAD_DIR.glob("*.mp4")))
        videos.extend(sorted(UPLOAD_DIR.glob("*.avi")))
    return videos


@st.cache_resource
def get_models(w_path=None, l_path=None):
    return load_models(w_path, l_path)


def _save_upload(upload) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / Path(upload.name).name
    dest.write_bytes(upload.getbuffer())
    return dest


def _device_label() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return f"CUDA ({torch.cuda.get_device_name(0)})"
    except Exception:
        pass
    return "CPU"


def _records_table(records: list[dict]) -> pd.DataFrame:
    rows = []
    for rec in records:
        raw = rec.get("raw_center") or (None, None)
        pred = rec.get("pred_center") or (None, None)
        rows.append(
            {
                "frame": rec.get("index"),
                "track_id": rec.get("track_id"),
                "state": rec.get("track_state"),
                "detected": rec.get("detected"),
                "class": rec.get("class_name"),
                "confidence": rec.get("conf"),
                "lesion_x": None if raw[0] is None else round(raw[0], 1),
                "lesion_y": None if raw[1] is None else round(raw[1], 1),
                "pred_x": None if pred[0] is None else round(pred[0], 1),
                "pred_y": None if pred[1] is None else round(pred[1], 1),
                "match_iou": None if rec.get("match_iou") is None else round(rec["match_iou"], 3),
                "match_dist_px": None if rec.get("match_distance") is None else round(rec["match_distance"], 1),
                "consecutive_hits": rec.get("consecutive_hits"),
                "lstm_ready": rec.get("lstm_ready"),
            }
        )
    return pd.DataFrame(rows)


def _fmt_pt(pt):
    if not pt:
        return "—"
    return f"({pt[0]:.1f}, {pt[1]:.1f})"


def _stat_line(stat, unit=""):
    if not stat:
        return "Not available for this clip"
    return (
        f"n={stat['n']}  ·  mean {stat['mean']:.2f}{unit}  ·  "
        f"median {stat['median']:.2f}{unit}  ·  "
        f"min {stat['min']:.2f}{unit}  ·  max {stat['max']:.2f}{unit}"
    )


def _style_figure(fig, title, height=260):
    fig.update_layout(height=height, title=dict(text=title, font=dict(size=12)), **PLOT)
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)", zeroline=False)
    return fig


def _trajectory_figure(records: list[dict], primary_track_id=None, height=260):
    fig = go.Figure()
    by_track: dict[int, list] = {}
    px, py, pframes = [], [], []
    for rec in records:
        if rec.get("detected") and rec.get("raw_center") and rec.get("track_id") is not None:
            by_track.setdefault(int(rec["track_id"]), []).append(rec)
        if rec.get("lstm_ready") and rec.get("pred_center"):
            px.append(rec["pred_center"][0])
            py.append(rec["pred_center"][1])
            pframes.append(rec["index"])
    colors = ["#3d9ead", "#7bdff2", "#9b5de5", "#d4a017"]
    for i, (tid, rows) in enumerate(sorted(by_track.items())):
        label = f"Track {tid}"
        if tid == primary_track_id:
            label += " primary"
        fig.add_trace(
            go.Scatter(
                x=[r["raw_center"][0] for r in rows],
                y=[r["raw_center"][1] for r in rows],
                mode="lines+markers",
                name=label,
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=5),
            )
        )
    if px:
        fig.add_trace(
            go.Scatter(
                x=px,
                y=py,
                mode="markers",
                name="LSTM prediction",
                marker=dict(color="#e05a6a", size=7, symbol="x"),
                hovertext=[f"frame {i}" for i in pframes],
            )
        )
    fig.update_yaxes(autorange="reversed", title="y (px)")
    fig.update_xaxes(title="x (px)")
    if not by_track:
        fig.add_annotation(text="No associated detections", showarrow=False)
    return _style_figure(fig, "Lesion trajectory", height)


def _lstm_compare_figure(records: list[dict], height=260):
    frames, ax, ay, px, py = [], [], [], [], []
    for rec, nxt in zip(records, records[1:]):
        if not rec.get("lstm_ready") or not rec.get("pred_center"):
            continue
        if not nxt.get("detected") or not nxt.get("raw_center"):
            continue
        if rec.get("track_id") is None or rec.get("track_id") != nxt.get("track_id"):
            continue
        frames.append(nxt.get("index"))
        ax.append(nxt["raw_center"][0])
        ay.append(nxt["raw_center"][1])
        px.append(rec["pred_center"][0])
        py.append(rec["pred_center"][1])
    fig = go.Figure()
    if frames:
        fig.add_trace(go.Scatter(x=ax, y=ay, mode="markers+lines", name="Actual center", line=dict(color="#3d9ead", width=2), marker=dict(size=6)))
        fig.add_trace(go.Scatter(x=px, y=py, mode="markers", name="LSTM predicted", marker=dict(color="#e05a6a", size=7, symbol="x")))
    else:
        fig.add_annotation(text="LSTM not ready or no next-frame pairs", showarrow=False)
    fig.update_yaxes(autorange="reversed", title="y (px)")
    fig.update_xaxes(title="x (px)")
    return _style_figure(fig, "Actual vs predicted", height)


def _confidence_figure(records: list[dict], height=260):
    frames, confs = [], []
    for rec in records:
        if rec.get("detected") and rec.get("conf") is not None:
            frames.append(rec.get("index"))
            confs.append(float(rec["conf"]))
    fig = go.Figure()
    if frames:
        fig.add_trace(go.Scatter(x=frames, y=confs, mode="lines+markers", name="confidence", line=dict(color="#3d9ead", width=2), marker=dict(size=4)))
    else:
        fig.add_annotation(text="No detections", showarrow=False)
    fig.update_yaxes(range=[0, 1], title="confidence")
    fig.update_xaxes(title="frame")
    return _style_figure(fig, "Detection confidence", height)


def _candidate_path_figure(opt: dict, manual_plan=None, height=260):
    fig = go.Figure()
    candidates = list(opt.get("candidates") or [])
    recommended = opt.get("recommended")
    shortest = opt.get("shortest")
    safest = opt.get("safest")
    src = recommended or shortest or (candidates[0] if candidates else None)
    lesion = src["plan"].lesion if src else (manual_plan.lesion if manual_plan else None)
    predicted = src["plan"].predicted if src else (manual_plan.predicted if manual_plan else None)
    safety = src["plan"].safety_margin_px if src else (manual_plan.safety_margin_px if manual_plan else 0.0)

    if lesion is not None and safety > 0:
        theta = np.linspace(0, 2 * math.pi, 80)
        fig.add_trace(
            go.Scatter(
                x=lesion[0] + safety * np.cos(theta),
                y=lesion[1] + safety * np.sin(theta),
                mode="lines",
                fill="toself",
                name="Safety region",
                line=dict(color="rgba(61,158,173,0.9)", width=1),
                fillcolor="rgba(61,158,173,0.14)",
                hoverinfo="skip",
            )
        )

    def _seg(rows):
        xs, ys = [], []
        for row in rows:
            xs.extend([row["entry"][0], row["target"][0], None])
            ys.extend([row["entry"][1], row["target"][1], None])
        return xs, ys

    def _is(row, other):
        return other is not None and abs(row["entry"][0] - other["entry"][0]) < 0.05 and abs(row["entry"][1] - other["entry"][1]) < 0.05 and abs(row["target"][0] - other["target"][0]) < 0.05 and abs(row["target"][1] - other["target"][1]) < 0.05

    rejected = [row for row in candidates if not row.get("feasible")]
    feasible = [row for row in candidates if row.get("feasible") and not _is(row, recommended) and not _is(row, shortest) and not _is(row, safest)]
    if rejected:
        sample = rejected[:: max(1, len(rejected) // 18)][:18]
        xs, ys = _seg(sample)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Rejected", line=dict(color="#5a6570", width=1, dash="dot"), hoverinfo="skip"))
    if feasible:
        xs, ys = _seg(feasible)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Feasible", line=dict(color="#7a8f62", width=1), opacity=0.7, hoverinfo="skip"))
    if shortest is not None and not _is(shortest, recommended):
        fig.add_trace(
            go.Scatter(
                x=[shortest["entry"][0], shortest["target"][0]],
                y=[shortest["entry"][1], shortest["target"][1]],
                mode="lines+markers",
                name="Shortest feasible",
                line=dict(color="#d4a017", width=2),
                marker=dict(size=7),
            )
        )
    if safest is not None and not _is(safest, recommended) and not _is(safest, shortest):
        fig.add_trace(
            go.Scatter(
                x=[safest["entry"][0], safest["target"][0]],
                y=[safest["entry"][1], safest["target"][1]],
                mode="lines+markers",
                name="Safest feasible",
                line=dict(color="#2ec4b6", width=2),
                marker=dict(size=7),
            )
        )
    if recommended:
        fig.add_trace(
            go.Scatter(
                x=[recommended["entry"][0], recommended["target"][0]],
                y=[recommended["entry"][1], recommended["target"][1]],
                mode="lines+markers",
                name="Recommended",
                line=dict(color="#7bdff2", width=4),
                marker=dict(size=10, color="#7bdff2"),
            )
        )
    if lesion is not None:
        fig.add_trace(go.Scatter(x=[lesion[0]], y=[lesion[1]], mode="markers", name="Lesion", marker=dict(color="#70d25a", size=11)))
    if predicted is not None and (lesion is None or math.hypot(predicted[0] - lesion[0], predicted[1] - lesion[1]) >= 8):
        fig.add_trace(go.Scatter(x=[predicted[0]], y=[predicted[1]], mode="markers", name="LSTM target", marker=dict(color="#e05a6a", size=10, symbol="x")))
    if manual_plan is not None:
        rec_same = recommended is not None and abs(manual_plan.entry[0] - recommended["entry"][0]) < 2 and abs(manual_plan.entry[1] - recommended["entry"][1]) < 2
        if not rec_same:
            fig.add_trace(
                go.Scatter(
                    x=[manual_plan.entry[0], manual_plan.target[0]],
                    y=[manual_plan.entry[1], manual_plan.target[1]],
                    mode="lines+markers",
                    name="Manual",
                    line=dict(color="#c77dff", width=2, dash="dash"),
                    marker=dict(size=7, color="#c77dff"),
                )
            )
    fig.update_yaxes(autorange="reversed", title="y (px)")
    fig.update_xaxes(title="x (px)")
    if recommended is None and not candidates:
        fig.add_annotation(text="No candidate paths", showarrow=False)
    return _style_figure(fig, "Ultrasound workspace · candidate trajectories", height)


def _analysis_overlay(pane, last, last_pred, records, primary_id):
    img = pane.copy()
    pts = []
    for rec in records:
        if rec.get("detected") and rec.get("raw_center") and rec.get("track_id") == primary_id:
            pts.append((int(rec["raw_center"][0]), int(rec["raw_center"][1])))
    if len(pts) >= 2:
        cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, (61, 158, 173), 2, cv2.LINE_AA)
    if last and last.get("box_xyxy"):
        x1, y1, x2, y2 = [int(v) for v in last["box_xyxy"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), (61, 158, 173), 2, cv2.LINE_AA)
    if last and last.get("raw_center"):
        cx, cy = int(last["raw_center"][0]), int(last["raw_center"][1])
        cv2.circle(img, (cx, cy), 6, (61, 158, 173), -1, cv2.LINE_AA)
        tid = last.get("track_id")
        conf = last.get("conf")
        label = f"ID {tid if tid is not None else '—'}  {conf:.2f}" if conf is not None else f"ID {tid}"
        cv2.putText(img, label, (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 236, 240), 1, cv2.LINE_AA)
    pred = last_pred.get("pred_center") if last_pred else None
    if pred:
        px, py = int(pred[0]), int(pred[1])
        cv2.drawMarker(img, (px, py), (90, 90, 220), cv2.MARKER_TILTED_CROSS, 14, 2, cv2.LINE_AA)
        cv2.putText(img, "LSTM", (px + 8, py + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 230), 1, cv2.LINE_AA)
    return img


def _kv(label, value):
    return f'<div class="cc-kv"><span>{label}</span><b>{value}</b></div>'


def _section(title, rows):
    body = "".join(_kv(k, v) for k, v in rows)
    return f'<div class="cc-sec"><div class="cc-sec-h">{title}</div>{body}</div>'


def _inject_css():
    st.markdown(
        """
        <style>
        [data-testid="stHeader"] { display: none !important; height: 0 !important; }
        [data-testid="stToolbar"] { display: none !important; }
        [data-testid="stDecoration"] { display: none !important; }
        [data-testid="stAppViewContainer"] { padding-top: 0 !important; }
        [data-testid="stMain"] { padding-top: 0 !important; margin-top: 0 !important; }
        [data-testid="stMainBlockContainer"],
        .stMainBlockContainer,
        .block-container {
            padding-top: 0.7rem !important;
            padding-left: 1.25rem !important;
            padding-right: 1.25rem !important;
            padding-bottom: 1.4rem !important;
            max-width: 100% !important;
        }
        [data-testid="stSidebar"] { background: #080e14; border-right: 1px solid #1a2633; }
        [data-testid="stSidebar"] .block-container { padding-top: 0.7rem !important; max-width: 100% !important; }
        h1,h2,h3,h4,h5 { color: #e8eef4; }
        .cc-top {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 16px;
            margin: 0 0 10px 0;
            padding: 2px 0 0 0;
            min-width: 0;
        }
        .cc-brandwrap { min-width: 0; flex: 1 1 auto; }
        .cc-brand {
            font-size: 1.2rem;
            font-weight: 650;
            letter-spacing: 0.12em;
            color: #f3f7fa;
            margin: 0;
            line-height: 1.2;
            white-space: nowrap;
        }
        .cc-sub {
            font-size: 0.72rem;
            letter-spacing: 0.08em;
            color: #7f96a8;
            margin: 4px 0 0 0;
            line-height: 1.35;
            white-space: normal;
        }
        .cc-online {
            flex: 0 0 auto;
            font-size: 0.72rem;
            letter-spacing: 0.1em;
            color: #8ee0c8;
            font-family: ui-monospace, monospace;
            white-space: nowrap;
            padding-top: 4px;
        }
        .cc-dot { display:inline-block; width:7px; height:7px; border-radius:50%; background:#2ee59d; margin-right:6px; }
        .cc-kpis { display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:8px; margin:0 0 10px 0; }
        .cc-kpi { border:1px solid #1d2b38; background:#0d141c; border-radius:8px; padding:8px 10px; }
        .cc-kpi span { display:block; font-size:0.68rem; letter-spacing:0.08em; color:#7f96a8; }
        .cc-kpi b { display:block; margin-top:3px; font-size:0.98rem; color:#eef3f7; font-weight:600; }
        .cc-panel { border:1px solid #1d2b38; background:#0d141c; border-radius:8px; padding:10px 12px; margin-bottom:10px; }
        .cc-h { font-size:0.72rem; letter-spacing:0.1em; color:#7f96a8; margin:0 0 8px 0; }
        .cc-sec { margin-bottom:10px; }
        .cc-sec-h { font-size:0.68rem; letter-spacing:0.1em; color:#5f8f9c; margin-bottom:4px; }
        .cc-kv { display:flex; justify-content:space-between; gap:10px; font-size:0.8rem; padding:2px 0; color:#c5d0da; }
        .cc-kv b { color:#eef3f7; font-weight:550; text-align:right; }
        .cc-pipe { display:flex; flex-wrap:nowrap; align-items:center; gap:0; margin:6px 0 10px 0; overflow-x:auto; }
        .cc-node { border:1px solid #1d2b38; background:#0d141c; color:#9eb0bf; padding:6px 8px; font-size:0.62rem; letter-spacing:0.04em; border-radius:6px; font-family:ui-monospace,monospace; white-space:nowrap; }
        .cc-node.on { border-color:#2a6f68; color:#b6eadc; background:#10241f; }
        .cc-node.now { border-color:#3d9ead; color:#d7f3f7; background:#12323a; }
        .cc-link { width:12px; height:1px; background:#2a3a48; flex:0 0 12px; }
        .cc-ids { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:8px; margin:8px 0 10px 0; }
        .cc-id { border:1px solid #1d2b38; background:#0d141c; border-radius:8px; padding:8px 10px; }
        .cc-id.rec { border-color:#3d9ead; }
        .cc-id span { display:block; font-size:0.68rem; letter-spacing:0.08em; color:#7f96a8; }
        .cc-id b { display:block; margin-top:4px; color:#eef3f7; font-size:0.88rem; font-weight:600; }
        .cc-id em { display:block; margin-top:3px; color:#9eb0bf; font-size:0.75rem; font-style:normal; }
        .cc-side { font-size:0.68rem; letter-spacing:0.12em; color:#6f8596; margin:10px 0 6px 0; font-weight:600; }
        .cc-foot { margin-top:12px; padding-top:8px; border-top:1px solid #1d2b38; color:#6f8596; font-size:0.75rem; }
        .note { border-left:2px solid #3d9ead; background:#101820; padding:7px 10px; color:#b7c4d0; font-size:0.8rem; margin-bottom:8px; }
        .sim-ok { border-left:2px solid #2ec4b6; background:#10281f; color:#c8f0e6; padding:7px 10px; font-size:0.82rem; }
        .sim-bad { border-left:2px solid #e63946; background:#2a1418; color:#f3cfd3; padding:7px 10px; font-size:0.82rem; }
        [data-testid="stMetric"] { background:#0d141c; border:1px solid #1d2b38; border-radius:8px; padding:6px 8px; }
        [data-testid="stMetricValue"] { font-size:1.05rem !important; }
        .stButton > button { border-radius:6px; height:2.35rem; letter-spacing:0.06em; }
        [data-testid="stTabs"] button { font-size:0.82rem; }
        @media (max-width:1100px){
            .cc-kpis{grid-template-columns:repeat(3,minmax(0,1fr));}
            .cc-ids{grid-template-columns:1fr;}
            .cc-brand{font-size:1.05rem; letter-spacing:0.1em;}
        }
        @media (max-width:720px){
            .cc-top{flex-wrap:wrap; align-items:flex-start;}
            .cc-online{padding-top:0;}
            .cc-kpis{grid-template-columns:repeat(2,minmax(0,1fr));}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _header():
    st.markdown(
        """
        <div class="cc-top">
          <div class="cc-brandwrap">
            <div class="cc-brand">ULTRABOT AI</div>
            <div class="cc-sub">AI-ASSISTED BREAST ULTRASOUND ANALYSIS</div>
          </div>
          <div class="cc-online"><span class="cc-dot"></span>ONLINE</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _kpis(result, metrics, last, rec, sim):
    cov = metrics.get("detection_coverage")
    err = metrics.get("lstm_one_step_error_px")
    items = [
        ("DETECTION", f"{cov:.0%}" if cov is not None else "—"),
        ("TRACKING", "—" if not last or last.get("track_id") is None else f"ID {last.get('track_id')}"),
        ("PREDICTION", f"{err['mean']:.1f} px" if err else "STANDBY"),
        ("PATH PLANNING", f"{rec['path_length']:.0f} px" if rec else "—"),
        ("ROBOT", sim["status_label"] if sim else "STANDBY"),
        ("FRAMES", str(result.get("frames_processed", 0))),
    ]
    html = "".join(f'<div class="cc-kpi"><span>{k}</span><b>{v}</b></div>' for k, v in items)
    st.markdown(f'<div class="cc-kpis">{html}</div>', unsafe_allow_html=True)


def _pipeline_html(result=None, metrics=None, rec=None, sim=None, running=False):
    flags = [
        bool(result) or running,
        bool(result and result.get("detections")),
        bool(metrics and metrics.get("tracked_frames")),
        bool(metrics and metrics.get("lstm_ready_frames")),
        rec is not None,
        rec is not None,
        bool(sim),
    ]
    labels = [
        "VIDEO",
        "YOLO DETECTOR",
        "TEMPORAL TRACKER",
        "LSTM PREDICTION",
        "TRAJECTORY OPTIMIZER",
        "VIRTUAL BIOPSY",
        "ROBOT SIMULATION",
    ]
    first_off = next((i for i, f in enumerate(flags) if not f), None)
    parts = []
    for i, lab in enumerate(labels):
        cls = "on" if flags[i] else ("now" if i == first_off else "")
        if running and i <= 1:
            cls = "now" if i == 1 else "on"
        parts.append(f'<div class="cc-node {cls}">{lab}</div>')
        if i < len(labels) - 1:
            parts.append('<div class="cc-link"></div>')
    return f'<div class="cc-pipe">{"".join(parts)}</div>'


def _footer():
    st.markdown(f'<div class="cc-foot">{FOOTER}</div>', unsafe_allow_html=True)


def _sidebar_label(text):
    st.markdown(f'<div class="cc-side">{text}</div>', unsafe_allow_html=True)


def _planning_flow_html(opt, predicted, rec, sim):
    flags = [
        predicted is not None or (rec is not None and rec["plan"].lesion is not None),
        bool(opt and opt.get("n_candidates")),
        bool(opt and opt.get("n_feasible")),
        rec is not None,
        rec is not None,
        bool(sim),
    ]
    labels = [
        "LSTM TARGET",
        "CANDIDATE TRAJECTORIES",
        "FEASIBILITY + SAFETY",
        "PATH SCORING",
        "RECOMMENDED TRAJECTORY",
        "ROBOT NAVIGATION",
    ]
    first_off = next((i for i, f in enumerate(flags) if not f), None)
    parts = []
    for i, lab in enumerate(labels):
        cls = "on" if flags[i] else ("now" if i == first_off else "")
        parts.append(f'<div class="cc-node {cls}">{lab}</div>')
        if i < len(labels) - 1:
            parts.append('<div class="cc-link"></div>')
    return f'<div class="cc-pipe">{"".join(parts)}</div>'


def _path_identity_html(opt):
    rec = opt.get("recommended")
    shortest = opt.get("shortest")
    safest = opt.get("safest")

    def _card(title, row, extra, rec_cls=False):
        if row is None:
            body = "<b>None</b><em>No feasible path from the optimizer.</em>"
        else:
            body = (
                f"<b>Rank {row.get('rank')} · ({row['entry'][0]:.0f}, {row['entry'][1]:.0f})</b>"
                f"<em>{extra}</em>"
            )
        cls = "cc-id rec" if rec_cls else "cc-id"
        return f'<div class="{cls}"><span>{title}</span>{body}</div>'

    shortest_txt = "—" if shortest is None else f"Path {shortest['path_length']:.1f} px · angle {shortest['needle_angle_deg']:.1f}°"
    if safest is None:
        safest_txt = "—"
    else:
        err = f"{safest['target_error']:.1f} px" if safest["target_error"] is not None else "—"
        safest_txt = f"Safety {safest['safety_violations']} · target error {err}"
    rec_txt = "—" if rec is None else f"Score {rec['total_cost']:.3f} · path {rec['path_length']:.1f} px"
    html = (
        _card("SHORTEST FEASIBLE", shortest, shortest_txt)
        + _card("SAFEST FEASIBLE", safest, safest_txt)
        + _card("RECOMMENDED (BALANCED)", rec, rec_txt, rec_cls=True)
    )
    return f'<div class="cc-ids">{html}</div>'


def _ranked_path_table(opt) -> pd.DataFrame:
    rows = []
    for row in opt.get("ranked") or []:
        rows.append(
            {
                "Rank": row.get("rank"),
                "Entry": f"({row['entry'][0]:.1f}, {row['entry'][1]:.1f})",
                "Path Length": round(row["path_length"], 1),
                "Angle": round(row["needle_angle_deg"], 1),
                "Target Error": None if row["target_error"] is None else round(row["target_error"], 2),
                "Safety": row["safety_violations"],
                "Feasible": bool(row["feasible"]),
                "Score": None if row["total_cost"] is None else round(row["total_cost"], 4),
            }
        )
    return pd.DataFrame(rows)


def _collect_planning_controls(result, last, last_pred):
    lesion = last.get("raw_center") if last else None
    predicted = last_pred.get("pred_center") if last_pred else None
    defaults = suggested_defaults(lesion, predicted)
    run_id = f"{Path(result['source_video']).name}-{result['frames_processed']}"
    max_len = int(math.hypot(VIDEO_W, OUT_H))
    st.caption("Weights re-score optimizer candidates and update the recommended path. They do not edit the manual path.")
    w_path = st.slider("Path length weight", 0.0, 5.0, 1.0, 0.1, key=f"w_path_{run_id}")
    w_safe = st.slider("Safety weight", 0.0, 5.0, 1.0, 0.1, key=f"w_safe_{run_id}")
    w_ang = st.slider("Angle weight", 0.0, 5.0, 1.0, 0.1, key=f"w_ang_{run_id}")
    w_err = st.slider("Target accuracy weight", 0.0, 5.0, 1.0, 0.1, key=f"w_err_{run_id}")
    with st.expander("Advanced options"):
        st.caption("Entry X/Y and Target X/Y edit the manual trajectory only. Needle length and safety margin also apply to optimizer feasibility and safety scoring, but they never replace the recommended path with the manual path.")
        entry_x = st.slider("Entry X", 0, VIDEO_W - 1, int(defaults["entry"][0]), key=f"entry_x_{run_id}")
        entry_y = st.slider("Entry Y", 0, OUT_H - 1, int(defaults["entry"][1]), key=f"entry_y_{run_id}")
        target_x = st.slider("Target X", 0, VIDEO_W - 1, int(defaults["target"][0]), key=f"target_x_{run_id}")
        target_y = st.slider("Target Y", 0, OUT_H - 1, int(defaults["target"][1]), key=f"target_y_{run_id}")
        needle_len = st.slider(
            "Needle length (px)",
            10,
            max_len,
            int(min(max(defaults["needle_length_px"], 10), max_len)),
            key=f"needle_{run_id}",
        )
        safety = st.slider("Safety margin (px)", 0, 80, int(defaults["safety_margin_px"]), key=f"safety_{run_id}")
    return {
        "lesion": lesion,
        "predicted": predicted,
        "entry": (entry_x, entry_y),
        "target": (target_x, target_y),
        "safety": safety,
        "needle_len": needle_len,
        "w_path": w_path,
        "w_ang": w_ang,
        "w_safe": w_safe,
        "w_err": w_err,
    }


def _compute_biopsy_bundle(result, last, last_pred, controls):
    plan = plan_biopsy(
        entry=controls["entry"],
        target=controls["target"],
        lesion=controls["lesion"],
        predicted=controls["predicted"],
        safety_margin_px=controls["safety"],
        needle_length_px=controls["needle_len"],
        frame_index=last.get("index") if last else None,
        track_id=last.get("track_id") if last else None,
    )
    frame_idx = last.get("index") if last else result["frames_processed"]
    pane = load_ultrasound_pane(result["source_video"], frame_idx)
    if pane is None:
        pane = np.zeros((OUT_H, VIDEO_W, 3), dtype=np.uint8)
    opt = optimize_biopsy_paths(
        lesion=controls["lesion"],
        predicted=controls["predicted"],
        user_target=None,
        safety_margin_px=controls["safety"],
        needle_length_px=controls["needle_len"],
        path_length_weight=controls["w_path"],
        angle_weight=controls["w_ang"],
        target_error_weight=controls["w_err"],
        safety_weight=controls["w_safe"],
        frame_index=last.get("index") if last else None,
        track_id=last.get("track_id") if last else None,
    )
    rec = opt["recommended"]
    st.session_state["robot_sim_ctx"] = {
        "recommended": rec,
        "pane": pane,
        "path_key": _recommended_key(rec),
        "frame_index": frame_idx,
        "n_rejected": opt["n_rejected"],
    }
    return plan, opt, pane, frame_idx


def _recommended_key(rec):
    if rec is None:
        return None
    return (
        round(float(rec["entry"][0]), 1),
        round(float(rec["entry"][1]), 1),
        round(float(rec["target"][0]), 1),
        round(float(rec["target"][1]), 1),
        round(float(rec["path_length"]), 2),
        float(rec["plan"].needle_length_px),
        float(rec["plan"].safety_margin_px),
    )


def _capture_nav(rec, pane, frame_idx):
    payload = {
        "recommended": rec,
        "pane": pane,
        "path_key": _recommended_key(rec),
        "frame_index": frame_idx,
    }
    st.session_state["robot_nav"] = payload
    return payload


def _run_recommended_navigation(payload):
    rec = payload["recommended"]
    DASH_DIR.mkdir(parents=True, exist_ok=True)
    sim = run_robot_simulation(payload["pane"], rec, DASH_DIR / "robot_sim.gif")
    sim["path_key"] = payload["path_key"]
    st.session_state["robot_sim_result"] = sim
    return sim


def _render_trajectory_planning(last, last_pred, plan, opt, pane, frame_idx, sim):
    st.markdown(f'<div class="note">{DISCLAIMER}</div>', unsafe_allow_html=True)
    predicted = last_pred.get("pred_center") if last_pred else None
    rec = opt.get("recommended")
    st.markdown(_planning_flow_html(opt, predicted, rec, sim), unsafe_allow_html=True)
    overlay = render_planning_workspace(pane, opt, manual_plan=plan)
    DASH_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(DASH_DIR / "biopsy_plan.png"), overlay)

    vis, info = st.columns([1.35, 1])
    with vis:
        st.image(
            cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
            caption=f"Ultrasound workspace · planning frame {frame_idx}",
            width="stretch",
        )
        st.plotly_chart(_candidate_path_figure(opt, plan, 340), width="stretch", key="chart_candidate_paths")
    with info:
        st.caption("Recommended path is the lowest-cost feasible candidate from the optimizer. Manual Advanced options do not replace it.")
        st.markdown(_path_identity_html(opt), unsafe_allow_html=True)
        if rec is None:
            st.warning(f"No feasible path ({opt['n_rejected']} rejected). Robot navigation is unavailable.")
        else:
            a, b = st.columns(2)
            a.metric("Recommended entry", f"({rec['entry'][0]:.0f}, {rec['entry'][1]:.0f})")
            b.metric("Recommended target", f"({rec['target'][0]:.0f}, {rec['target'][1]:.0f})")
            c, d, e = st.columns(3)
            c.metric("Path length", f"{rec['path_length']:.1f} px")
            d.metric("Angle", f"{rec['needle_angle_deg']:.1f}°")
            e.metric("Score", f"{rec['total_cost']:.3f}")
            f, g, h = st.columns(3)
            f.metric("Target error", "—" if rec["target_error"] is None else f"{rec['target_error']:.1f} px")
            g.metric("Safety", rec["safety_violations"])
            h.metric("Feasible", "Yes" if rec["feasible"] else "No")
            nav_clicked = st.button("NAVIGATE RECOMMENDED PATH", type="primary", width="stretch")
            if nav_clicked:
                payload = _capture_nav(rec, pane, frame_idx)
                with st.spinner("Sending recommended trajectory to robot simulation…"):
                    _run_recommended_navigation(payload)
                st.success(
                    f"Recommended trajectory sent to Robot Simulation: "
                    f"({rec['entry'][0]:.1f}, {rec['entry'][1]:.1f}) → "
                    f"({rec['target'][0]:.1f}, {rec['target'][1]:.1f})"
                )
        st.caption(f"{opt['n_feasible']} feasible / {opt['n_rejected']} rejected · {opt['formula']}")
        if st.button("Open 3D Robotic Simulation", width="stretch"):
            try:
                from ultrabot_sim.bundle import launch, make_bundle, write_bundle

                bundle_path = write_bundle(make_bundle(st.session_state.get("result") or {}, opt, frame_idx))
                launch(bundle_path)
                st.info(f"3D simulation is opening in a separate window (plan bundle: {bundle_path.name}).")
            except Exception as exc:
                st.error(f"Could not start the 3D simulation: {exc}")

    st.markdown('<div class="cc-h">RANKED CANDIDATES</div>', unsafe_allow_html=True)
    table = _ranked_path_table(opt)
    if table.empty:
        st.write("No feasible candidates to rank.")
    else:
        st.dataframe(table, width="stretch", hide_index=True, height=min(360, 48 + 28 * min(len(table), 12)))

    st.markdown('<div class="cc-h">MANUAL TRAJECTORY (ADVANCED OPTIONS)</div>', unsafe_allow_html=True)
    st.caption("This path is independent of the optimizer recommendation. Robot navigation does not use it.")
    cls = "sim-ok" if plan.success else "sim-bad"
    st.markdown(f'<div class="{cls}"><b>{plan.status_label}</b> — {plan.status_reason}</div>', unsafe_allow_html=True)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Manual entry", f"({plan.entry[0]:.0f}, {plan.entry[1]:.0f})")
    m2.metric("Manual target", f"({plan.target[0]:.0f}, {plan.target[1]:.0f})")
    m3.metric("Path length", f"{plan.trajectory_length_px:.1f} px")
    m4.metric("Needle angle", f"{plan.needle_angle_deg:.1f}°")
    m5.metric("Tip → target", f"{plan.tip_to_target_px:.1f} px")
    n1, n2, n3, n4 = st.columns(4)
    n1.metric("Needle length", f"{plan.needle_length_px:.0f} px")
    n2.metric("Safety margin", f"{plan.safety_margin_px:.0f} px")
    n3.metric("Target error", "—" if plan.target_error_px is None else f"{plan.target_error_px:.1f} px")
    n4.metric("Reach", "Yes" if plan.reach_ok else "No")
    d1, d2 = st.columns(2)
    d1.write(f"**Lesion:** {_fmt_pt(plan.lesion)}")
    d2.write(f"**LSTM target:** {_fmt_pt(plan.predicted)}")
    if last:
        st.caption(f"Track {last.get('track_id')} · frame {last.get('index')}")


def _render_virtual_biopsy(last, plan, pane, frame_idx):
    st.markdown(f'<div class="note">{DISCLAIMER}</div>', unsafe_allow_html=True)
    st.caption("Manual trajectory from Advanced options. The optimizer recommended path lives in Trajectory Planning and is not overwritten here.")
    overlay = render_biopsy_plan(pane, plan)
    DASH_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(DASH_DIR / "manual_biopsy_plan.png"), overlay)
    vis, info = st.columns([1.3, 1])
    with vis:
        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption=f"Manual plan · frame {frame_idx}", width="stretch")
    with info:
        cls = "sim-ok" if plan.success else "sim-bad"
        st.markdown(f'<div class="{cls}"><b>{plan.status_label}</b> — {plan.status_reason}</div>', unsafe_allow_html=True)
        a, b = st.columns(2)
        a.metric("Entry", f"({plan.entry[0]:.0f}, {plan.entry[1]:.0f})")
        b.metric("Target", f"({plan.target[0]:.0f}, {plan.target[1]:.0f})")
        c, d = st.columns(2)
        c.metric("Path length", f"{plan.trajectory_length_px:.1f} px")
        d.metric("Needle angle", f"{plan.needle_angle_deg:.1f}°")
        e, f = st.columns(2)
        e.metric("Tip → target", f"{plan.tip_to_target_px:.1f} px")
        f.metric("Target error", "—" if plan.target_error_px is None else f"{plan.target_error_px:.1f} px")
        g, h = st.columns(2)
        g.metric("Needle length", f"{plan.needle_length_px:.0f} px")
        h.metric("Safety margin", f"{plan.safety_margin_px:.0f} px")
    d1, d2, d3, d4 = st.columns(4)
    d1.write(f"**Lesion:** {_fmt_pt(plan.lesion)}")
    d2.write(f"**LSTM target:** {_fmt_pt(plan.predicted)}")
    d3.write(f"**Reach:** {'Yes' if plan.reach_ok else 'No'}")
    d4.write(f"**Alignment:** {'Yes' if plan.alignment_ok else '—' if plan.alignment_ok is None else 'No'}")
    if last:
        st.caption(f"Track {last.get('track_id')} · frame {last.get('index')}")


@st.fragment
def _render_robot_simulation(pipeline_result=None):
    st.markdown(f'<div class="note">{ROBOT_DISCLAIMER}</div>', unsafe_allow_html=True)
    is_synth = pipeline_result.get("is_synthetic", False) if isinstance(pipeline_result, dict) else True
    mode = "full" if (pipeline_result is not None and not is_synth) else "synthetic"

    # Main Area: Interactive 3D Articulated Robot Arm Simulation
    if render_3d_simulation_panel is not None:
        render_3d_simulation_panel(pipeline_result=pipeline_result, mode=mode)
    else:
        st.error("Robotics 3D simulation module not available.")

    # Optional Collapsible Reference: 2D Ultrasound Trajectory Overlay
    with st.expander("🖼️ 2D Ultrasound Trajectory Reference", expanded=False):
        nav = st.session_state.get("robot_nav")
        ctx = st.session_state.get("robot_sim_ctx")
        payload = nav if nav and nav.get("recommended") is not None else None
        live_rec = None if not ctx else ctx.get("recommended")
        rec = None if payload is None else payload.get("recommended")
        if rec is None:
            if live_rec is not None:
                st.info("A recommended path is available. Click NAVIGATE RECOMMENDED PATH in Trajectory Planning to send that exact path here.")
                preview = render_robot_frame(ctx["pane"], live_rec, 0.0, status_label="PLANNED")
                st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), caption="Live optimizer path (not yet navigated)", width="stretch")
            else:
                st.caption("No recommended 2D trajectory captured yet.")
        else:
            pane = payload["pane"]
            preview = render_robot_frame(pane, rec, 0.0, status_label="PLANNED")
            DASH_DIR.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(DASH_DIR / "robot_sim_planned.png"), preview)
            drifted = live_rec is not None and payload.get("path_key") != _recommended_key(live_rec)
            left, right = st.columns([1.3, 1])
            with left:
                st.image(
                    cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                    caption=f"Entry → trajectory → lesion target · frame {payload.get('frame_index')}",
                    width="stretch",
                )
            with right:
                st.caption("Robot follows the captured optimizer recommended path.")
                a, b = st.columns(2)
                a.metric("Path length", f"{rec['path_length']:.1f} px")
                b.metric("Needle angle", f"{rec['needle_angle_deg']:.1f}°")
                c, d = st.columns(2)
                c.metric("Target error", "—" if rec["target_error"] is None else f"{rec['target_error']:.1f} px")
                d.metric("Safety violations", rec["safety_violations"])
                replay = st.button("REPLAY 2D NEEDLE INSERTION", width="stretch")
            if drifted:
                st.caption("Planning weights changed the live recommended path. Click NAVIGATE RECOMMENDED PATH again to send the new path.")
            if replay:
                with st.spinner("Inserting virtual needle…"):
                    _run_recommended_navigation(payload)
            sim = st.session_state.get("robot_sim_result")
            if sim and sim.get("path_key") == payload.get("path_key"):
                cls = "sim-ok" if sim["success"] else "sim-bad"
                st.markdown(f'<div class="{cls}"><b>{sim["status_label"]}</b> — {sim["status_reason"]}</div>', unsafe_allow_html=True)
                gif_path = Path(sim["gif_path"])
                if gif_path.exists():
                    st.image(str(gif_path), caption="Virtual needle along the recommended path", width="stretch")


def _experiment_table(result, metrics, rec, sim):
    cov = metrics.get("detection_coverage")
    conf = metrics.get("confidence")
    err = metrics.get("lstm_one_step_error_px")
    disp = metrics.get("consecutive_center_displacement_px")
    rows = [
        ("Detection coverage", f"{cov:.0%}" if cov is not None else "—", "Frames with a YOLO box / processed frames. Not mAP."),
        ("Mean confidence", f"{conf['mean']:.2f}" if conf else "—", "Mean YOLO score on detected frames."),
        ("Track length", str(metrics.get("tracked_frames")), "Associated detections on the primary track."),
        ("LSTM error", f"{err['mean']:.1f} px" if err else "—", "Prediction at t vs next real center on the same track."),
        ("Path length", f"{rec['path_length']:.1f} px" if rec else "—", "Recommended entry-to-target distance."),
        (
            "Target error",
            "—" if not rec or rec.get("target_error") is None else f"{rec['target_error']:.1f} px",
            "Recommended target vs lesion center.",
        ),
        ("Safety violations", "—" if not rec else str(rec["safety_violations"]), "0 = recommended path stays inside the safety rule."),
        ("Simulation result", sim["status_label"] if sim else "STANDBY", "Virtual needle reach vs safety margin."),
    ]
    if disp:
        rows.insert(3, ("Center displacement", f"{disp['mean']:.1f} px", "Mean step between consecutive real centers."))
    st.dataframe(pd.DataFrame(rows, columns=["Metric", "Value", "Interpretation"]), width="stretch", hide_index=True)
    st.caption("Official validation mAP / ADE are not stored with the weights and are not shown.")


def _inference_panel(result, metrics, last, last_pred, detector_label):
    pred_src = last_pred or (last if last and last.get("lstm_ready") else None)
    err = metrics.get("lstm_one_step_error_px")
    disp = metrics.get("consecutive_center_displacement_px")
    html = (
        _section("MODEL", [("Detector", detector_label), ("Weights", Path(result["weights_path"]).name), ("Device", _device_label())])
        + _section(
            "DETECTION",
            [
                ("Class", (last.get("class_name") or last.get("class_id") or "—") if last else "—"),
                ("Confidence", f"{last['conf']:.3f}" if last and last.get("conf") is not None else "—"),
                ("Bounding box", str(tuple(round(v, 1) for v in last["box_xyxy"])) if last and last.get("box_xyxy") else "—"),
                ("Center", _fmt_pt(last.get("raw_center")) if last else "—"),
            ],
        )
        + _section(
            "TRACKING",
            [
                ("Track ID", "—" if not last or last.get("track_id") is None else str(last.get("track_id"))),
                ("Coverage", f"{metrics['detection_coverage']:.0%}" if metrics.get("detection_coverage") is not None else "—"),
                ("Misses", str(metrics.get("missed_detections"))),
                ("Displacement", f"{disp['mean']:.1f} px" if disp else "—"),
            ],
        )
        + _section(
            "PREDICTION",
            [
                ("LSTM status", "READY" if pred_src and pred_src.get("pred_center") else "WAITING"),
                ("Next center", _fmt_pt(pred_src.get("pred_center")) if pred_src else "—"),
                ("Error", f"{err['mean']:.1f} px" if err else "—"),
            ],
        )
    )
    st.markdown(f'<div class="cc-panel"><div class="cc-h">AI INFERENCE</div>{html}</div>', unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="ULTRABOT", layout="wide", initial_sidebar_state="expanded")
    _inject_css()
    _header()

    model_status = check_model_availability(YOLO_WEIGHTS, LSTM_CHECKPOINT)
    samples = _list_sample_videos()
    sample_labels = {str(p): f"{p.name}  ({p.parent.name})" for p in samples}

    with st.sidebar:
        _sidebar_label("DATA")
        source_mode = st.radio("Source", ["Sample", "Upload"], index=0, horizontal=True)
        video_path = None
        if source_mode == "Sample":
            if samples:
                choice = st.selectbox("Video", list(sample_labels.keys()), format_func=lambda k: sample_labels[k])
                video_path = Path(choice)
            else:
                st.warning("No sample MP4 in outputs/.")
        else:
            upload = st.file_uploader("Ultrasound video", type=["mp4", "avi", "mov"])
            if upload is not None:
                video_path = _save_upload(upload)

        max_frames = st.slider("Max frames (0 = all)", 0, 400, 100, 10)

        _sidebar_label("DETECTOR & WEIGHTS")
        if model_status["yolo_available"]:
            st.markdown("🟢 **YOLO Detector**: Ready")
        else:
            st.markdown("⚠️ **YOLO Detector**: Weights Missing")
            st.caption(f"Required: `{Path(model_status['yolo_path']).name}`")

        if model_status["lstm_available"]:
            st.markdown("🟢 **LSTM Predictor**: Ready")
        else:
            st.markdown("⚠️ **LSTM Predictor**: Checkpoint Missing")
            st.caption(f"Required: `{Path(model_status['lstm_path']).name}`")

        conf = st.slider("Confidence", 0.05, 0.80, 0.10, 0.05)

        _sidebar_label("ACTION")
        can_run_real = (video_path is not None) and model_status["all_available"]
        run_real_clicked = st.button(
            "RUN REAL AI INFERENCE",
            type="primary" if can_run_real else "secondary",
            width="stretch",
            disabled=not can_run_real,
            help=(
                "Run real YOLOv8 detection and LSTM prediction on the video."
                if can_run_real
                else "Real AI inference is disabled: requires video and trained models/best.pt + models/lstm_checkpoint.pth."
            ),
        )
        run_synthetic_clicked = st.button(
            "RUN SYNTHETIC SIMULATION DEMO",
            type="primary" if not can_run_real else "secondary",
            width="stretch",
            help="Generate an end-to-end simulated ultrasound demonstration without AI weights.",
        )
        if st.session_state.get("result"):
            if st.button("RESET TO STANDBY", width="stretch"):
                st.session_state.pop("result", None)
                st.session_state.pop("robot_sim_result", None)
                st.session_state.pop("robot_sim_ctx", None)
                st.session_state.pop("robot_nav", None)
                st.rerun()

    # Model status alert if real AI weights are missing
    if not model_status["all_available"]:
        st.warning(
            "⚠️ **Real AI Models Not Loaded — Synthetic Simulation Mode Active**\n\n"
            + (f"• **YOLO Weights**: {model_status['yolo_message']}\n\n" if not model_status["yolo_available"] else "")
            + (f"• **LSTM Checkpoint**: {model_status['lstm_message']}\n\n" if not model_status["lstm_available"] else "")
            + "Real AI inference is disabled until model weights are placed in the `models/` folder. "
            "The UltraBot interface, 3D Robot Simulation tab, and synthetic demo mode are fully functional below."
        )

    # Handle synthetic demo trigger
    if run_synthetic_clicked:
        DASH_DIR.mkdir(parents=True, exist_ok=True)
        with st.spinner("Generating synthetic ultrasound simulation…"):
            pipe = generate_synthetic_pipeline_result(
                output_path=DASH_DIR / "synthetic_demo.mp4",
                preview_path=DASH_DIR / "synthetic_demo_preview.png",
                num_frames=60,
            )
            st.session_state["result"] = pipe.to_dict()
            st.session_state.pop("robot_sim_result", None)
            st.session_state.pop("robot_sim_ctx", None)
            st.session_state.pop("robot_nav", None)
            st.rerun()

    # Handle real AI inference trigger
    if run_real_clicked and video_path is not None:
        if not model_status["all_available"]:
            st.error("Cannot run real AI inference: model weights are missing. Run the synthetic demo instead.")
            return

        DASH_DIR.mkdir(parents=True, exist_ok=True)
        run_slot = st.empty()
        with run_slot.container():
            st.markdown(_pipeline_html(running=True), unsafe_allow_html=True)
            progress = st.progress(0, text="Loading models…")
        try:
            models = get_models(YOLO_WEIGHTS, LSTM_CHECKPOINT)

            def _cb(cur, total, det):
                progress.progress(min(1.0, cur / max(total, 1)), text=f"Frame {cur}/{total} · detections {det}")

            pipe = run_pipeline(
                video_path=video_path,
                output_path=DASH_DIR / "dashboard_run.mp4",
                weights_path=YOLO_WEIGHTS,
                lstm_path=LSTM_CHECKPOINT,
                max_frames=max_frames,
                conf_threshold=conf,
                progress_cb=_cb,
                models=models,
            )
            st.session_state["result"] = pipe.to_dict()
            st.session_state.pop("robot_sim_result", None)
            st.session_state.pop("robot_sim_ctx", None)
            st.session_state.pop("robot_nav", None)
            run_slot.empty()
        except Exception as exc:
            run_slot.empty()
            st.error(f"Pipeline failed: {exc}")
            # Do not return here to avoid hiding the UI
            pass

    result = st.session_state.get("result")
    if not result:
        st.markdown(_pipeline_html(running=False), unsafe_allow_html=True)

        st.markdown(
            """
            <div style="background: #101923; border: 1px solid #1f2d3d; border-radius: 8px; padding: 18px; margin-bottom: 20px;">
                <h3 style="color: #7bdff2; margin-top: 0; margin-bottom: 8px;">UltraBot — Autonomous Robotic Ultrasound Biopsy Guidance</h3>
                <p style="color: #c5d0da; font-size: 14px; margin-bottom: 8px;">
                    UltraBot integrates real-time tumor detection, temporal tracking, LSTM motion prediction, 
                    safety-constrained biopsy trajectory optimization, and 3D robotic arm simulation.
                </p>
                <div style="color: #8b949e; font-size: 13px;">
                    💡 <b>Getting Started:</b> Switch to <b>Tab 7: 3D Robot Simulation</b> below to test synthetic robotic control and kinematics anytime, 
                    or click <b>RUN SYNTHETIC SIMULATION DEMO</b> in the sidebar to simulate full end-to-end trajectory planning.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        tabs = st.tabs(
            [
                "Overview",
                "Detection & Tracking",
                "Temporal Prediction",
                "Trajectory Planning",
                "Virtual Biopsy",
                "Robot Simulation",
            ]
        )

        with tabs[0]:
            st.subheader("System Architecture & Pipeline Flow")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("#### 1. Real-Time Detection")
                st.write("YOLO detector identifies lesion bounding boxes and classes (`benign` / `malignant`).")
            with c2:
                st.markdown("#### 2. Temporal Association")
                st.write("Greedy IoU and center-distance tracker forms consistent lesion tracks across frames.")
            with c3:
                st.markdown("#### 3. Motion Prediction")
                st.write("3-layer LSTM forecasts lesion velocity to compensate for respiratory motion.")

            c4, c5, c6 = st.columns(3)
            with c4:
                st.markdown("#### 4. Trajectory Optimization")
                st.write("Evaluates candidate entry-to-target paths, filtering for reachability and safety margin.")
            with c5:
                st.markdown("#### 5. Virtual Biopsy")
                st.write("Simulates virtual needle insertion and alignment against tracked tumor targets.")
            with c6:
                st.markdown("#### 6. 3D Robotic Workspace")
                st.write("Transforms 2D ultrasound coordinates to 3D robot workspace coordinates for manipulator control.")

        with tabs[1]:
            st.subheader("YOLOv8 Lesion Detection & Temporal Tracking")
            st.info("Select a video from the sidebar and click **RUN REAL AI INFERENCE** (when weights are present) or **RUN SYNTHETIC SIMULATION DEMO** to view live detection bounding boxes and tracking IDs.")
            if samples:
                st.write(f"Available sample videos in workspace: {', '.join(f'`{p.name}`' for p in samples)}")

        with tabs[2]:
            st.subheader("LSTM Motion Prediction")
            st.markdown(
                "The temporal prediction module uses a 3-layer LSTM network with hidden dimension 128 "
                "operating on normalized coordinate displacement sequences."
            )

        with tabs[3]:
            st.subheader("Trajectory Planning Workspace (Standby)")
            st.caption("Sample trajectory calculation using baseline tumor target:")
            sample_opt = optimize_biopsy_paths(
                lesion=(480, 340),
                predicted=(495, 335),
                user_target=None,
                safety_margin_px=24.0,
                needle_length_px=400.0,
            )
            sample_plan = plan_biopsy(
                entry=(480, 50),
                target=(480, 340),
                lesion=(480, 340),
                predicted=(495, 335),
            )
            st.plotly_chart(_candidate_path_figure(sample_opt, sample_plan, 320), width="stretch", key="sb_path")

        with tabs[4]:
            st.subheader("Virtual Biopsy Simulation")
            st.caption("Interactive needle reach and safety margin validation are configured per target.")

        with tabs[5]:
            _render_robot_simulation(pipeline_result=None)

        _footer()
        return

    is_synth = result.get("is_synthetic", False)
    if is_synth:
        st.info("🧪 **SYNTHETIC SIMULATION DEMO** — Displaying simulated ultrasound targets and synthetic trajectory. Real AI inference was not run.")
        detector_label = "Synthetic Generator (Demo)"
    else:
        detector_label = "YOLO Detector (Real Inference)"

    metrics = compute_run_metrics(result)
    records = result["records"]
    last = metrics.get("last_detection")
    last_pred = metrics.get("last_prediction")
    table = _records_table(records)

    with st.sidebar:
        _sidebar_label("PLANNING")
        controls = _collect_planning_controls(result, last, last_pred)

    plan, opt, pane, frame_idx = _compute_biopsy_bundle(result, last, last_pred, controls)
    rec = opt["recommended"]
    nav = st.session_state.get("robot_nav")
    sim = st.session_state.get("robot_sim_result")
    if sim and (not nav or sim.get("path_key") != nav.get("path_key")):
        sim = None

    _kpis(result, metrics, last, rec, sim)
    st.markdown(_pipeline_html(result, metrics, rec, sim), unsafe_allow_html=True)

    still = _analysis_overlay(pane, last, last_pred, records, metrics.get("primary_track_id"))
    left, right = st.columns([0.65, 0.35], gap="small")
    with left:
        st.markdown('<div class="cc-panel"><div class="cc-h">ULTRASOUND ANALYSIS</div></div>', unsafe_allow_html=True)
        st.image(cv2.cvtColor(still, cv2.COLOR_BGR2RGB), width="stretch")
        video_file = Path(result["output_video"])
        if video_file.exists():
            with st.expander("Processed video playback"):
                st.video(str(video_file))
                st.download_button("Download processed MP4", data=video_file.read_bytes(), file_name=video_file.name, mime="video/mp4")
    with right:
        _inference_panel(result, metrics, last, last_pred, detector_label)

    p1, p2, p3, p4 = st.columns(4, gap="small")
    with p1:
        st.markdown('<div class="cc-h">TEMPORAL TRACKING</div>', unsafe_allow_html=True)
        st.plotly_chart(_trajectory_figure(records, metrics.get("primary_track_id")), width="stretch", key="ov_traj")
    with p2:
        st.markdown('<div class="cc-h">LSTM PREDICTION</div>', unsafe_allow_html=True)
        st.plotly_chart(_lstm_compare_figure(records), width="stretch", key="ov_lstm")
    with p3:
        st.markdown('<div class="cc-h">TRAJECTORY PLANNING</div>', unsafe_allow_html=True)
        st.plotly_chart(_candidate_path_figure(opt, plan), width="stretch", key="ov_path")
    with p4:
        st.markdown('<div class="cc-h">ROBOT SIMULATION</div>', unsafe_allow_html=True)
        robot_src = nav if nav and nav.get("recommended") is not None else st.session_state.get("robot_sim_ctx")
        if robot_src and robot_src.get("recommended") is not None:
            robot_img = render_robot_frame(
                robot_src["pane"],
                robot_src["recommended"],
                1.0 if sim else 0.0,
                status_label=sim["status_label"] if sim else "PLANNED",
            )
            st.image(cv2.cvtColor(robot_img, cv2.COLOR_BGR2RGB), width="stretch")
        else:
            st.caption("No recommended path.")

    st.markdown('<div class="cc-h">EXPERIMENT RESULTS</div>', unsafe_allow_html=True)
    _experiment_table(result, metrics, rec, sim)

    tabs = st.tabs(
        [
            "Overview",
            "Detection & Tracking",
            "Temporal Prediction",
            "Trajectory Planning",
            "Virtual Biopsy",
            "Robot Simulation",
        ]
    )

    with tabs[0]:
        st.caption("Command-center view above is the run overview. Details are in the other tabs.")
        samples_jpeg = result.get("sample_frames") or []
        if samples_jpeg:
            pick = st.slider("HUD frame strip", 0, len(samples_jpeg) - 1, len(samples_jpeg) - 1)
            item = samples_jpeg[pick]
            st.image(item["jpeg"], caption=f"HUD sample at frame {item['index']}", width="stretch")
        preview = Path(result.get("preview_image") or "")
        if preview.exists():
            st.image(str(preview), caption="Last processed HUD frame", width="stretch")

    with tabs[1]:
        c1, c2 = st.columns(2)
        with c1:
            st.write(f"Source: `{Path(result['source_video']).name}`")
            st.write(f"{result['video_size'][0]}×{result['video_size'][1]} @ {result['src_fps']:.2f} fps")
            st.write(f"Primary track: {metrics['primary_track_id'] if metrics['primary_track_id'] is not None else '—'}")
            st.write(f"Tracked frames: {metrics['tracked_frames']}")
            st.write(f"Misses: {metrics['missed_detections']} · coasting: {metrics['coasting_frames']}")
            st.write(f"LSTM-ready frames: {metrics['lstm_ready_frames']}")
            settings = result.get("tracker_settings") or {}
            if settings:
                st.caption(
                    f"{settings.get('association')} · max center {settings.get('max_center_dist')} px · "
                    f"min IoU {settings.get('min_iou')} · coast {settings.get('max_missed')}"
                )
        with c2:
            if metrics["tracks"]:
                st.dataframe(pd.DataFrame(metrics["tracks"]), width="stretch", hide_index=True)
            if metrics["class_counts"]:
                st.bar_chart(pd.Series(metrics["class_counts"], name="frames"))
            st.caption("Empty lesion coordinates mean a miss — no fake center was inserted.")
        st.plotly_chart(_confidence_figure(records, 300), width="stretch", key="chart_confidence_tracking")
        st.dataframe(table, width="stretch", hide_index=True, height=320)

    with tabs[2]:
        st.plotly_chart(_trajectory_figure(records, metrics.get("primary_track_id"), 380), width="stretch", key="chart_trajectory_main")
        st.plotly_chart(_lstm_compare_figure(records, 320), width="stretch", key="chart_lstm_main")
        if last:
            pred = last_pred.get("pred_center") if last_pred else None
            st.caption(f"Track {last.get('track_id')} · current {_fmt_pt(last.get('raw_center'))} · predicted {_fmt_pt(pred)}")
        st.write(
            f"LSTM checkpoint: sequence {result['lstm_meta']['sequence_length']}, "
            f"hidden {result['lstm_meta']['hidden_size']}, epochs {result['lstm_meta']['epochs']}"
        )
        st.write(f"LSTM one-step error: {_stat_line(metrics['lstm_one_step_error_px'], ' px')}")

    with tabs[3]:
        _render_trajectory_planning(last, last_pred, plan, opt, pane, frame_idx, sim)

    with tabs[4]:
        _render_virtual_biopsy(last, plan, pane, frame_idx)

    with tabs[5]:
        _render_robot_simulation(pipeline_result=result)

    with st.expander("How to read these numbers"):
        for item in metrics["notes"]:
            st.write(f"- {item}")
        st.write("**Not available**")
        for item in metrics["unavailable"]:
            st.write(f"- {item}")

    _footer()


if __name__ == "__main__":
    main()
