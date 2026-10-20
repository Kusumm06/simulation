"""YOLOv8 detection + real temporal tracking + LSTM, reused by the CLI and dashboard."""

from __future__ import annotations

import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from ultrabot.render import FPS_OUT, OUT_H, OUT_W, VIDEO_W, annotate_frame, build_sidebar, clamp
from ultrabot.tracker import CenterIoUTracker, Detection


def get_project_root() -> Path:
    """Resolve project root reliably regardless of current working directory.

    Checks:
    1. ULTRABOT_ROOT environment variable (if set and valid).
    2. Repository root containing ultrabot/ and app.py based on this file's location.
    3. Current working directory if app.py exists there.
    """
    env_root = os.environ.get("ULTRABOT_ROOT")
    if env_root:
        p = Path(env_root).resolve()
        if p.exists():
            return p
    pkg_root = Path(__file__).resolve().parents[1]
    if (pkg_root / "app.py").exists() or (pkg_root / "ultrabot").exists():
        return pkg_root
    cwd = Path.cwd().resolve()
    if (cwd / "app.py").exists() or (cwd / "ultrabot").exists():
        return cwd
    return pkg_root


def get_yolo_weights_path(custom_path: Optional[str | Path] = None) -> Path:
    """Return configured or default YOLO weights path."""
    if custom_path:
        return Path(custom_path).resolve()
    env_path = os.environ.get("ULTRABOT_YOLO_WEIGHTS")
    if env_path:
        return Path(env_path).resolve()
    return (get_project_root() / "models" / "best.pt").resolve()


def get_lstm_checkpoint_path(custom_path: Optional[str | Path] = None) -> Path:
    """Return configured or default LSTM checkpoint path."""
    if custom_path:
        return Path(custom_path).resolve()
    env_path = os.environ.get("ULTRABOT_LSTM_CHECKPOINT")
    if env_path:
        return Path(env_path).resolve()
    return (get_project_root() / "models" / "lstm_checkpoint.pth").resolve()


def validate_yolo_weights(weights_path: Optional[str | Path] = None) -> Tuple[bool, str]:
    """Validate YOLO weights path independently with explicit messaging."""
    path = get_yolo_weights_path(weights_path)
    if not path.exists():
        return False, (
            f"YOLO weights not found: {path}. "
            "Breast lesion detection requires trained weights file (e.g. best.pt) "
            "compatible with classes in data.yaml (benign, malignant)."
        )
    return True, f"YOLO weights found: {path}"


def validate_lstm_checkpoint(lstm_path: Optional[str | Path] = None) -> Tuple[bool, str]:
    """Validate LSTM checkpoint path independently with explicit messaging."""
    path = get_lstm_checkpoint_path(lstm_path)
    if not path.exists():
        return False, (
            f"LSTM checkpoint not found: {path}. "
            "Temporal trajectory prediction requires trained checkpoint file (e.g. lstm_checkpoint.pth)."
        )
    return True, f"LSTM checkpoint found: {path}"


def check_model_availability(
    weights_path: Optional[str | Path] = None,
    lstm_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Inspect availability of YOLO detector weights and LSTM checkpoint separately."""
    y_ok, y_msg = validate_yolo_weights(weights_path)
    l_ok, l_msg = validate_lstm_checkpoint(lstm_path)
    return {
        "yolo_path": str(get_yolo_weights_path(weights_path)),
        "yolo_available": y_ok,
        "yolo_message": y_msg,
        "lstm_path": str(get_lstm_checkpoint_path(lstm_path)),
        "lstm_available": l_ok,
        "lstm_message": l_msg,
        "all_available": y_ok and l_ok,
    }


ROOT = get_project_root()
YOLO_WEIGHTS = get_yolo_weights_path()
LSTM_CHECKPOINT = get_lstm_checkpoint_path()
DEFAULT_VIDEO = ROOT / "outputs" / "input.mp4"
DEFAULT_OUTPUT = ROOT / "outputs" / "ultrabot_live.mp4"

ProgressFn = Optional[Callable[[int, int, int], None]]


class LSTMModel(torch.nn.Module):
    """Architecture from the existing notebook / demo checkpoint."""

    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=2,
            hidden_size=128,
            num_layers=3,
            batch_first=True,
            dropout=0.2,
        )
        self.fc1 = torch.nn.Linear(128, 64)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(64, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc2(self.relu(self.fc1(out)))


def predict_next(model, sequence):
    seq = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        pred_velocity = model(seq)
    return pred_velocity.numpy()[0]


@dataclass
class FrameRecord:
    index: int
    detected: bool
    conf: Optional[float] = None
    class_id: Optional[int] = None
    class_name: Optional[str] = None
    box_xyxy: Optional[tuple] = None
    raw_center: Optional[tuple] = None
    display_center: Optional[tuple] = None
    pred_center: Optional[tuple] = None
    pred_velocity: Optional[tuple] = None
    needle_tip: Optional[tuple] = None
    lstm_ready: bool = False
    track_id: Optional[int] = None
    track_state: Optional[str] = None
    consecutive_hits: int = 0
    match_iou: Optional[float] = None
    match_distance: Optional[float] = None
    detections_this_frame: int = 0


@dataclass
class PipelineResult:
    output_video: str
    preview_image: str
    records: list
    class_names: dict
    video_size: tuple
    src_fps: float
    frames_processed: int
    detections: int
    lstm_meta: dict
    conf_threshold: float
    weights_path: str
    lstm_path: str
    source_video: str
    elapsed_sec: float
    demo_jitter: bool
    sample_frames: list = field(default_factory=list)
    tracks: list = field(default_factory=list)
    primary_track_id: Optional[int] = None
    tracker_settings: dict = field(default_factory=dict)
    is_synthetic: bool = False

    def to_dict(self):
        return asdict(self)


def load_models(weights_path=None, lstm_path=None):
    """Load YOLO and LSTM models, validating each checkpoint separately with clear error messages."""
    w_path = get_yolo_weights_path(weights_path)
    l_path = get_lstm_checkpoint_path(lstm_path)

    y_ok, y_msg = validate_yolo_weights(w_path)
    if not y_ok:
        raise FileNotFoundError(y_msg)

    l_ok, l_msg = validate_lstm_checkpoint(l_path)
    if not l_ok:
        raise FileNotFoundError(l_msg)

    model_yolo = YOLO(str(w_path))
    checkpoint = torch.load(l_path, map_location="cpu", weights_only=False)
    model_lstm = LSTMModel()
    model_lstm.load_state_dict(checkpoint["model_state_dict"])
    model_lstm.eval()
    lstm_meta = {
        "sequence_length": int(checkpoint.get("sequence_length", 20)),
        "hidden_size": int(checkpoint.get("hidden_size", 128)),
        "epochs": int(checkpoint.get("epochs", 0)),
    }
    names = {int(k): str(v) for k, v in getattr(model_yolo, "names", {}).items()}
    return model_yolo, model_lstm, lstm_meta, names


def safe_load_models(weights_path=None, lstm_path=None) -> Tuple[Optional[tuple], dict[str, Any]]:
    """Attempt to load models safely; returns (models_tuple, availability_status).

    Does not raise FileNotFoundError if weights are missing, allowing dashboards
    to inspect status and degrade gracefully.
    """
    status = check_model_availability(weights_path, lstm_path)
    if not status["all_available"]:
        return None, status
    try:
        models = load_models(weights_path, lstm_path)
        return models, status
    except Exception as exc:
        status["all_available"] = False
        status["error"] = str(exc)
        return None, status


def _open_writer(path: Path, fps: float, size: tuple):
    path.parent.mkdir(parents=True, exist_ok=True)
    for code in ("mp4v", "XVID"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*code), fps, size)
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f"Could not open video writer for {path}")


def _yolo_detections(results, class_names) -> list[Detection]:
    dets = []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else None
        clss = r.boxes.cls.cpu().numpy() if r.boxes.cls is not None else None
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = [float(v) for v in xyxy[i].tolist()]
            conf = float(confs[i]) if confs is not None else 0.0
            class_id = int(clss[i]) if clss is not None else None
            dets.append(
                Detection(
                    box=(x1, y1, x2, y2),
                    center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                    conf=conf,
                    class_id=class_id,
                    class_name=class_names.get(class_id) if class_id is not None else None,
                )
            )
    return dets


def run_pipeline(
    video_path,
    output_path=None,
    weights_path=YOLO_WEIGHTS,
    lstm_path=LSTM_CHECKPOINT,
    max_frames=120,
    conf_threshold=0.10,
    demo_jitter=False,
    progress_cb: ProgressFn = None,
    models=None,
    max_missed=8,
    max_center_dist=90.0,
    min_iou=0.10,
):
    """Run YOLO detection, associate real boxes over time, then LSTM on consecutive centers.

    demo_jitter is accepted for API compatibility and is ignored. Positions are never fabricated.
    """
    video_path = Path(video_path)
    if output_path is None:
        output_path = DEFAULT_OUTPUT
    output_path = Path(output_path)
    preview_path = output_path.with_name(output_path.stem + "_preview.png")

    if models is None:
        model_yolo, model_lstm, lstm_meta, class_names = load_models(weights_path, lstm_path)
    else:
        model_yolo, model_lstm, lstm_meta, class_names = models

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or FPS_OUT)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if max_frames and max_frames > 0 and total > 0:
        planned = min(max_frames, total)
    elif max_frames and max_frames > 0:
        planned = max_frames
    else:
        planned = total

    writer = _open_writer(output_path, FPS_OUT, (OUT_W, OUT_H))
    seq_len = int(lstm_meta.get("sequence_length", 20))
    tracker = CenterIoUTracker(max_missed=max_missed, max_center_dist=max_center_dist, min_iou=min_iou)
    needle_x = float(VIDEO_W // 2)
    needle_y = float(OUT_H - 40)
    records: list[FrameRecord] = []
    sample_frames = []
    last_hud = None
    detections = 0
    frame_count = 0
    prev_time = time.time()
    t0 = time.time()
    sample_every = max(1, planned // 12) if planned else 5

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and max_frames > 0 and frame_count >= max_frames:
            break

        video = cv2.resize(frame, (VIDEO_W, OUT_H))
        frame_count += 1
        now = time.time()
        fps = 1.0 / max(1e-6, now - prev_time)
        prev_time = now
        elapsed = int(frame_count / FPS_OUT)
        vh, vw = video.shape[:2]

        results = model_yolo(video, conf=conf_threshold, verbose=False)
        frame_dets = _yolo_detections(results, class_names)
        matches = tracker.update(frame_count, frame_dets)
        primary = tracker.primary_track()
        primary_match = None
        if primary is not None:
            for item in matches:
                if item.track.track_id == primary.track_id:
                    primary_match = item
                    break

        matched = primary_match.detection if primary_match is not None else None
        tumor_found = matched is not None
        if tumor_found:
            detections += 1
        raw_cx = raw_cy = None
        box_xyxy = None
        conf_val = None
        class_id = None
        class_name = None
        match_iou = primary_match.iou if primary_match is not None else None
        match_dist = primary_match.distance if primary_match is not None else None
        if tumor_found:
            raw_cx, raw_cy = matched.center
            box_xyxy = matched.box
            conf_val = matched.conf
            class_id = matched.class_id
            class_name = matched.class_name

        lstm_ready = False
        pred_velocity = None
        pred_pt = None
        tumor_pt = None
        needle_pt = None
        if tumor_found and primary is not None and primary.consecutive_hits >= seq_len:
            run = primary.history[-seq_len:]
            frames = [h["frame"] for h in run]
            consecutive = all(frames[i] == frames[i - 1] + 1 for i in range(1, len(frames)))
            if consecutive:
                sequence = [[h["center"][0] / vw, h["center"][1] / vh] for h in run]
                pred_vx, pred_vy = predict_next(model_lstm, sequence)
                pred_velocity = (float(pred_vx), float(pred_vy))
                pred_x = clamp((raw_cx / vw + float(pred_vx)) * vw, 0, vw - 1)
                pred_y = clamp((raw_cy / vh + float(pred_vy)) * vh, 0, vh - 1)
                pred_pt = (int(pred_x), int(pred_y))
                lstm_ready = True
                needle_x += (pred_x - needle_x) * 0.10
                needle_y += (pred_y - needle_y) * 0.10
                needle_pt = (int(needle_x), int(needle_y))

        if tumor_found:
            tumor_pt = (int(raw_cx), int(raw_cy))

        trajectory = []
        if primary is not None:
            trajectory = [(int(h["center"][0]), int(h["center"][1])) for h in primary.history[-40:]]

        hud = annotate_frame(
            video,
            fps,
            elapsed,
            tumor_found,
            tumor_pt,
            pred_pt,
            needle_pt,
            trajectory,
            track_id=primary.track_id if primary is not None else None,
            track_status=primary.state if primary is not None else None,
        )
        writer.write(hud)
        last_hud = hud
        if frame_count == 1 or frame_count % sample_every == 0 or tumor_found:
            if len(sample_frames) < 16 or tumor_found:
                rgb = cv2.cvtColor(hud, cv2.COLOR_BGR2RGB)
                if tumor_found or len(sample_frames) < 12:
                    sample_frames.append({"index": frame_count, "image": rgb})
                    if len(sample_frames) > 16:
                        sample_frames.pop(0)

        records.append(
            FrameRecord(
                index=frame_count,
                detected=tumor_found,
                conf=conf_val,
                class_id=class_id,
                class_name=class_name,
                box_xyxy=box_xyxy,
                raw_center=(float(raw_cx), float(raw_cy)) if tumor_found else None,
                display_center=(float(raw_cx), float(raw_cy)) if tumor_found else None,
                pred_center=(float(pred_pt[0]), float(pred_pt[1])) if pred_pt else None,
                pred_velocity=pred_velocity,
                needle_tip=(float(needle_x), float(needle_y)) if needle_pt else None,
                lstm_ready=lstm_ready,
                track_id=primary.track_id if primary is not None else None,
                track_state=primary.state if primary is not None else None,
                consecutive_hits=primary.consecutive_hits if primary is not None else 0,
                match_iou=match_iou,
                match_distance=match_dist,
                detections_this_frame=len(frame_dets),
            )
        )
        if progress_cb is not None:
            progress_cb(frame_count, planned or frame_count, detections)

    cap.release()
    writer.release()

    if last_hud is not None:
        cv2.imwrite(str(preview_path), last_hud)

    compact_samples = []
    for item in sample_frames:
        bgr = cv2.cvtColor(item["image"], cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            compact_samples.append({"index": item["index"], "jpeg": buf.tobytes()})

    primary = tracker.primary_track()
    return PipelineResult(
        output_video=str(output_path),
        preview_image=str(preview_path) if preview_path.exists() else "",
        records=[asdict(r) for r in records],
        class_names=class_names,
        video_size=(src_w, src_h),
        src_fps=src_fps,
        frames_processed=frame_count,
        detections=detections,
        lstm_meta=lstm_meta,
        conf_threshold=float(conf_threshold),
        weights_path=str(Path(weights_path)),
        lstm_path=str(Path(lstm_path)),
        source_video=str(video_path),
        elapsed_sec=time.time() - t0,
        demo_jitter=False,
        sample_frames=compact_samples,
        tracks=tracker.summary(),
        primary_track_id=primary.track_id if primary is not None else None,
        tracker_settings={
            "max_missed": max_missed,
            "max_center_dist": max_center_dist,
            "min_iou": min_iou,
            "association": "greedy IoU + nearest-center",
        },
        is_synthetic=False,
    )


def generate_synthetic_pipeline_result(
    output_path: Optional[str | Path] = None,
    preview_path: Optional[str | Path] = None,
    num_frames: int = 60,
    fps: float = 10.0,
) -> PipelineResult:
    """Generate a realistic synthetic ultrasound simulation run without AI model weights.

    Creates simulated lesion coordinates, tracker history, LSTM trajectory predictions,
    and a synthetic video + preview HUD. Clearly flagged with is_synthetic=True.
    """
    if output_path is None:
        output_path = ROOT / "outputs" / "dashboard" / "synthetic_demo.mp4"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if preview_path is None:
        preview_path = output_path.with_name(output_path.stem + "_preview.png")
    preview_path = Path(preview_path)

    vw, vh = VIDEO_W, OUT_H
    seq_len = 20
    records: list[FrameRecord] = []
    sample_frames = []

    # Synthetic lesion motion: gentle respiratory / probe drift centered around (495, 360)
    base_x, base_y = 495.0, 360.0
    w_box, h_box = 100.0, 80.0

    writer = None
    try:
        writer = _open_writer(output_path, fps, (OUT_W, OUT_H))
    except Exception:
        pass

    last_hud = None
    t0 = time.time()
    needle_x = float(vw // 2)
    needle_y = float(vh - 40)
    trajectory = []

    for idx in range(1, num_frames + 1):
        t = idx / fps
        cx = base_x + 35.0 * math.sin(0.8 * t) + 10.0 * math.cos(1.6 * t)
        cy = base_y + 20.0 * math.cos(0.8 * t) + 6.0 * math.sin(1.4 * t)

        x1 = clamp(cx - w_box / 2, 0, vw - 1)
        y1 = clamp(cy - h_box / 2, 0, vh - 1)
        x2 = clamp(cx + w_box / 2, 0, vw - 1)
        y2 = clamp(cy + h_box / 2, 0, vh - 1)
        box = (float(x1), float(y1), float(x2), float(y2))

        lstm_ready = idx >= seq_len
        pred_center = None
        pred_velocity = None
        if lstm_ready:
            vx = 35.0 * 0.8 * (1.0 / fps) * math.cos(0.8 * t) - 10.0 * 1.6 * (1.0 / fps) * math.sin(1.6 * t)
            vy = -20.0 * 0.8 * (1.0 / fps) * math.sin(0.8 * t) + 6.0 * 1.4 * (1.0 / fps) * math.cos(1.4 * t)
            pred_velocity = (float(vx), float(vy))
            pred_center = (round(cx + vx * 1.5, 1), round(cy + vy * 1.5, 1))
            needle_x += (pred_center[0] - needle_x) * 0.10
            needle_y += (pred_center[1] - needle_y) * 0.10
        needle_pt = (int(needle_x), int(needle_y))

        rec = FrameRecord(
            index=idx,
            detected=True,
            conf=round(0.85 + 0.08 * math.sin(t * 1.2), 3),
            class_id=0,
            class_name="benign",
            box_xyxy=box,
            raw_center=(round(cx, 1), round(cy, 1)),
            display_center=(round(cx, 1), round(cy, 1)),
            pred_center=pred_center,
            pred_velocity=pred_velocity,
            needle_tip=needle_pt if lstm_ready else None,
            lstm_ready=lstm_ready,
            track_id=1,
            track_state="ACTIVE",
            consecutive_hits=idx,
            match_iou=round(0.88 + 0.05 * math.cos(t), 3),
            match_distance=round(3.0 + 2.0 * math.sin(t), 1),
            detections_this_frame=1,
        )
        records.append(rec)

        frame = np.full((vh, vw, 3), 16, dtype=np.uint8)
        speckle = np.random.randint(0, 45, (vh, vw, 3), dtype=np.uint8)
        frame = cv2.add(frame, speckle)
        cv2.ellipse(frame, (vw // 2, 60), (380, 560), 0, 40, 140, (35, 45, 50), -1, cv2.LINE_AA)
        cv2.ellipse(frame, (int(cx), int(cy)), (int(w_box // 2), int(h_box // 2)), 15, 0, 360, (22, 28, 32), -1, cv2.LINE_AA)
        cv2.ellipse(frame, (int(cx), int(cy)), (int(w_box // 2), int(h_box // 2)), 15, 0, 360, (50, 75, 80), 1, cv2.LINE_AA)

        trajectory.append((int(cx), int(cy)))
        if len(trajectory) > 25:
            trajectory.pop(0)

        pred_pt_int = (int(pred_center[0]), int(pred_center[1])) if pred_center else None
        annotate_frame(
            frame,
            fps=fps,
            elapsed=int(t),
            tumor_found=True,
            tumor_pt=(int(cx), int(cy)),
            pred_pt=pred_pt_int,
            needle_pt=needle_pt if lstm_ready else None,
            trajectory=list(trajectory),
            track_id=1,
            track_status="SYNTHETIC",
        )
        sidebar = build_sidebar(
            fps=fps,
            tumor_pt=(int(cx), int(cy)),
            pred_pt=pred_pt_int,
            needle_pt=needle_pt if lstm_ready else None,
            elapsed_s=int(t),
            tumor_found=True,
        )
        hud = np.hstack([frame, sidebar])
        last_hud = hud

        if writer is not None:
            writer.write(hud)

        if idx % max(1, num_frames // 10) == 0:
            sample_frames.append({"index": idx, "image": cv2.cvtColor(hud, cv2.COLOR_BGR2RGB)})

    if writer is not None:
        writer.release()

    if last_hud is not None:
        cv2.imwrite(str(preview_path), last_hud)

    compact_samples = []
    for item in sample_frames:
        bgr = cv2.cvtColor(item["image"], cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            compact_samples.append({"index": item["index"], "jpeg": buf.tobytes()})

    track_summary = [{
        "track_id": 1,
        "first_frame": 1,
        "last_frame": num_frames,
        "hits": num_frames,
        "missed": 0,
        "max_streak": num_frames,
        "state": "ACTIVE",
    }]

    return PipelineResult(
        output_video=str(output_path) if output_path.exists() else "",
        preview_image=str(preview_path) if preview_path.exists() else "",
        records=[asdict(r) for r in records],
        class_names={0: "benign", 1: "malignant"},
        video_size=(vw, vh),
        src_fps=fps,
        frames_processed=num_frames,
        detections=num_frames,
        lstm_meta={"sequence_length": seq_len, "hidden_size": 128, "epochs": 50, "mode": "synthetic_generator"},
        conf_threshold=0.10,
        weights_path="None (Synthetic Demo Mode)",
        lstm_path="None (Synthetic Demo Mode)",
        source_video=str(output_path),
        elapsed_sec=time.time() - t0,
        demo_jitter=False,
        sample_frames=compact_samples,
        tracks=track_summary,
        primary_track_id=1,
        tracker_settings={
            "max_missed": 8,
            "max_center_dist": 90.0,
            "min_iou": 0.10,
            "association": "synthetic-generator",
        },
        is_synthetic=True,
    )
