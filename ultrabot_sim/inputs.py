"""
Inputs: video / single image / image folder / dashboard plan bundle / demo.

Every source yields (index, pane) where pane is the frame resized to the
990 x 720 pane the UltraBot pipeline works in, and builds the matching
perception object (live YOLO + tracker + LSTM, or a replay of stored
FrameRecords). A source can be rewound (compare mode replays the same
window for both paths).

Auto window: the recorded video is a concatenation of many short patient
clips. A biopsy needs ~25 s of continuous tracking, so with --auto and no
--start-frame the video is pre-scanned ONCE with the same pipeline (cached
in runs/cache/) and playback starts just before the longest lesion track.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np

from ultrabot_sim import config as cfg

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def to_pane(frame):
    return cv2.resize(frame, (cfg.PANE_W_PX, cfg.PANE_H_PX))


class Source:
    mode = "video"
    synthetic = False
    fps = 5.0
    label = ""
    start_frame = 1

    @property
    def frame_dt(self):
        return 1.0 / max(self.fps, 1e-3)

    def rewind(self, start=None):
        raise NotImplementedError

    def next(self):
        raise NotImplementedError

    def make_perception(self, models=None, **kw):
        from ultrabot_sim.perception import Perception
        return Perception(models=models, mode=self.mode, **kw)

    def close(self):
        pass


class VideoSource(Source):
    mode = "video"

    def __init__(self, path, start_frame=1, max_frames=None):
        self.path = str(path)
        if not os.path.isfile(self.path):
            raise FileNotFoundError(f"video not found: {self.path}")
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open video: {self.path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 5.0)
        if not (0.5 < self.fps < 240):
            self.fps = 5.0
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.max_frames = max_frames
        self.label = Path(self.path).name
        self.rewind(start_frame)

    def rewind(self, start=None):
        if start is not None:
            self.start_frame = max(1, int(start))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame - 1)
        self.index = self.start_frame - 1
        self.count = 0

    def next(self):
        if self.max_frames and self.count >= self.max_frames:
            return None
        ok, frame = self.cap.read()
        if not ok:
            return None
        self.index += 1
        self.count += 1
        return self.index, to_pane(frame)

    def close(self):
        self.cap.release()


class FramesSource(Source):
    mode = "frames"

    def __init__(self, folder, fps=5.0, start_frame=1):
        self.files = sorted(str(p) for p in Path(folder).iterdir()
                            if p.suffix.lower() in IMG_EXT)
        if not self.files:
            raise FileNotFoundError(f"no images in {folder}")
        self.fps = float(fps)
        self.label = Path(folder).name + f" ({len(self.files)} frames)"
        self.rewind(start_frame)

    def rewind(self, start=None):
        if start is not None:
            self.start_frame = max(1, int(start))
        self.i = self.start_frame - 1

    def next(self):
        if self.i >= len(self.files):
            return None
        img = cv2.imread(self.files[self.i])
        self.i += 1
        if img is None:
            return self.next()
        return self.i, to_pane(img)


class ImageSource(Source):
    """A single uploaded ultrasound image. YOLO + planning work; the LSTM
    cannot run (it needs 20 consecutive frames) and nothing is faked."""
    mode = "image"

    def __init__(self, path):
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"could not read image: {path}")
        self.pane = to_pane(img)
        self.label = Path(path).name
        self.fps = 5.0
        self.rewind()

    def rewind(self, start=None):
        self.done = False

    def next(self):
        if self.done:
            return None
        self.done = True
        return 1, self.pane.copy()


class ReplaySource(Source):
    """Frames + stored FrameRecords (bundle / synthetic demo)."""

    def __init__(self, records, frames_fn, fps, mode, synthetic=False,
                 label="", lstm_meta=None):
        self.records = sorted(records, key=lambda r: int(r["index"]))
        self.frames_fn = frames_fn
        self.fps = float(fps)
        self.mode = mode
        self.synthetic = synthetic
        self.label = label
        self.lstm_meta = lstm_meta or {}
        self.indices = [int(r["index"]) for r in self.records]
        self.rewind(self.indices[0] if self.indices else 1)

    def rewind(self, start=None):
        if start is not None:
            self.start_frame = int(start)
        self.pos = 0
        while self.pos < len(self.indices) and \
                self.indices[self.pos] < self.start_frame:
            self.pos += 1

    def next(self):
        if self.pos >= len(self.indices):
            return None
        idx = self.indices[self.pos]
        self.pos += 1
        return idx, self.frames_fn(idx)

    def make_perception(self, models=None, **kw):
        from ultrabot_sim.perception import RecordReplay
        seq = int(self.lstm_meta.get("sequence_length", 20))
        return RecordReplay(self.records, seq_len=seq, mode=self.mode,
                            synthetic=self.synthetic)


# ---------------------------------------------------------------- bundle
BUNDLE_KEYS = ("source_video", "planning_frame", "records", "lstm_meta",
               "class_names", "pane_size", "conf_threshold")


def validate_bundle(b):
    missing = [k for k in BUNDLE_KEYS if k not in b]
    if missing:
        raise ValueError("plan bundle is missing keys: " + ", ".join(missing))
    if not isinstance(b["records"], list) or not b["records"]:
        raise ValueError("plan bundle has no FrameRecords")
    if tuple(b["pane_size"]) != (cfg.PANE_W_PX, cfg.PANE_H_PX):
        raise ValueError(f"plan bundle pane {b['pane_size']} != "
                         f"{(cfg.PANE_W_PX, cfg.PANE_H_PX)}")
    for r in b["records"][:5]:
        for k in ("index", "detected", "raw_center", "box_xyxy",
                  "lstm_ready", "track_id"):
            if k not in r:
                raise ValueError(f"FrameRecord without '{k}'")
    return True


def load_bundle(path):
    with open(path, "r", encoding="utf-8") as f:
        b = json.load(f)
    validate_bundle(b)
    vid = b["source_video"]
    cap = cv2.VideoCapture(vid) if vid and os.path.isfile(vid) else None
    if cap is not None and not cap.isOpened():
        cap = None
    fps = float(b.get("src_fps") or (cap.get(cv2.CAP_PROP_FPS) if cap
                                     else 5.0) or 5.0)
    cache = {}

    def frames_fn(idx):
        if cap is None:
            img = np.zeros((cfg.PANE_H_PX, cfg.PANE_W_PX, 3), np.uint8)
            cv2.putText(img, "source video not found - records only",
                        (40, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (200, 200, 200), 2, cv2.LINE_AA)
            return img
        if cache.get("next") != idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx - 1)
        ok, fr = cap.read()
        cache["next"] = idx + 1
        if not ok:
            return np.zeros((cfg.PANE_H_PX, cfg.PANE_W_PX, 3), np.uint8)
        return to_pane(fr)

    src = ReplaySource(b["records"], frames_fn, fps, "bundle",
                       synthetic=bool(b.get("is_synthetic")),
                       label=f"bundle: {Path(vid).name if vid else '-'}",
                       lstm_meta=b.get("lstm_meta"))
    src.bundle = b
    return src


# ---------------------------------------------------------------- demo
def synthetic_frame(rec, w=None, h=None, seed=0):
    """Clean synthetic ultrasound pane for one synthetic FrameRecord (same
    recipe as ultrabot.pipeline.generate_synthetic_pipeline_result, without
    its HUD drawings)."""
    w = w or cfg.PANE_W_PX
    h = h or cfg.PANE_H_PX
    rng = np.random.default_rng(seed + int(rec["index"]))
    frame = np.full((h, w, 3), 16, dtype=np.uint8)
    speckle = rng.integers(0, 45, (h, w, 3), dtype=np.uint8)
    frame = cv2.add(frame, speckle)
    cv2.ellipse(frame, (w // 2, 60), (380, 560), 0, 40, 140, (35, 45, 50),
                -1, cv2.LINE_AA)
    x1, y1, x2, y2 = rec["box_xyxy"]
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    ax, ay = int((x2 - x1) / 2), int((y2 - y1) / 2)
    cv2.ellipse(frame, (cx, cy), (ax, ay), 15, 0, 360, (22, 28, 32), -1,
                cv2.LINE_AA)
    cv2.ellipse(frame, (cx, cy), (ax, ay), 15, 0, 360, (50, 75, 80), 1,
                cv2.LINE_AA)
    return frame


def demo_source(num_frames=400, fps=10.0):
    from ultrabot.pipeline import generate_synthetic_pipeline_result
    cfg.ensure_runs_dir()
    out = cfg.RUNS_DIR / "synthetic_demo.mp4"
    res = generate_synthetic_pipeline_result(output_path=out,
                                             num_frames=num_frames, fps=fps)
    recs = res.records
    by = {int(r["index"]): r for r in recs}
    src = ReplaySource(recs, lambda i: synthetic_frame(by[i]), fps, "demo",
                       synthetic=True, label="SYNTHETIC demo",
                       lstm_meta=res.lstm_meta)
    return src


# ---------------------------------------------------------------- auto window
def _cache_key(video, conf, weights):
    st = os.stat(video)
    wst = os.stat(weights) if weights and os.path.isfile(weights) else None
    raw = f"{os.path.abspath(video)}|{st.st_size}|{int(st.st_mtime)}|{conf}|" \
          f"{wst.st_size if wst else 0}|{int(wst.st_mtime) if wst else 0}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def scan_video(video, models, conf=None, progress=True):
    """Run the perception pipeline over the whole video once (no drawing,
    no output video) and cache the per-frame primary-track summary."""
    from ultrabot.pipeline import get_yolo_weights_path
    from ultrabot_sim.perception import Perception
    conf = cfg.CONF if conf is None else conf
    key = _cache_key(video, conf, str(get_yolo_weights_path()))
    cdir = cfg.RUNS_DIR / "cache"
    cdir.mkdir(parents=True, exist_ok=True)
    cpath = cdir / f"scan_{key}.json"
    if cpath.is_file():
        try:
            return json.loads(cpath.read_text())
        except Exception:                                   # noqa: BLE001
            pass
    src = VideoSource(video)
    per = Perception(models=models, conf=conf, mode="video")
    rows = []
    n = src.n_frames or 0
    while True:
        item = src.next()
        if item is None:
            break
        idx, pane = item
        o = per.step(pane, idx)
        rows.append({"index": idx, "detected": o.detected,
                     "track_id": o.track_id, "lstm_ready": o.lstm_ready})
        if progress and idx % 50 == 0:
            print(f"\r[scan] pre-scanning the video for a stable lesion "
                  f"track: {idx}/{n} frames", end="", flush=True)
    if progress:
        print()
    src.close()
    data = {"video": str(video), "conf": conf, "rows": rows}
    cpath.write_text(json.dumps(data))
    return data


def best_window(rows, max_gap=None):
    """Longest run of frames on one primary track (misses <= max_gap).
    Returns (start_frame, end_frame, track_id, hits)."""
    gap = cfg.TRACK_MAX_MISSED if max_gap is None else max_gap
    best = None
    cur = None
    for r in rows:
        if not r["detected"]:
            if cur is not None:
                cur["miss"] += 1
                if cur["miss"] > gap:
                    best = _better(best, cur)
                    cur = None
            continue
        tid = r["track_id"]
        if cur is not None and tid == cur["tid"]:
            cur["end"] = r["index"]
            cur["hits"] += 1
            cur["miss"] = 0
        else:
            if cur is not None:
                best = _better(best, cur)
            cur = {"tid": tid, "start": r["index"], "end": r["index"],
                   "hits": 1, "miss": 0}
    if cur is not None:
        best = _better(best, cur)
    if best is None:
        return None
    return best["start"], best["end"], best["tid"], best["hits"]


def _better(a, b):
    if a is None:
        return b
    return b if b["hits"] > a["hits"] else a


def make_source(args):
    if args.demo:
        return demo_source()
    if args.bundle:
        return load_bundle(args.bundle)
    if args.image:
        return ImageSource(args.image)
    if args.frames:
        return FramesSource(args.frames, fps=args.fps or 5.0,
                            start_frame=args.start_frame or 1)
    return VideoSource(args.video or cfg.DEFAULT_VIDEO,
                       start_frame=args.start_frame or 1,
                       max_frames=args.max_frames)
