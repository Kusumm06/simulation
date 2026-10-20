"""Associate YOLO boxes across frames using IoU and nearest-center distance."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


def bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def center_distance(p, q) -> float:
    return math.hypot(float(p[0]) - float(q[0]), float(p[1]) - float(q[1]))


@dataclass
class Detection:
    box: tuple
    center: tuple
    conf: float
    class_id: Optional[int] = None
    class_name: Optional[str] = None


@dataclass
class Track:
    track_id: int
    last_frame: int
    last_box: tuple
    last_center: tuple
    last_conf: float
    last_class_id: Optional[int]
    last_class_name: Optional[str]
    hits: int = 1
    misses: int = 0
    miss_streak: int = 0
    consecutive_hits: int = 1
    first_frame: int = 0
    state: str = "active"
    history: list = field(default_factory=list)

    def real_centers(self) -> list[tuple]:
        return [h["center"] for h in self.history]


@dataclass
class MatchResult:
    track: Track
    detection: Optional[Detection]
    iou: Optional[float]
    distance: Optional[float]


class CenterIoUTracker:
    """Greedy multi-object tracker. Positions are never invented."""

    def __init__(self, max_missed=8, max_center_dist=90.0, min_iou=0.10):
        self.max_missed = int(max_missed)
        self.max_center_dist = float(max_center_dist)
        self.min_iou = float(min_iou)
        self._next_id = 1
        self.tracks: list[Track] = []

    def update(self, frame_idx: int, detections: list[Detection]) -> list[MatchResult]:
        live = [t for t in self.tracks if t.state in ("active", "coasting")]
        pairs = []
        for track in live:
            for det in detections:
                iou = bbox_iou(track.last_box, det.box)
                dist = center_distance(track.last_center, det.center)
                if iou >= self.min_iou or dist <= self.max_center_dist:
                    cost = 0.55 * (1.0 - iou) + 0.45 * min(dist / max(self.max_center_dist, 1e-6), 2.5)
                    pairs.append((cost, track.track_id, id(det), iou, dist, track, det))
        pairs.sort(key=lambda x: x[0])

        used_tracks = set()
        used_dets = set()
        results: list[MatchResult] = []

        for _cost, tid, did, iou, dist, track, det in pairs:
            if tid in used_tracks or did in used_dets:
                continue
            self._hit(track, frame_idx, det)
            used_tracks.add(tid)
            used_dets.add(did)
            results.append(MatchResult(track=track, detection=det, iou=float(iou), distance=float(dist)))

        for det in detections:
            if id(det) in used_dets:
                continue
            track = self._start(frame_idx, det)
            results.append(MatchResult(track=track, detection=det, iou=None, distance=None))

        for track in live:
            if track.track_id in used_tracks:
                continue
            track.miss_streak += 1
            track.misses += 1
            track.consecutive_hits = 0
            if track.miss_streak > self.max_missed:
                track.state = "lost"
            else:
                track.state = "coasting"
            results.append(MatchResult(track=track, detection=None, iou=None, distance=None))

        return results

    def primary_track(self) -> Optional[Track]:
        active = [t for t in self.tracks if t.state == "active"]
        candidates = active or [t for t in self.tracks if t.state == "coasting"]
        if not candidates:
            return None
        candidates.sort(key=lambda t: (t.hits, t.last_frame), reverse=True)
        return candidates[0]

    def summary(self) -> list[dict]:
        rows = []
        for t in self.tracks:
            rows.append(
                {
                    "track_id": t.track_id,
                    "state": t.state,
                    "hits": t.hits,
                    "misses": t.misses,
                    "first_frame": t.first_frame,
                    "last_frame": t.last_frame,
                    "consecutive_hits": t.consecutive_hits,
                }
            )
        return rows

    def _start(self, frame_idx: int, det: Detection) -> Track:
        track = Track(
            track_id=self._next_id,
            last_frame=frame_idx,
            last_box=det.box,
            last_center=det.center,
            last_conf=det.conf,
            last_class_id=det.class_id,
            last_class_name=det.class_name,
            first_frame=frame_idx,
        )
        track.history.append(
            {
                "frame": frame_idx,
                "center": det.center,
                "box": det.box,
                "conf": det.conf,
                "class_id": det.class_id,
                "class_name": det.class_name,
            }
        )
        self._next_id += 1
        self.tracks.append(track)
        return track

    def _hit(self, track: Track, frame_idx: int, det: Detection) -> None:
        track.last_frame = frame_idx
        track.last_box = det.box
        track.last_center = det.center
        track.last_conf = det.conf
        track.last_class_id = det.class_id
        track.last_class_name = det.class_name
        track.hits += 1
        track.miss_streak = 0
        track.consecutive_hits += 1
        track.state = "active"
        track.history.append(
            {
                "frame": frame_idx,
                "center": det.center,
                "box": det.box,
                "conf": det.conf,
                "class_id": det.class_id,
                "class_name": det.class_name,
            }
        )
