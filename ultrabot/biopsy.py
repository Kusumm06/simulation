"""Virtual 2D biopsy planning from real image coordinates. Simulation only — research prototype."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ultrabot.render import OUT_H, VIDEO_W, clamp


DISCLAIMER = "Virtual biopsy trajectory simulation — not clinical or robotic control."


@dataclass
class BiopsyPlan:
    entry: tuple
    target: tuple
    lesion: Optional[tuple]
    predicted: Optional[tuple]
    needle_tip: tuple
    safety_margin_px: float
    needle_length_px: float
    trajectory_length_px: float
    needle_angle_deg: float
    tip_to_target_px: float
    target_error_px: Optional[float]
    alignment_ok: Optional[bool]
    reach_ok: bool
    success: bool
    status_label: str
    status_reason: str
    frame_index: Optional[int] = None
    track_id: Optional[int] = None

    def to_dict(self):
        return asdict(self)


def _as_xy(pt) -> Optional[tuple[float, float]]:
    if not pt or len(pt) < 2 or pt[0] is None or pt[1] is None:
        return None
    return (float(pt[0]), float(pt[1]))


def _dist(a, b) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def plan_biopsy(
    entry,
    target,
    lesion=None,
    predicted=None,
    safety_margin_px=24.0,
    needle_length_px=400.0,
    frame_size=(VIDEO_W, OUT_H),
    frame_index=None,
    track_id=None,
) -> BiopsyPlan:
    """Compute a 2D plan from the given pixel coordinates. Nothing is invented."""
    vw, vh = frame_size
    entry_xy = (
        clamp(float(entry[0]), 0, vw - 1),
        clamp(float(entry[1]), 0, vh - 1),
    )
    target_xy = (
        clamp(float(target[0]), 0, vw - 1),
        clamp(float(target[1]), 0, vh - 1),
    )
    lesion_xy = _as_xy(lesion)
    predicted_xy = _as_xy(predicted)
    safety = max(0.0, float(safety_margin_px))
    needle_len = max(0.0, float(needle_length_px))

    dx = target_xy[0] - entry_xy[0]
    dy = target_xy[1] - entry_xy[1]
    path_len = math.hypot(dx, dy)
    angle = math.degrees(math.atan2(dy, dx)) if path_len > 1e-6 else 0.0

    if path_len < 1e-6:
        tip = entry_xy
    else:
        travel = min(needle_len, path_len)
        tip = (entry_xy[0] + dx / path_len * travel, entry_xy[1] + dy / path_len * travel)

    tip_to_target = _dist(tip, target_xy)
    reach_ok = needle_len + 1e-6 >= path_len

    target_error = _dist(target_xy, lesion_xy) if lesion_xy else None
    alignment_ok = None if target_error is None else target_error <= safety

    if lesion_xy is None:
        success = False
        label = "NO LESION"
        reason = "No tracked lesion center is available to score target error."
    elif not alignment_ok and not reach_ok:
        success = False
        label = "FAILURE"
        reason = (
            f"Target is {target_error:.1f} px from the lesion (safety {safety:.1f} px) "
            f"and the needle is shorter than the {path_len:.1f} px path."
        )
    elif not alignment_ok:
        success = False
        label = "FAILURE"
        reason = f"Target error {target_error:.1f} px exceeds the {safety:.1f} px safety margin."
    elif not reach_ok:
        success = False
        label = "FAILURE"
        reason = f"Needle length {needle_len:.1f} px does not reach the {path_len:.1f} px trajectory."
    else:
        success = True
        label = "SUCCESS"
        reason = (
            f"Target is {target_error:.1f} px from the lesion (within {safety:.1f} px) "
            f"and the needle reaches the target."
        )

    return BiopsyPlan(
        entry=entry_xy,
        target=target_xy,
        lesion=lesion_xy,
        predicted=predicted_xy,
        needle_tip=tip,
        safety_margin_px=safety,
        needle_length_px=needle_len,
        trajectory_length_px=path_len,
        needle_angle_deg=angle,
        tip_to_target_px=tip_to_target,
        target_error_px=target_error,
        alignment_ok=alignment_ok,
        reach_ok=reach_ok,
        success=success,
        status_label=label,
        status_reason=reason,
        frame_index=frame_index,
        track_id=track_id,
    )


def load_ultrasound_pane(video_path, frame_index, size=(VIDEO_W, OUT_H)):
    """Load one source frame and resize it into the tracker coordinate pane."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    idx = max(0, int(frame_index) - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame = None
        for _ in range(idx + 1):
            ok, frame = cap.read()
            if not ok:
                break
    cap.release()
    if frame is None:
        return None
    return cv2.resize(frame, (int(size[0]), int(size[1])))


def render_biopsy_plan(frame_bgr, plan: BiopsyPlan):
    """Draw a clean academic overlay on an ultrasound pane (BGR)."""
    img = frame_bgr.copy()
    h, w = img.shape[:2]
    overlay = img.copy()

    status_color = (70, 190, 90) if plan.success else (60, 60, 220)
    if plan.status_label == "NO LESION":
        status_color = (40, 170, 230)

    if plan.lesion is not None and plan.safety_margin_px > 0:
        cv2.circle(
            overlay,
            (int(plan.lesion[0]), int(plan.lesion[1])),
            int(plan.safety_margin_px),
            (80, 200, 220),
            -1,
            cv2.LINE_AA,
        )
        img = cv2.addWeighted(overlay, 0.18, img, 0.82, 0)
        cv2.circle(
            img,
            (int(plan.lesion[0]), int(plan.lesion[1])),
            int(plan.safety_margin_px),
            (80, 200, 220),
            2,
            cv2.LINE_AA,
        )

    ex, ey = int(plan.entry[0]), int(plan.entry[1])
    tx, ty = int(plan.target[0]), int(plan.target[1])
    nx, ny = int(plan.needle_tip[0]), int(plan.needle_tip[1])

    if plan.trajectory_length_px > 1:
        cv2.line(img, (ex, ey), (tx, ty), (90, 100, 110), 1, cv2.LINE_AA)
    cv2.line(img, (ex, ey), (nx, ny), (230, 230, 235), 3, cv2.LINE_AA)
    if not plan.reach_ok:
        cv2.line(img, (nx, ny), (tx, ty), (80, 80, 210), 2, cv2.LINE_AA)

    if plan.lesion is not None:
        lx, ly = int(plan.lesion[0]), int(plan.lesion[1])
        cv2.circle(img, (lx, ly), 7, (70, 210, 90), -1, cv2.LINE_AA)
        cv2.circle(img, (lx, ly), 12, (70, 210, 90), 2, cv2.LINE_AA)
        _label(img, "Lesion", (lx + 16, ly + 6), (70, 210, 90))

    show_pred = plan.predicted is not None and (
        plan.lesion is None or _dist(plan.predicted, plan.lesion) >= 8
    )
    if show_pred:
        px, py = int(plan.predicted[0]), int(plan.predicted[1])
        cv2.drawMarker(img, (px, py), (50, 50, 220), cv2.MARKER_TILTED_CROSS, 16, 2, cv2.LINE_AA)
        _label(img, "Predicted", (px + 14, py - 12), (50, 50, 220))

    cv2.circle(img, (tx, ty), 6, (40, 180, 255), -1, cv2.LINE_AA)
    cv2.drawMarker(img, (tx, ty), (40, 180, 255), cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
    _label(img, "Target", (tx + 16, ty - 14), (40, 180, 255))

    cv2.circle(img, (ex, ey), 7, (240, 240, 240), -1, cv2.LINE_AA)
    cv2.circle(img, (ex, ey), 11, (240, 240, 240), 2, cv2.LINE_AA)
    _label(img, "Entry", (ex + 14, ey - 12), (240, 240, 240))

    if _dist(plan.needle_tip, plan.target) >= 8:
        cv2.circle(img, (nx, ny), 5, (255, 255, 255), -1, cv2.LINE_AA)
        _label(img, "Tip", (nx + 12, ny + 16), (255, 255, 255))

    cv2.rectangle(img, (0, 0), (w, 46), (10, 12, 16), -1)
    cv2.putText(
        img,
        f"{plan.status_label}",
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        status_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        f"angle {plan.needle_angle_deg:.1f} deg   path {plan.trajectory_length_px:.1f} px   tip-target {plan.tip_to_target_px:.1f} px",
        (170, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (210, 216, 220),
        1,
        cv2.LINE_AA,
    )

    cv2.rectangle(img, (0, h - 32), (w, h), (10, 12, 16), -1)
    cv2.putText(
        img,
        DISCLAIMER,
        (16, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (170, 180, 190),
        1,
        cv2.LINE_AA,
    )
    return img


def _label(img, text, org, color):
    x, y = int(org[0]), int(org[1])
    y = max(42, min(img.shape[0] - 36, y))
    x = max(8, min(img.shape[1] - 80, x))
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def suggested_defaults(lesion, predicted, frame_size=(VIDEO_W, OUT_H)):
    vw, vh = frame_size
    entry = (20.0, float(vh - 10))
    target = _as_xy(predicted) or _as_xy(lesion) or (vw * 0.5, vh * 0.4)
    path = _dist(entry, target)
    return {
        "entry": entry,
        "target": target,
        "safety_margin_px": 24.0,
        "needle_length_px": max(path, 80.0),
    }


def candidate_entries(frame_size=(VIDEO_W, OUT_H), margin=24, step=90):
    """Workspace-border entry points. No clinical workspace model is assumed."""
    vw, vh = int(frame_size[0]), int(frame_size[1])
    margin = max(8, int(margin))
    xs = list(range(margin, max(margin + 1, vw - margin), step))
    ys = list(range(margin, max(margin + 1, vh - margin), step))
    if xs[-1] != vw - 1 - margin:
        xs.append(vw - 1 - margin)
    if ys[-1] != vh - 1 - margin:
        ys.append(vh - 1 - margin)
    pts = []
    for x in xs:
        pts.append((float(x), float(margin)))
        pts.append((float(x), float(vh - 1 - margin)))
    for y in ys:
        pts.append((float(margin), float(y)))
        pts.append((float(vw - 1 - margin), float(y)))
    return list(dict.fromkeys(pts))


def candidate_targets(lesion, predicted, user_target, safety_margin_px):
    """Lesion, prediction, the user's target, and a few nearby samples."""
    pts = []
    for pt in (user_target, lesion, predicted):
        xy = _as_xy(pt)
        if xy:
            pts.append((round(xy[0], 2), round(xy[1], 2)))
    lesion_xy = _as_xy(lesion)
    if lesion_xy and safety_margin_px > 0:
        radius = 0.45 * float(safety_margin_px)
        for deg in (0, 90, 180, 270):
            rad = math.radians(deg)
            pts.append(
                (
                    round(lesion_xy[0] + radius * math.cos(rad), 2),
                    round(lesion_xy[1] + radius * math.sin(rad), 2),
                )
            )
    return list(dict.fromkeys(pts))


def _safety_violations(plan: BiopsyPlan) -> int:
    if plan.lesion is None or plan.target_error_px is None:
        return 1
    return 0 if plan.target_error_px <= plan.safety_margin_px else 1


def _is_feasible(plan: BiopsyPlan, min_path_px=16.0) -> bool:
    if plan.lesion is None:
        return False
    if plan.trajectory_length_px < min_path_px:
        return False
    if not plan.reach_ok:
        return False
    return True


def _minmax(values):
    if not values:
        return {}
    lo, hi = min(values), max(values)
    span = hi - lo
    if span < 1e-9:
        return {i: 0.0 for i in range(len(values))}
    return {i: (v - lo) / span for i, v in enumerate(values)}


def optimize_biopsy_paths(
    lesion,
    predicted=None,
    user_target=None,
    safety_margin_px=24.0,
    needle_length_px=800.0,
    frame_size=(VIDEO_W, OUT_H),
    path_length_weight=1.0,
    angle_weight=1.0,
    target_error_weight=1.0,
    safety_weight=1.0,
    frame_index=None,
    track_id=None,
):
    """Score border-entry candidates with a transparent weighted cost.

    total_cost =
        path_length_weight * normalized_path_length
        + angle_weight * normalized_angle
        + target_error_weight * normalized_target_error
        + safety_weight * safety_violations

    Normalization is min-max over feasible candidates only.
    Angle term uses |angle_deg| / 180 so smaller image-plane angles cost less.
    This is a simulation ranking, not a validated clinical planner.
    """
    weights = {
        "path_length_weight": float(path_length_weight),
        "angle_weight": float(angle_weight),
        "target_error_weight": float(target_error_weight),
        "safety_weight": float(safety_weight),
    }
    entries = candidate_entries(frame_size)
    targets = candidate_targets(lesion, predicted, user_target, safety_margin_px)
    if not targets:
        fallback = (frame_size[0] * 0.5, frame_size[1] * 0.4)
        targets = [fallback]

    raw_rows = []
    for entry in entries:
        for target in targets:
            plan = plan_biopsy(
                entry=entry,
                target=target,
                lesion=lesion,
                predicted=predicted,
                safety_margin_px=safety_margin_px,
                needle_length_px=needle_length_px,
                frame_size=frame_size,
                frame_index=frame_index,
                track_id=track_id,
            )
            raw_rows.append(
                {
                    "plan": plan,
                    "feasible": _is_feasible(plan),
                    "safety_violations": _safety_violations(plan),
                }
            )

    feasible = [row for row in raw_rows if row["feasible"]]
    norms_len = _minmax([row["plan"].trajectory_length_px for row in feasible])
    norms_ang = _minmax([abs(row["plan"].needle_angle_deg) / 180.0 for row in feasible])
    norms_err = _minmax(
        [0.0 if row["plan"].target_error_px is None else row["plan"].target_error_px for row in feasible]
    )

    scored = []
    feas_i = 0
    for row in raw_rows:
        item = {
            "entry": row["plan"].entry,
            "target": row["plan"].target,
            "path_length": row["plan"].trajectory_length_px,
            "needle_angle_deg": row["plan"].needle_angle_deg,
            "target_error": row["plan"].target_error_px,
            "safety_violations": row["safety_violations"],
            "feasible": row["feasible"],
            "reach_ok": row["plan"].reach_ok,
            "normalized_path_length": None,
            "normalized_angle": None,
            "normalized_target_error": None,
            "total_cost": None,
            "plan": row["plan"],
        }
        if row["feasible"]:
            n_len = norms_len[feas_i]
            n_ang = norms_ang[feas_i]
            n_err = norms_err[feas_i]
            item["normalized_path_length"] = n_len
            item["normalized_angle"] = n_ang
            item["normalized_target_error"] = n_err
            item["total_cost"] = (
                weights["path_length_weight"] * n_len
                + weights["angle_weight"] * n_ang
                + weights["target_error_weight"] * n_err
                + weights["safety_weight"] * row["safety_violations"]
            )
            feas_i += 1
        scored.append(item)

    ranked = sorted(
        [row for row in scored if row["feasible"] and row["total_cost"] is not None],
        key=lambda r: (r["total_cost"], r["path_length"]),
    )
    for i, row in enumerate(ranked, start=1):
        row["rank"] = i
    recommended = ranked[0] if ranked else None
    shortest = min(ranked, key=lambda r: (r["path_length"], r["total_cost"])) if ranked else None
    safest = (
        min(
            ranked,
            key=lambda r: (
                r["safety_violations"],
                1e9 if r["target_error"] is None else r["target_error"],
                r["total_cost"],
            ),
        )
        if ranked
        else None
    )
    if recommended:
        recommended["is_recommended"] = True
    if shortest:
        shortest["is_shortest"] = True
    if safest:
        safest["is_safest"] = True
    top3 = ranked[:3]
    return {
        "recommended": recommended,
        "shortest": shortest,
        "safest": safest,
        "ranked": ranked,
        "top3": top3,
        "candidates": scored,
        "n_candidates": len(scored),
        "n_feasible": len(ranked),
        "n_rejected": len(scored) - len(ranked),
        "weights": weights,
        "formula": (
            "total_cost = path_length_weight * normalized_path_length + "
            "angle_weight * normalized_angle + "
            "target_error_weight * normalized_target_error + "
            "safety_weight * safety_violations"
        ),
    }


def _same_path_row(row, other) -> bool:
    if row is None or other is None:
        return False
    return (
        abs(float(row["entry"][0]) - float(other["entry"][0])) < 0.05
        and abs(float(row["entry"][1]) - float(other["entry"][1])) < 0.05
        and abs(float(row["target"][0]) - float(other["target"][0])) < 0.05
        and abs(float(row["target"][1]) - float(other["target"][1])) < 0.05
    )


def _dashed_line(img, p0, p1, color, thickness=1, gap=10):
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    dist = math.hypot(x1 - x0, y1 - y0)
    if dist < 1:
        return
    n = max(2, int(dist / max(gap, 1)))
    for i in range(0, n, 2):
        t0 = i / n
        t1 = min((i + 1) / n, 1.0)
        a = (int(x0 + (x1 - x0) * t0), int(y0 + (y1 - y0) * t0))
        b = (int(x0 + (x1 - x0) * t1), int(y0 + (y1 - y0) * t1))
        cv2.line(img, a, b, color, thickness, cv2.LINE_AA)


def _path_endpoints(row):
    return (int(row["entry"][0]), int(row["entry"][1])), (int(row["target"][0]), int(row["target"][1]))


def render_planning_workspace(frame_bgr, opt, manual_plan=None, max_rejected=18):
    """Ultrasound workspace with lesion, safety region, candidates, and recommended path."""
    img = frame_bgr.copy()
    h, w = img.shape[:2]
    candidates = list(opt.get("candidates") or [])
    recommended = opt.get("recommended")
    shortest = opt.get("shortest")
    safest = opt.get("safest")
    src = recommended or shortest or (candidates[0] if candidates else None)
    lesion = src["plan"].lesion if src else (manual_plan.lesion if manual_plan else None)
    predicted = src["plan"].predicted if src else (manual_plan.predicted if manual_plan else None)
    safety = src["plan"].safety_margin_px if src else (manual_plan.safety_margin_px if manual_plan else 0.0)

    overlay = img.copy()
    if lesion is not None and safety > 0:
        cx, cy = int(lesion[0]), int(lesion[1])
        cv2.circle(overlay, (cx, cy), int(safety), (80, 200, 220), -1, cv2.LINE_AA)
        img = cv2.addWeighted(overlay, 0.16, img, 0.84, 0)
        cv2.circle(img, (cx, cy), int(safety), (80, 200, 220), 2, cv2.LINE_AA)

    rejected = [row for row in candidates if not row.get("feasible")]
    feasible = [row for row in candidates if row.get("feasible")]
    if len(rejected) > max_rejected:
        step = max(1, len(rejected) // max_rejected)
        rejected = rejected[::step][:max_rejected]

    reject_layer = img.copy()
    for row in rejected:
        p0, p1 = _path_endpoints(row)
        _dashed_line(reject_layer, p0, p1, (70, 70, 100), 1, gap=12)
    img = cv2.addWeighted(reject_layer, 0.35, img, 0.65, 0)

    for row in feasible:
        if _same_path_row(row, recommended) or _same_path_row(row, shortest) or _same_path_row(row, safest):
            continue
        p0, p1 = _path_endpoints(row)
        cv2.line(img, p0, p1, (120, 145, 90), 1, cv2.LINE_AA)

    if shortest is not None and not _same_path_row(shortest, recommended):
        p0, p1 = _path_endpoints(shortest)
        cv2.line(img, p0, p1, (0, 165, 255), 2, cv2.LINE_AA)
        cv2.circle(img, p0, 5, (0, 165, 255), -1, cv2.LINE_AA)
        _label(img, "Shortest", (p0[0] + 8, p0[1] - 10), (0, 165, 255))

    if safest is not None and not _same_path_row(safest, recommended) and not _same_path_row(safest, shortest):
        p0, p1 = _path_endpoints(safest)
        cv2.line(img, p0, p1, (90, 200, 140), 2, cv2.LINE_AA)
        cv2.circle(img, p0, 5, (90, 200, 140), -1, cv2.LINE_AA)
        _label(img, "Safest", (p0[0] + 8, p0[1] - 10), (90, 200, 140))

    if recommended is not None:
        p0, p1 = _path_endpoints(recommended)
        cv2.line(img, p0, p1, (40, 210, 255), 5, cv2.LINE_AA)
        cv2.circle(img, p0, 8, (40, 210, 255), -1, cv2.LINE_AA)
        cv2.circle(img, p1, 6, (40, 180, 255), -1, cv2.LINE_AA)
        cv2.drawMarker(img, p1, (40, 180, 255), cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
        _label(img, "Recommended", (p0[0] + 10, p0[1] + 18), (40, 210, 255))
        _label(img, "Target", (p1[0] + 12, p1[1] - 12), (40, 180, 255))

    if lesion is not None:
        lx, ly = int(lesion[0]), int(lesion[1])
        cv2.circle(img, (lx, ly), 7, (70, 210, 90), -1, cv2.LINE_AA)
        cv2.circle(img, (lx, ly), 12, (70, 210, 90), 2, cv2.LINE_AA)
        _label(img, "Lesion", (lx + 16, ly + 6), (70, 210, 90))

    show_pred = predicted is not None and (lesion is None or _dist(predicted, lesion) >= 8)
    if show_pred:
        px, py = int(predicted[0]), int(predicted[1])
        cv2.drawMarker(img, (px, py), (50, 50, 220), cv2.MARKER_TILTED_CROSS, 16, 2, cv2.LINE_AA)
        _label(img, "LSTM target", (px + 14, py - 12), (50, 50, 220))

    if manual_plan is not None:
        rec_plan = recommended["plan"] if recommended else None
        same_as_rec = rec_plan is not None and (
            abs(manual_plan.entry[0] - rec_plan.entry[0]) < 2
            and abs(manual_plan.entry[1] - rec_plan.entry[1]) < 2
            and abs(manual_plan.target[0] - rec_plan.target[0]) < 2
            and abs(manual_plan.target[1] - rec_plan.target[1]) < 2
        )
        if not same_as_rec:
            p0 = (int(manual_plan.entry[0]), int(manual_plan.entry[1]))
            p1 = (int(manual_plan.target[0]), int(manual_plan.target[1]))
            _dashed_line(img, p0, p1, (210, 120, 210), 2, gap=8)
            cv2.circle(img, p0, 6, (210, 120, 210), -1, cv2.LINE_AA)
            _label(img, "Manual", (p0[0] + 8, p0[1] - 12), (210, 120, 210))

    cv2.rectangle(img, (0, 0), (w, 28), (10, 12, 16), -1)
    cv2.putText(
        img,
        f"Candidates {opt.get('n_candidates', 0)}  ·  feasible {opt.get('n_feasible', 0)}  ·  rejected {opt.get('n_rejected', 0)}",
        (12, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (210, 216, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.rectangle(img, (0, h - 28), (w, h), (10, 12, 16), -1)
    cv2.putText(
        img,
        DISCLAIMER,
        (12, h - 9),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (170, 180, 190),
        1,
        cv2.LINE_AA,
    )
    return img


def render_optimized_overlay(frame_bgr, recommended, top3, manual_plan=None, candidates=None, shortest=None, safest=None):
    """Draw optimizer candidates on the ultrasound pane. Recommended path is never replaced by the manual path."""
    opt = {
        "recommended": recommended,
        "shortest": shortest,
        "safest": safest,
        "candidates": candidates if candidates is not None else list(top3 or []),
        "n_candidates": len(candidates) if candidates is not None else len(list(top3 or [])),
        "n_feasible": sum(1 for row in (candidates if candidates is not None else list(top3 or [])) if row.get("feasible")),
        "n_rejected": 0,
    }
    if candidates is not None:
        opt["n_rejected"] = sum(1 for row in candidates if not row.get("feasible"))
    return render_planning_workspace(frame_bgr, opt, manual_plan=manual_plan)


def save_biopsy_preview(frame_bgr, plan: BiopsyPlan, output_path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = render_biopsy_plan(frame_bgr, plan)
    cv2.imwrite(str(output_path), rendered)
    return output_path
