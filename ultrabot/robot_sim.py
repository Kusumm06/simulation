"""2D virtual needle animation along an already-recommended biopsy path."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ultrabot.biopsy import DISCLAIMER as PLAN_DISCLAIMER

ROBOT_DISCLAIMER = "Virtual robotic biopsy simulation — not clinical or robotic control."


def _dist(a, b) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _lerp(a, b, t: float):
    return (float(a[0]) + (float(b[0]) - float(a[0])) * t, float(a[1]) + (float(b[1]) - float(a[1])) * t)


def needle_pose(entry, target, needle_length_px, t: float):
    """Interpolate the tip along the planned segment, clipped by needle length."""
    path = _dist(entry, target)
    if path < 1e-6:
        return entry, entry, 0.0, 0.0
    usable = min(float(needle_length_px), path)
    traveled = usable * max(0.0, min(1.0, t))
    ux = (float(target[0]) - float(entry[0])) / path
    uy = (float(target[1]) - float(entry[1])) / path
    tip = (float(entry[0]) + ux * traveled, float(entry[1]) + uy * traveled)
    return entry, tip, traveled, usable


def evaluate_simulation(recommended: dict):
    """Score the finished insertion from the recommended path coordinates."""
    plan = recommended["plan"]
    entry = plan.entry
    target = plan.target
    lesion = plan.lesion
    _, tip, traveled, usable = needle_pose(entry, target, plan.needle_length_px, 1.0)
    tip_to_target = _dist(tip, target)
    tip_to_lesion = _dist(tip, lesion) if lesion is not None else None
    safety = float(plan.safety_margin_px)
    reached_target = tip_to_target <= safety + 1e-6
    success = bool(reached_target)
    if success:
        reason = (
            f"Simulated needle reached the target ({tip_to_target:.1f} px tip-to-target, "
            f"within the {safety:.1f} px safety margin)."
        )
        nav_status = "SUCCESS"
    else:
        reason = (
            f"Needle cannot reach the target: tip is {tip_to_target:.1f} px away "
            f"(safety margin {safety:.1f} px, needle {plan.needle_length_px:.1f} px, "
            f"path {float(recommended['path_length']):.1f} px)."
        )
        nav_status = "FAILURE"
    return {
        "entry": entry,
        "target": target,
        "lesion": lesion,
        "final_tip": tip,
        "path_length": float(recommended["path_length"]),
        "needle_angle_deg": float(recommended["needle_angle_deg"]),
        "target_error": recommended.get("target_error"),
        "safety_violations": recommended.get("safety_violations"),
        "feasible": bool(recommended.get("feasible")),
        "tip_to_target_px": tip_to_target,
        "tip_to_lesion_px": tip_to_lesion,
        "traveled_px": traveled,
        "usable_needle_px": usable,
        "safety_margin_px": safety,
        "success": success,
        "status_label": "SUCCESS" if success else "FAILURE",
        "navigation_status": nav_status,
        "status_reason": reason,
        "disclaimer": ROBOT_DISCLAIMER,
    }


def render_robot_frame(workspace_bgr, recommended, t: float, status_label=None):
    """Draw the workspace, planned path, and a simple 2D needle/robot at progress t."""
    plan = recommended["plan"]
    img = workspace_bgr.copy()
    h, w = img.shape[:2]
    overlay = img.copy()
    entry, tip, traveled, _usable = needle_pose(plan.entry, plan.target, plan.needle_length_px, t)

    if plan.lesion is not None and plan.safety_margin_px > 0:
        cx, cy = int(plan.lesion[0]), int(plan.lesion[1])
        cv2.circle(overlay, (cx, cy), int(plan.safety_margin_px), (80, 200, 220), -1, cv2.LINE_AA)
        img = cv2.addWeighted(overlay, 0.16, img, 0.84, 0)
        cv2.circle(img, (cx, cy), int(plan.safety_margin_px), (80, 200, 220), 2, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 6, (70, 210, 90), -1, cv2.LINE_AA)

    cv2.rectangle(img, (6, 6), (w - 6, h - 42), (90, 110, 130), 1, cv2.LINE_AA)
    cv2.putText(
        img,
        "Ultrasound workspace",
        (14, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 196, 210),
        1,
        cv2.LINE_AA,
    )

    tx, ty = int(plan.target[0]), int(plan.target[1])
    ex, ey = int(entry[0]), int(entry[1])
    nx, ny = int(tip[0]), int(tip[1])
    cv2.line(img, (ex, ey), (tx, ty), (40, 210, 255), 2, cv2.LINE_AA)
    cv2.drawMarker(img, (tx, ty), (40, 180, 255), cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
    cv2.circle(img, (tx, ty), 5, (40, 180, 255), -1, cv2.LINE_AA)
    cv2.putText(img, "Trajectory", (int((ex + tx) / 2) + 8, int((ey + ty) / 2) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 210, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "Entry", (ex + 12, ey + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 210, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "Target", (tx + 10, ty - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 180, 255), 1, cv2.LINE_AA)
    if plan.lesion is not None:
        cv2.putText(
            img,
            "Lesion target",
            (int(plan.lesion[0]) + 10, int(plan.lesion[1]) + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (70, 210, 90),
            1,
            cv2.LINE_AA,
        )

    dx, dy = tx - ex, ty - ey
    path = math.hypot(dx, dy)
    if path > 1:
        ux, uy = dx / path, dy / path
        px, py = -uy, ux
        base = np.array(
            [
                [ex - ux * 16 + px * 14, ey - uy * 16 + py * 14],
                [ex - ux * 16 - px * 14, ey - uy * 16 - py * 14],
                [ex + ux * 8 - px * 14, ey + uy * 8 - py * 14],
                [ex + ux * 8 + px * 14, ey + uy * 8 + py * 14],
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(img, base, (36, 42, 48))
        cv2.polylines(img, [base], True, (230, 230, 235), 2, cv2.LINE_AA)

    cv2.line(img, (ex, ey), (nx, ny), (235, 235, 240), 4, cv2.LINE_AA)
    cv2.circle(img, (ex, ey), 8, (40, 210, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (nx, ny), 5, (255, 255, 255), -1, cv2.LINE_AA)

    label = status_label or "SIMULATING"
    color = (70, 190, 90) if label == "SUCCESS" else ((60, 60, 220) if label == "FAILURE" else (40, 210, 255))
    cv2.rectangle(img, (0, h - 36), (w, h), (10, 12, 16), -1)
    cv2.putText(img, label, (16, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    cv2.putText(
        img,
        f"{int(t * 100):3d}%  {traveled:.1f} px   {ROBOT_DISCLAIMER}",
        (170, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (170, 180, 190),
        1,
        cv2.LINE_AA,
    )
    return img


def run_robot_simulation(workspace_bgr, recommended, output_gif, n_steps=24):
    """Animate the needle and write a GIF. Does not modify the recommended path."""
    if recommended is None:
        raise ValueError("No recommended trajectory is available to simulate.")
    result = evaluate_simulation(recommended)
    frames = []
    for i in range(n_steps):
        t = i / max(n_steps - 1, 1)
        label = None if i < n_steps - 1 else result["status_label"]
        frames.append(render_robot_frame(workspace_bgr, recommended, t, status_label=label))

    output_gif = Path(output_gif)
    output_gif.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
    images[0].save(output_gif, save_all=True, append_images=images[1:], duration=90, loop=0)
    final_path = output_gif.with_name(output_gif.stem + "_final.png")
    cv2.imwrite(str(final_path), frames[-1])
    result["gif_path"] = str(output_gif)
    result["final_image_path"] = str(final_path)
    result["n_steps"] = n_steps
    result["planner_disclaimer"] = PLAN_DISCLAIMER
    return result
