"""HUD drawing helpers used by the existing YOLOv8 + LSTM demo."""

from __future__ import annotations

import math

import cv2
import numpy as np

OUT_W, OUT_H = 1280, 720
SIDEBAR_W = 290
VIDEO_W = OUT_W - SIDEBAR_W
FPS_OUT = 10

C_CYAN = (255, 220, 0)
C_GREEN = (55, 230, 55)
C_RED = (50, 50, 220)
C_WHITE = (240, 240, 240)
C_YELLOW = (55, 210, 255)
C_PANEL = (12, 14, 18)
C_BORDER = (42, 48, 58)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def txt(img, text, org, scale=0.5, color=C_WHITE, thickness=1):
    cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def txt_shadow(img, text, org, scale=0.5, color=C_WHITE, thickness=1):
    cv2.putText(img, text, (org[0] + 1, org[1] + 1), FONT, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def h_rule(img, x1, x2, y, color=C_BORDER, thickness=1):
    cv2.line(img, (x1, y), (x2, y), color, thickness, cv2.LINE_AA)


def corner_brackets(img, x1, y1, x2, y2, color, size=22, thickness=1):
    pts = [
        ((x1, y1), (x1 + size, y1), (x1, y1 + size)),
        ((x2, y1), (x2 - size, y1), (x2, y1 + size)),
        ((x1, y2), (x1 + size, y2), (x1, y2 - size)),
        ((x2, y2), (x2 - size, y2), (x2, y2 - size)),
    ]
    for corner, h_end, v_end in pts:
        cv2.line(img, corner, h_end, color, thickness, cv2.LINE_AA)
        cv2.line(img, corner, v_end, color, thickness, cv2.LINE_AA)


def dot_line(img, p1, p2, spacing=10, r=1, color=C_WHITE):
    x1, y1 = p1
    x2, y2 = p2
    dist = math.hypot(x2 - x1, y2 - y1)
    if dist < 1:
        return
    steps = max(1, int(dist // spacing))
    for i in range(steps + 1):
        t = i / steps
        cv2.circle(img, (int(x1 + (x2 - x1) * t), int(y1 + (y2 - y1) * t)), r, color, -1, cv2.LINE_AA)


def target_rings(img, center, color, radii=(8, 13), thickness=1):
    for radius in radii:
        cv2.circle(img, center, radius, color, thickness, cv2.LINE_AA)


def fading_trail(img, pts, color=(55, 210, 255)):
    n = len(pts)
    if n < 2:
        return
    for i, p in enumerate(pts):
        a = (i + 1) / n
        c = tuple(int(ch * a) for ch in color)
        cv2.circle(img, p, max(1, int(1 + a * 2)), c, -1, cv2.LINE_AA)


def ecg_line(img, x0, y0, w=180, color=C_GREEN):
    pts = np.array(
        [
            [0, 0],
            [10, 0],
            [16, -3],
            [21, 6],
            [27, -14],
            [33, 6],
            [40, 0],
            [52, 0],
            [58, -3],
            [64, -10],
            [70, 3],
            [76, 0],
            [85, -8],
            [92, 0],
            [w, 0],
        ]
    )
    pts[:, 0] = (pts[:, 0] * w / pts[-1, 0]).astype(int)
    coords = np.column_stack([pts[:, 0] + x0, pts[:, 1] + y0]).reshape(-1, 1, 2).astype(np.int32)
    cv2.polylines(img, [coords], False, color, 1, cv2.LINE_AA)


def semi_rect(img, x1, y1, x2, y2, color=(0, 0, 0), alpha=0.55):
    roi = img[y1:y2, x1:x2]
    overlay = np.full_like(roi, color)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)


def build_sidebar(fps, tumor_pt, pred_pt, needle_pt, elapsed_s, tumor_found):
    sb = np.full((OUT_H, SIDEBAR_W, 3), C_PANEL, dtype=np.uint8)
    cv2.line(sb, (0, 0), (0, OUT_H), C_BORDER, 1)
    sx = 24

    txt(sb, "ULTRA-BOT", (sx, 36), scale=0.82, color=C_CYAN, thickness=2)
    txt(sb, "AI GUIDED NEEDLE SYSTEM", (sx, 56), scale=0.36, color=(170, 180, 190), thickness=1)
    h_rule(sb, sx, SIDEBAR_W - sx, 68)

    txt(sb, "STATUS", (sx, 98), scale=0.38, color=(120, 220, 120), thickness=1)
    txt(
        sb,
        "ACTIVE" if tumor_found else "SCANNING",
        (sx, 130),
        scale=0.90,
        color=C_GREEN if tumor_found else (60, 200, 255),
        thickness=2,
    )
    ecg_line(sb, sx, 154, w=SIDEBAR_W - sx * 2, color=C_GREEN if tumor_found else (60, 200, 255))
    h_rule(sb, sx, SIDEBAR_W - sx, 172)

    txt(sb, "MODEL", (sx, 200), scale=0.38, color=(210, 185, 60), thickness=1)
    txt(sb, "YOLOv8 + LSTM", (sx, 222), scale=0.52, color=C_WHITE, thickness=1)
    h_rule(sb, sx, SIDEBAR_W - sx, 236)

    txt(sb, "MODE", (sx, 264), scale=0.38, color=(210, 185, 60), thickness=1)
    txt(sb, "TUMOR TRACKING", (sx, 285), scale=0.46, color=C_WHITE, thickness=1)
    txt(sb, "NEEDLE GUIDANCE", (sx, 305), scale=0.46, color=C_WHITE, thickness=1)
    h_rule(sb, sx, SIDEBAR_W - sx, 320)

    txt(sb, "COORDINATES", (sx, 348), scale=0.38, color=(210, 185, 60), thickness=1)

    def coord_row(y, dot_color, label, pt):
        cv2.circle(sb, (sx + 6, y - 5), 5, dot_color, -1, cv2.LINE_AA)
        cv2.circle(sb, (sx + 6, y - 5), 5, C_WHITE, 1, cv2.LINE_AA)
        txt(sb, label, (sx + 18, y), scale=0.38, color=(160, 170, 180), thickness=1)
        coord_str = f"({pt[0]}, {pt[1]})" if pt else "--"
        txt(sb, coord_str, (sx + 18, y + 16), scale=0.46, color=C_WHITE, thickness=1)

    coord_row(378, C_GREEN, "TUMOR CENTER", tumor_pt if tumor_found else None)
    coord_row(420, C_RED, "PREDICTION", pred_pt if tumor_found else None)
    coord_row(462, C_WHITE, "NEEDLE TIP", needle_pt if tumor_found else None)
    h_rule(sb, sx, SIDEBAR_W - sx, 484)

    txt(sb, "LEGEND", (sx, 510), scale=0.38, color=(210, 185, 60), thickness=1)

    def legend_item(y, dot_color, label, filled=True):
        if filled:
            cv2.circle(sb, (sx + 8, y), 6, dot_color, -1, cv2.LINE_AA)
        else:
            cv2.circle(sb, (sx + 8, y), 6, dot_color, 1, cv2.LINE_AA)
        txt(sb, label, (sx + 22, y + 5), scale=0.42, color=(200, 210, 220), thickness=1)

    legend_item(534, C_GREEN, "TUMOR CENTER")
    legend_item(558, C_RED, "PREDICTION")
    legend_item(582, C_WHITE, "NEEDLE TIP", filled=False)
    for i in range(6):
        a = (i + 1) / 6
        c = tuple(int(ch * a) for ch in C_YELLOW)
        cv2.circle(sb, (sx + 4 + i * 10, 607), 2, c, -1, cv2.LINE_AA)
    txt(sb, "TRAJECTORY", (sx + 22, 611), scale=0.42, color=(200, 210, 220), thickness=1)
    h_rule(sb, sx, SIDEBAR_W - sx, 626)

    mins = elapsed_s // 60
    secs = elapsed_s % 60
    txt(sb, f"FPS  {int(fps):>3}", (sx, 658), scale=0.46, color=C_GREEN, thickness=1)
    txt(sb, f"TIME  {mins:02}:{secs:02}", (sx + 115, 658), scale=0.46, color=C_CYAN, thickness=1)
    return sb


def annotate_frame(
    video,
    fps,
    elapsed,
    tumor_found,
    tumor_pt,
    pred_pt,
    needle_pt,
    trajectory,
    track_id=None,
    track_status=None,
):
    """Draw the existing cinematic HUD onto the ultrasound pane."""
    vh, vw = video.shape[:2]
    if trajectory:
        fading_trail(video, trajectory, color=C_YELLOW)
    if needle_pt and pred_pt:
        cv2.line(video, (20, vh - 10), needle_pt, (210, 215, 220), 2, cv2.LINE_AA)
        dot_line(video, needle_pt, pred_pt, spacing=10, r=1, color=(200, 205, 210))
        cv2.circle(video, needle_pt, 5, C_WHITE, -1, cv2.LINE_AA)
        cv2.circle(video, needle_pt, 9, C_WHITE, 1, cv2.LINE_AA)
    if tumor_found and tumor_pt:
        cv2.circle(video, tumor_pt, 6, C_GREEN, -1, cv2.LINE_AA)
        target_rings(video, tumor_pt, C_GREEN, radii=(10, 15))
    if pred_pt:
        cv2.circle(video, pred_pt, 5, C_RED, -1, cv2.LINE_AA)
        target_rings(video, pred_pt, C_RED, radii=(9, 14))

    semi_rect(video, 0, 0, vw, 44, color=(0, 0, 0), alpha=0.55)
    txt_shadow(video, "ULTRA-BOT", (14, 30), scale=0.80, color=C_CYAN, thickness=2)
    txt_shadow(video, "GUIDED NEEDLE SYSTEM", (154, 30), scale=0.42, color=(190, 200, 210), thickness=1)
    txt_shadow(video, f"FPS  {int(fps)}", (vw - 100, 30), scale=0.52, color=C_GREEN, thickness=1)
    corner_brackets(video, 6, 6, vw - 6, vh - 6, color=(0, 220, 255), size=24, thickness=1)

    readouts = []
    if track_id is not None:
        status = (track_status or "ACTIVE").upper()
        readouts.append((C_CYAN, f"TRACK {track_id}", None, status))
    if tumor_found and tumor_pt:
        readouts.append((C_GREEN, "CURRENT", tumor_pt, None))
    if pred_pt:
        readouts.append((C_RED, "PREDICTION", pred_pt, None))
    if needle_pt:
        readouts.append((C_WHITE, "NEEDLE TIP", needle_pt, None))
    base_y = 80
    for item in readouts:
        dot_c, label, pt, extra = item
        semi_rect(video, 14, base_y - 20, 210, base_y + 14, color=(0, 0, 0), alpha=0.45)
        cv2.circle(video, (28, base_y - 6), 5, dot_c, -1, cv2.LINE_AA)
        txt_shadow(video, label, (40, base_y - 4), scale=0.40, color=(190, 200, 210), thickness=1)
        value = extra if extra else (f"({pt[0]}, {pt[1]})" if pt else "--")
        txt_shadow(video, value, (40, base_y + 10), scale=0.46, color=C_WHITE, thickness=1)
        base_y += 52

    semi_rect(video, 0, vh - 34, 180, vh, color=(0, 0, 0), alpha=0.55)
    mins = elapsed // 60
    secs = elapsed % 60
    txt_shadow(video, "TIME", (14, vh - 12), scale=0.42, color=C_CYAN, thickness=1)
    txt_shadow(video, f"{mins:02}:{secs:02}", (70, vh - 12), scale=0.46, color=C_WHITE, thickness=1)

    sidebar = build_sidebar(fps, tumor_pt, pred_pt, needle_pt, elapsed, tumor_found)
    return np.hstack([video, sidebar])
