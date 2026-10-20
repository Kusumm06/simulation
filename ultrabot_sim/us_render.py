"""
Drawing INTO the ultrasound pane (990 x 720, BGR):

  * mm scale bars (lateral and axial ticks every 5 mm);
  * the parametric anatomy MODEL (bands, fascia, ribs with acoustic shadow,
    pleura, vessels with their margin ring, Cooper's ligaments), labelled
    as a model;
  * YOLO box with class / confidence, track id / state, centre, trail;
  * the LSTM one-step prediction and the autoregressive rollout chain with
    growing uncertainty circles;
  * candidate paths, shortest (orange), safest (green), recommended (cyan),
    the legacy 2D planner (grey, toggle L);
  * the virtual needle as the arm actually holds it: echogenic shaft with
    comet-tail reverberation, a bright tip, only the in-plane part, plus the
    throw zone bracket.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from ultrabot_sim import config as cfg

FONT = cv2.FONT_HERSHEY_SIMPLEX
C_SHORT = (0, 150, 255)       # orange (BGR)
C_SAFE = (90, 210, 90)        # green
C_REC = (255, 215, 40)        # cyan-ish
C_LEGACY = (150, 150, 150)
C_LESION_B = (110, 230, 110)
C_LESION_M = (80, 80, 245)
C_PRED = (230, 80, 230)
C_VESSEL = (60, 60, 220)
C_FASCIA = (70, 110, 200)
PATH_COLORS = {"shortest": C_SHORT, "safest": C_SAFE, "recommended": C_REC,
               "legacy": C_LEGACY}


def text(img, s, org, color=(230, 235, 240), scale=0.45, thick=1, bg=True):
    x, y = int(org[0]), int(org[1])
    if bg:
        (w, h), _ = cv2.getTextSize(s, FONT, scale, thick)
        cv2.rectangle(img, (x - 3, y - h - 4), (x + w + 3, y + 4),
                      (12, 14, 18), -1)
    cv2.putText(img, s, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def _blend(img, overlay, a):
    cv2.addWeighted(overlay, a, img, 1 - a, 0, dst=img)


def dashed(img, p0, p1, color, thick=1, gap=8):
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    L = math.hypot(x1 - x0, y1 - y0)
    if L < 1:
        return
    n = max(2, int(L / gap))
    for i in range(0, n, 2):
        a = (int(x0 + (x1 - x0) * i / n), int(y0 + (y1 - y0) * i / n))
        b = (int(x0 + (x1 - x0) * min(i + 1, n) / n),
             int(y0 + (y1 - y0) * min(i + 1, n) / n))
        cv2.line(img, a, b, color, thick, cv2.LINE_AA)


def scale_bars(img, calib, color=(220, 220, 220)):
    h, w = img.shape[:2]
    kx = w / calib.pane_w
    ky = h / calib.pane_h
    # axial ticks along the left edge
    for mm in range(0, int(calib.depth_mm) + 1, 5):
        v = int(mm / calib.mm_per_px_axial * ky)
        L = 10 if mm % 10 == 0 else 5
        cv2.line(img, (0, v), (L, v), color, 1, cv2.LINE_AA)
        if mm % 10 == 0 and 0 < v < h - 10:
            cv2.putText(img, f"{mm}", (13, v + 4), FONT, 0.35 * max(kx, 0.7),
                        color, 1, cv2.LINE_AA)
    # lateral ticks along the bottom
    half = calib.width_mm / 2
    for mm in range(-int(half), int(half) + 1, 5):
        u = int((mm / calib.mm_per_px_lateral + calib.pane_w / 2) * kx)
        L = 10 if mm % 10 == 0 else 5
        cv2.line(img, (u, h - 1), (u, h - 1 - L), color, 1, cv2.LINE_AA)
    cv2.putText(img, "mm", (13, h - 14), FONT, 0.35 * max(kx, 0.7), color, 1,
                cv2.LINE_AA)


def _px(calib, x, z):
    u, v = calib.mm_to_px(x, z)
    return int(round(float(u))), int(round(float(v)))


def draw_anatomy(img, an, calib):
    if an is None:
        return
    H, W = img.shape[:2]
    xs = np.linspace(-calib.half_width_mm - 2, calib.half_width_mm + 2, 60)
    ov = img.copy()
    # fat band (skin -> fat bottom)
    top = [_px(calib, x, float(an.skin_depth(x)) + cfg.SKIN_THICK_MM)
           for x in xs]
    bot = [_px(calib, x, float(an.fat_bottom(x))) for x in xs]
    poly = np.array(top + bot[::-1], np.int32)
    cv2.fillPoly(ov, [poly], (60, 170, 210))
    # muscle band (fascia -> muscle bottom)
    ft = [_px(calib, x, float(an.fascia_depth(x))) for x in xs]
    mb = [_px(calib, x, float(an.muscle_bottom(x))) for x in xs]
    cv2.fillPoly(ov, [np.array(ft + mb[::-1], np.int32)], (60, 70, 170))
    # below the pleura: lung (dark)
    pl = [_px(calib, x, float(an.pleura_depth(x))) for x in xs]
    cv2.fillPoly(ov, [np.array(pl + [(pl[-1][0], H + 5), (pl[0][0], H + 5)],
                               np.int32)], (40, 30, 30))
    _blend(img, ov, 0.18)
    cv2.polylines(img, [np.array(ft, np.int32)], False, C_FASCIA, 2,
                  cv2.LINE_AA)
    cv2.polylines(img, [np.array(pl, np.int32)], False, (235, 235, 235), 1,
                  cv2.LINE_AA)
    # ribs with acoustic shadow
    for (x, z, r) in an.ribs:
        if abs(x) > calib.half_width_mm + r:
            continue
        c = _px(calib, x, z)
        rx = int(r / calib.mm_per_px_lateral)
        rz = int(r / calib.mm_per_px_axial)
        sh = img.copy()
        cv2.rectangle(sh, (c[0] - rx, c[1]), (c[0] + rx, H), (0, 0, 0), -1)
        _blend(img, sh, 0.35)
        cv2.ellipse(img, c, (rx, rz), 0, 180, 360, (225, 225, 225), 2,
                    cv2.LINE_AA)
    # Cooper's ligaments
    for (a, b) in an.ligaments:
        cv2.line(img, _px(calib, *a), _px(calib, *b), (200, 200, 205), 1,
                 cv2.LINE_AA)
    # vessels (+ margin ring)
    for v in an.vessels:
        c = _px(calib, v.x, v.z)
        rx = max(2, int(v.r / calib.mm_per_px_lateral))
        rz = max(2, int(v.r / calib.mm_per_px_axial))
        cv2.ellipse(img, c, (rx, rz), 0, 0, 360, (20, 20, 40), -1,
                    cv2.LINE_AA)
        cv2.ellipse(img, c, (rx, rz), 0, 0, 360, C_VESSEL, 2, cv2.LINE_AA)
        m = v.r + cfg.VESSEL_MARGIN_MM
        _dashed_ellipse(img, c, (int(m / calib.mm_per_px_lateral),
                                 int(m / calib.mm_per_px_axial)), C_VESSEL)
    text(img, "MODEL anatomy (parametric, not detected)", (W - 330, H - 52),
         (180, 190, 200), 0.4)


def _dashed_ellipse(img, c, axes, color, n=36):
    for k in range(0, n, 2):
        cv2.ellipse(img, c, axes, 0, k * 360 / n, (k + 1) * 360 / n, color,
                    1, cv2.LINE_AA)


def _clip_draw_line(img, p0, p1, color, thick, dash=False):
    H, W = img.shape[:2]
    ok, a, b = cv2.clipLine((0, 0, W, H), tuple(map(int, p0)),
                            tuple(map(int, p1)))
    if not ok:
        return False
    if dash:
        dashed(img, a, b, color, thick)
    else:
        cv2.line(img, a, b, color, thick, cv2.LINE_AA)
    return True


def draw_plan(img, plan, calib, color, thick=2, label=None, faint=False):
    if plan is None:
        return
    E = _px(calib, *plan.entry_mm)
    P = _px(calib, *plan.prefire_mm)
    F = _px(calib, *plan.fired_mm)
    _clip_draw_line(img, E, F, color, thick, dash=faint)
    if not faint:
        # throw zone bracket
        ov = img.copy()
        cv2.line(ov, P, F, color, thick + 6, cv2.LINE_AA)
        _blend(img, ov, 0.25)
        H, W = img.shape[:2]
        # entry marker at the image edge (entries are beside the probe)
        side = 1 if plan.entry_mm[0] > 0 else -1
        ex = W - 4 if side > 0 else 4
        ok, a, b = cv2.clipLine((0, 0, W, H), E, F)
        y = a[1] if ok else 12
        cv2.arrowedLine(img, (ex, y), (ex - side * 22, y), color, 2,
                        cv2.LINE_AA, tipLength=0.5)
        if label:
            tx = W - 220 if side > 0 else 30
            text(img, label, (tx, max(y - 10, 16)), color, 0.42)


def draw_needle(img, scene, proc, calib, pose):
    """The needle as the arm holds it, as an ultrasound would show it."""
    f, d, _ = scene.needle_frame()
    hub = f + d * (cfg.GUN_LEN)
    L = cfg.NEEDLE_LEN_MM / 1000.0
    nd = scene.needle
    c_tip = hub + d * (L + nd.cannula_ext / 1000.0)
    s_tip = hub + d * (L + nd.stylet_ext / 1000.0)
    pts = np.linspace(0, 1, 120)[:, None]
    seg = hub[None] + (s_tip - hub)[None] * pts
    P = pose.world_to_probe(seg)
    vis = np.abs(P[:, 1]) <= cfg.US_SLICE_MM / 2
    inside = P[:, 2] > 0
    sel = vis & inside
    if sel.sum() < 2:
        return
    P = P[sel]
    uv = np.array([calib.mm_to_px(p[0], p[2]) for p in P], float)
    a, b = uv[0], uv[-1]
    # drawn a little thinner than the true 2.1 mm so the target stays
    # visible under it (the echo of a real needle is ~1-2 mm)
    od_px = 0.55 * cfg.NEEDLE_OD_MM / calib.mm_per_px_axial
    dvec = b - a
    Lp = np.linalg.norm(dvec)
    if Lp < 2:
        return
    t = dvec / Lp
    n = np.array([-t[1], t[0]])
    if n[1] < 0:
        n = -n
    ov = img.copy()
    # comet-tail reverberation: fading copies deeper in the image
    for k in range(1, 6):
        off = n * od_px * 0.9 * k
        cv2.line(ov, tuple((a + off).astype(int)), tuple((b + off).astype(int)),
                 (200, 200, 200), max(1, int(od_px * 0.18)), cv2.LINE_AA)
        _blend(img, ov, 0.18 * (1 - k / 6))
        ov = img.copy()
    # shaft: two bright specular walls, slightly brighter core
    for s in (-0.5, 0.5):
        off = n * od_px * s
        cv2.line(img, tuple((a + off).astype(int)), tuple((b + off).astype(int)),
                 (245, 248, 250), max(2, int(od_px * 0.22)), cv2.LINE_AA)
    cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), (200, 205, 210),
             max(1, int(od_px * 0.4)), cv2.LINE_AA)
    # bright tip
    cv2.circle(img, tuple(b.astype(int)), max(4, int(od_px * 0.55)),
               (255, 255, 255), -1, cv2.LINE_AA)
    if nd.stylet_ext > nd.cannula_ext + 0.5:
        ct = pose.world_to_probe(c_tip)
        if abs(ct[1]) <= cfg.US_SLICE_MM / 2 and ct[2] > 0:
            cu = calib.mm_to_px(ct[0], ct[2])
            cv2.circle(img, (int(cu[0]), int(cu[1])), 5, (40, 220, 255), 2,
                       cv2.LINE_AA)


def draw_lesion(img, obs, lt, calib, rollout=None):
    if obs is None:
        return
    tr = [tuple(map(int, c)) for _, c in obs.track_history[-cfg.TRAIL_LEN:]]
    for i in range(1, len(tr)):
        a = i / len(tr)
        cv2.line(img, tr[i - 1], tr[i], (int(200 * a), int(220 * a), 80), 2,
                 cv2.LINE_AA)
    if obs.detected and obs.box_xyxy is not None:
        x1, y1, x2, y2 = map(int, obs.box_xyxy)
        col = C_LESION_M if (obs.class_name or "").startswith("mal") \
            else C_LESION_B
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)
        cx, cy = map(int, obs.center_px)
        cv2.drawMarker(img, (cx, cy), col, cv2.MARKER_CROSS, 18, 2,
                       cv2.LINE_AA)
        lab = f"{obs.class_name or 'lesion'} {obs.conf or 0:.2f}  " \
              f"track {obs.track_id} {obs.track_state or ''}"
        text(img, lab, (x1, max(y1 - 8, 16)), col, 0.45)
        st = lt.state if lt is not None else None
        if st is not None and st.frame == obs.index:
            g = st.geom
            c = _px(calib, g.x, g.z)
            cv2.ellipse(img, c, (int(g.a / calib.mm_per_px_lateral),
                                 int(g.b / calib.mm_per_px_axial)), 0, 0, 360,
                        col, 1, cv2.LINE_AA)
    elif obs.track_state:
        text(img, f"no detection (track {obs.track_id} "
             f"{obs.track_state})", (20, 70), (60, 160, 255), 0.5)
    if obs.pred_center_px is not None:
        p = tuple(map(int, obs.pred_center_px))
        cv2.drawMarker(img, p, C_PRED, cv2.MARKER_TILTED_CROSS, 16, 2,
                       cv2.LINE_AA)
    if rollout is not None and rollout.k > 0:
        prev = tuple(map(int, lt.state.center_px)) if lt.state else None
        for i, ((u, v), r) in enumerate(zip(rollout.points_px,
                                            rollout.radii_mm)):
            a = 1.0 - i / (rollout.k + 1)
            col = (int(C_PRED[0] * a), int(C_PRED[1] * a), int(C_PRED[2] * a))
            pt = (int(u), int(v))
            if prev is not None:
                dashed(img, prev, pt, col, 1, 4)
            cv2.circle(img, pt, 3, col, -1, cv2.LINE_AA)
            if i == rollout.k - 1:
                _dashed_ellipse(img, pt, (int(r / calib.mm_per_px_lateral),
                                          int(r / calib.mm_per_px_axial)), col)
            prev = pt
        text(img, f"LSTM rollout k={rollout.k}", (int(u) + 10, int(v) + 18),
             C_PRED, 0.4)


def draw_legacy(img, legacy):
    if not legacy or "error" in legacy:
        return
    for k in ("shortest", "safest", "recommended"):
        row = legacy.get(k)
        if not row:
            continue
        a = tuple(map(int, row["entry"]))
        b = tuple(map(int, row["target"]))
        cv2.line(img, a, b, C_LEGACY, 1, cv2.LINE_AA)
        cv2.circle(img, a, 4, C_LEGACY, -1, cv2.LINE_AA)
    r = legacy.get("recommended")
    if r:
        text(img, "legacy 2D planner", (int(r["entry"][0]) + 6,
                                         int(r["entry"][1]) + 16), C_LEGACY,
             0.38)


def render_overlay(pane, obs, lt, proc, calib, show_anatomy=True,
                   show_legacy=False):
    img = pane.copy()
    ps = proc.plan_set if proc is not None else None
    an = proc.anatomy if proc is not None else None
    if show_anatomy:
        draw_anatomy(img, an, calib)
    if ps is not None and show_legacy:
        draw_legacy(img, ps.legacy)
    ro = ps.rollout if ps is not None else None
    draw_lesion(img, obs, lt, calib, ro)
    if proc is not None:
        if proc.locked_plan is not None:
            p = proc.locked_plan
            draw_plan(img, p, calib, PATH_COLORS.get(p.kind, C_REC), 3,
                      f"LOCKED {p.kind} {p.insertion_mm:.1f} mm")
        elif ps is not None:
            for kind in ("shortest", "safest", "recommended"):
                p = ps.get(kind)
                if p is None:
                    continue
                sel = kind == proc.kind
                draw_plan(img, p, calib, PATH_COLORS[kind], 4 if sel else 2,
                          f"{kind} {p.insertion_mm:.1f} mm, clear "
                          f"{p.min_clearance_mm:.1f} mm" if sel else None)
        pose = proc.locked_pose or proc.scene.probe_pose()
        draw_needle(img, proc.scene, proc, calib, pose)
    scale_bars(img, calib)
    return img


def render_raw(pane, calib, label="RAW"):
    img = pane.copy()
    scale_bars(img, calib)
    return img
