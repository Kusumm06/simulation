"""
Surgeon console (one OpenCV window, 1920 x 1080, scaled to the screen).

 +------------------------------------------------------------------------+
 | title / source / badges                                                |
 | state machine strip (active state lit)                                 |
 +-----------------------+-----------------------+------------------------+
 | ULTRASOUND (raw)      | AI OVERLAY            | LIVE METRICS           |
 | scale bars            | YOLO, LSTM, anatomy,  |                        |
 |                       | paths, needle         |                        |
 +-----------------------+-----------------------+------------------------+
 | OPERATING ROOM (3D)   | PLANNING (probe plane)| COMPARISON TABLE       |
 |                       | + clearance profile   | EVENT LOG              |
 +-----------------------+-----------------------+------------------------+
 | disclaimer / keys                                                      |
 +------------------------------------------------------------------------+
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from ultrabot_sim import config as cfg
from ultrabot_sim.control import SPECIAL, STATES
from ultrabot_sim.safety import CLEAR, LEVEL_BGR
from ultrabot_sim.us_render import (C_REC, C_SAFE, C_SHORT,
                                    PATH_COLORS, dashed, text)

BG = (16, 18, 22)
PANEL = (24, 27, 33)
EDGE = (52, 58, 68)
FG = (225, 230, 235)
DIM = (140, 150, 160)
ACCENT = (230, 190, 60)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_B = cv2.FONT_HERSHEY_DUPLEX

W, H = 1920, 1080
R1_Y, R1_H = 88, 540
R2_Y, R2_H = 636, 404
COL_W = 742
X0, X1, X2 = 8, 758, 1508
RIGHT_W = W - X2 - 8


def _panel(img, x, y, w, h, title, color=ACCENT):
    cv2.rectangle(img, (x, y), (x + w, y + h), PANEL, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), EDGE, 1)
    cv2.rectangle(img, (x, y), (x + 4, y + 22), color, -1)
    cv2.putText(img, title, (x + 10, y + 16), FONT_B, 0.5, FG, 1, cv2.LINE_AA)


def _fit(img, w, h):
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    out = cv2.resize(img, (max(1, int(iw * s)), max(1, int(ih * s))),
                     interpolation=cv2.INTER_LINEAR)
    return out


def _put(canvas, img, x, y, w, h):
    f = _fit(img, w, h)
    fh, fw = f.shape[:2]
    ox, oy = x + (w - fw) // 2, y + (h - fh) // 2
    canvas[oy:oy + fh, ox:ox + fw] = f
    return ox, oy, fw, fh


def _fmt(v, nd=2, unit=""):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "-"
    return f"{v:.{nd}f}{unit}"


class Console:
    def __init__(self, calib):
        self.calib = calib
        self.canvas = np.zeros((H, W, 3), np.uint8)

    # ------------------------------------------------------------ main
    def compose(self, raw, overlay, or_img, proc, obs, info):
        c = self.canvas
        c[:] = BG
        self._header(c, proc, obs, info)
        self._states(c, proc)
        _panel(c, X0, R1_Y, COL_W, R1_H, "ULTRASOUND (raw video frame)")
        _put(c, raw, X0 + 2, R1_Y + 26, COL_W - 4, R1_H - 30)
        _panel(c, X1, R1_Y, COL_W, R1_H, "AI OVERLAY  YOLO + LSTM + planner "
               "+ needle")
        _put(c, overlay, X1 + 2, R1_Y + 26, COL_W - 4, R1_H - 30)
        self._metrics(c, X2, R1_Y, RIGHT_W, R1_H, proc, obs, info)
        _panel(c, X0, R2_Y, COL_W, R2_H, "OPERATING ROOM (3D)  v/b angle, "
               "o window")
        if or_img is not None:
            _put(c, cv2.cvtColor(or_img, cv2.COLOR_RGB2BGR), X0 + 2,
                 R2_Y + 26, COL_W - 4, R2_H - 30)
            text(c, info.get("or_view", ""), (X0 + 12, R2_Y + R2_H - 12),
                 DIM, 0.42)
        else:
            text(c, info.get("or_status", "3D panel starting..."),
                 (X0 + 20, R2_Y + 60), DIM, 0.5)
        _panel(c, X1, R2_Y, COL_W, R2_H, "PLANNING  probe plane (mm)")
        self._planning(c, X1, R2_Y + 24, COL_W, R2_H - 24, proc, info)
        self._table(c, X2, R2_Y, RIGHT_W, 214, proc)
        self._events(c, X2, R2_Y + 220, RIGHT_W, R2_H - 220, proc, info)
        self._footer(c, info)
        return c

    # ------------------------------------------------------------ header
    def _header(self, c, proc, obs, info):
        cv2.putText(c, "ULTRABOT  |  robotic ultrasound-guided core-needle "
                    "breast biopsy  (3D kinematic simulation)", (14, 30),
                    FONT_B, 0.72, FG, 1, cv2.LINE_AA)
        x = 1200
        for lab, col in info.get("badges", []):
            (tw, th), _ = cv2.getTextSize(lab, FONT_B, 0.5, 1)
            cv2.rectangle(c, (x, 10), (x + tw + 14, 36), col, -1)
            cv2.putText(c, lab, (x + 7, 29), FONT_B, 0.5, (15, 15, 15), 1,
                        cv2.LINE_AA)
            x += tw + 22
        src = info.get("source", "")
        cv2.putText(c, src, (14, 52), FONT, 0.45, DIM, 1, cv2.LINE_AA)

    def _states(self, c, proc):
        y = 58
        x = 14
        names = STATES + SPECIAL
        w = (W - 28 - 6 * len(names)) / len(names)
        for n in names:
            on = proc is not None and proc.state == n
            col = (40, 170, 240) if n in SPECIAL else (70, 200, 120)
            if n == "E_STOP":
                col = (60, 60, 235)
            x0, x1 = int(x), int(x + w)
            cv2.rectangle(c, (x0, y), (x1, y + 24), col if on else PANEL, -1)
            cv2.rectangle(c, (x0, y), (x1, y + 24), col, 1)
            lab = n.replace("_", " ")
            fs = 0.38
            (tw, th), _ = cv2.getTextSize(lab, FONT, fs, 1)
            cv2.putText(c, lab, (x0 + max(2, (x1 - x0 - tw) // 2), y + 16),
                        FONT, fs, (15, 15, 15) if on else DIM, 1,
                        cv2.LINE_AA)
            x += w + 6

    # ------------------------------------------------------------ metrics
    def _metrics(self, c, x, y, w, h, proc, obs, info):
        _panel(c, x, y, w, h, "LIVE METRICS")
        r = proc.report if proc is not None else None
        lvl = r.level if r else CLEAR
        cv2.rectangle(c, (x + 10, y + 30), (x + w - 10, y + 62),
                      LEVEL_BGR[lvl], -1)
        cv2.putText(c, f"SAFETY: {lvl}", (x + 20, y + 54), FONT_B, 0.7,
                    (15, 15, 15), 1, cv2.LINE_AA)
        rows = [
            ("state", proc.state if proc else "-"),
            ("path / planner", f"{proc.kind} / {cfg.DRIVE_PLANNER}"
             if proc else "-"),
            ("LSTM", obs.lstm_status if obs is not None else "-"),
            ("lesion", f"{obs.class_name} {obs.conf:.2f}" if obs is not None
             and obs.detected else "not detected"),
            ("frame / fps", f"{obs.index if obs else '-'} / "
             f"{info.get('loop_fps', 0):.0f} Hz"),
            ("YOLO / planner", f"{info.get('yolo_ms', 0):.0f} ms / "
             f"{info.get('plan_ms', 0):.0f} ms"),
            ("IK error", _fmt(proc.ik_err_mm if proc else None, 4, " mm")),
            ("axis error", _fmt(proc.axis_err_deg if proc else None, 4,
                                " deg")),
            ("skin RCM dev.", _fmt(proc.rcm_dev_mm if proc else None, 4,
                                   " mm")),
            ("out of plane", _fmt(proc.out_of_plane_mm if proc else None, 3,
                                  " mm")),
            ("tip depth", _fmt(proc.s if proc and proc.locked else None, 2,
                               " mm")),
            ("tip -> aim", _fmt((proc.s_target - proc.s) if proc and
                                proc.locked else None, 2, " mm")),
            ("pred. miss / offset", f"{_fmt(proc.predicted_miss_mm, 1)} / "
             f"{_fmt(proc.aim_offset_mm, 1)} mm" if proc else "-"),
            ("needle clearance", (f"{_fmt(r.needle_clearance_mm, 1)} mm "
                                  f"({r.clearance_struct})")
             if r and r.needle_clearance_mm is not None else "-"),
            ("arm-arm", _fmt(proc._arm_clear.get("arm_arm", None) * 100
                             if proc and proc._arm_clear else None, 1,
                             " cm")),
            ("speed / weights", f"{proc.speed:.0f} mm/s / {proc.preset}"
             if proc else "-"),
            ("US scale", f"{self.calib.mm_per_px_lateral:.4f} x "
             f"{self.calib.mm_per_px_axial:.4f} mm/px"),
            ("time", f"{proc.t:.1f} s" if proc else "-"),
        ]
        yy = y + 86
        for k, v in rows:
            cv2.putText(c, k, (x + 14, yy), FONT, 0.46, DIM, 1, cv2.LINE_AA)
            cv2.putText(c, str(v), (x + 170, yy), FONT, 0.46, FG, 1,
                        cv2.LINE_AA)
            yy += 23
        if r and r.messages:
            col = LEVEL_BGR[r.level]
            for m in r.messages[-2:]:
                cv2.putText(c, m[:52], (x + 14, yy + 4), FONT, 0.42, col, 1,
                            cv2.LINE_AA)
                yy += 19

    # ------------------------------------------------------------ planning
    def _planning(self, c, x, y, w, h, proc, info):
        if proc is None or proc.anatomy is None:
            text(c, "waiting for a detected lesion...", (x + 20, y + 40),
                 DIM, 0.5)
            return
        an = proc.anatomy
        ps = proc.plan_set
        pw = int(w * 0.62)
        X_MIN, X_MAX = -95.0, 95.0
        Z_MIN = -12.0
        Z_MAX = max(an.pleura_depth(0.0) + 4, 45.0)
        sx = (pw - 20) / (X_MAX - X_MIN)
        sz = (h - 20) / (Z_MAX - Z_MIN)
        s = min(sx, sz)
        ox = x + 10 - X_MIN * s
        oz = y + 10 - Z_MIN * s

        def P(px, pz):
            return int(ox + px * s), int(oz + pz * s)
        xs = np.linspace(X_MIN, X_MAX, 120)
        # layers
        skin = [P(v, float(an.skin_depth(v))) for v in xs]
        fat = [P(v, float(an.fat_bottom(v))) for v in xs]
        fas = [P(v, float(an.fascia_depth(v))) for v in xs]
        mus = [P(v, float(an.muscle_bottom(v))) for v in xs]
        ple = [P(v, float(an.pleura_depth(v))) for v in xs]
        cv2.fillPoly(c, [np.array(skin + fas[::-1], np.int32)], (52, 64, 78))
        cv2.fillPoly(c, [np.array(skin + fat[::-1], np.int32)], (60, 92, 110))
        cv2.fillPoly(c, [np.array(fas + mus[::-1], np.int32)], (50, 50, 110))
        cv2.polylines(c, [np.array(skin, np.int32)], False, (150, 175, 210), 2,
                      cv2.LINE_AA)
        cv2.polylines(c, [np.array(fas, np.int32)], False, (70, 110, 200), 2,
                      cv2.LINE_AA)
        cv2.polylines(c, [np.array(ple, np.int32)], False, (220, 220, 220), 1,
                      cv2.LINE_AA)
        for (rx, rz, rr) in an.ribs:
            if X_MIN < rx < X_MAX:
                cv2.circle(c, P(rx, rz), max(2, int(rr * s)), (210, 210, 210),
                           1, cv2.LINE_AA)
        for (a, b) in an.ligaments:
            cv2.line(c, P(*a), P(*b), (170, 170, 175), 1, cv2.LINE_AA)
        for v in an.vessels:
            cv2.circle(c, P(v.x, v.z), max(2, int(v.r * s)), (60, 60, 220), -1,
                       cv2.LINE_AA)
            cv2.circle(c, P(v.x, v.z), int((v.r + cfg.VESSEL_MARGIN_MM) * s),
                       (60, 60, 220), 1, cv2.LINE_AA)
        # nipple zone and probe footprint
        nx = cfg.NIPPLE_X_MM
        cv2.line(c, P(nx - cfg.NIPPLE_R_MM, float(an.skin_depth(nx)) - 2),
                 P(nx + cfg.NIPPLE_R_MM, float(an.skin_depth(nx)) - 2),
                 (190, 120, 230), 3, cv2.LINE_AA)
        text(c, "nipple", P(nx - 8, float(an.skin_depth(nx)) - 5),
             (190, 120, 230), 0.35, bg=False)
        hw = self.calib.half_width_mm
        cv2.rectangle(c, P(-hw - 3, -10), P(hw + 3, 0), (200, 205, 210), -1)
        text(c, "probe", P(-8, -3), (30, 30, 30), 0.35, bg=False)
        # image region
        cv2.rectangle(c, P(-hw, 0), P(hw, self.calib.depth_mm), (110, 120, 130),
                      1)
        # candidates
        if ps is not None:
            for cand in ps.candidates[::2]:
                a = P(*cand["entry"])
                b = P(*cand["fired"])
                if cand["feasible"]:
                    cv2.line(c, a, b, (70, 110, 70), 1, cv2.LINE_AA)
                else:
                    dashed(c, a, b, (60, 60, 95), 1, 6)
        # lesion
        if proc.lt.state is not None:
            g = proc.lt.state.geom
            cv2.ellipse(c, P(g.x, g.z), (max(2, int(g.a * s)),
                                         max(2, int(g.b * s))), 0, 0, 360,
                        (110, 230, 110), 2, cv2.LINE_AA)
        # paths
        plans = []
        if proc.locked_plan is not None:
            plans = [proc.locked_plan]
        elif ps is not None:
            plans = [ps.get(k) for k in ("shortest", "safest", "recommended")]
        for p in plans:
            if p is None:
                continue
            col = PATH_COLORS.get(p.kind, C_REC)
            th = 3 if p.kind == proc.kind else 2
            cv2.line(c, P(*p.entry_mm), P(*p.fired_mm), col, th, cv2.LINE_AA)
            cv2.circle(c, P(*p.entry_mm), 4, col, -1, cv2.LINE_AA)
            cv2.drawMarker(c, P(*p.aim_mm), col, cv2.MARKER_CROSS, 10, 1,
                           cv2.LINE_AA)
        # needle (actual)
        sc = proc.scene
        tip, d = sc.needle_axis()
        pose = proc.locked_pose or sc.probe_pose()
        f, d, _ = sc.needle_frame()
        hub = f + d * cfg.GUN_LEN
        s_tip = hub + d * ((cfg.NEEDLE_LEN_MM + proc.needle.stylet_ext) /
                           1000.0)
        a = pose.world_to_probe(hub)
        b = pose.world_to_probe(s_tip)
        if abs(a[1]) < 30 and abs(b[1]) < 30:
            ok, pa, pb = cv2.clipLine((x, y, pw, h), P(a[0], a[2]),
                                      P(b[0], b[2]))
            if ok:
                cv2.line(c, pa, pb, (250, 250, 250), 2, cv2.LINE_AA)
            tb = P(b[0], b[2])
            if x <= tb[0] < x + pw and y <= tb[1] < y + h:
                cv2.circle(c, tb, 3, (255, 255, 255), -1, cv2.LINE_AA)
        # final needle positions of finished runs (compare mode)
        for r in info.get("finals", []):
            f = r.fire
            pl = r.plan or {}
            if not f or not pl.get("entry_mm"):
                continue
            col = PATH_COLORS.get(r.kind, C_REC)
            e = P(*pl["entry_mm"])
            t = P(*f["fired_tip_mm"])
            dashed(c, e, t, col, 2, 5)
            cv2.drawMarker(c, t, col, cv2.MARKER_DIAMOND, 10, 2, cv2.LINE_AA)
        # legend
        lx = x + 12
        for lab, col in (("shortest", C_SHORT), ("safest", C_SAFE),
                         ("recommended", C_REC)):
            cv2.line(c, (lx, y + h - 12), (lx + 18, y + h - 12), col, 3)
            cv2.putText(c, lab, (lx + 22, y + h - 8), FONT, 0.4, FG, 1,
                        cv2.LINE_AA)
            lx += 110
        if ps is not None:
            cv2.putText(c, f"{ps.n_feasible}/{ps.n_candidates} feasible | "
                        f"{ps.aim_source}", (x + 12, y + 14), FONT, 0.4, DIM,
                        1, cv2.LINE_AA)
        # clearance profile of the active path
        p = proc.locked_plan or (ps.get(proc.kind) if ps else None)
        self._profile(c, x + pw + 4, y + 4, w - pw - 12, h * 0.62, p)
        yy = int(y + h * 0.62 + 24)
        cv2.putText(c, "recommended cost (min-max normalised):", (x + pw + 6,
                    yy), FONT, 0.38, DIM, 1, cv2.LINE_AA)
        for line in ("w_len*len + w_clear*(1-clear)", "+ w_angle*angle + "
                     "w_lig*lig", "+ w_motion*motion + w_target*err"):
            yy += 16
            cv2.putText(c, line, (x + pw + 6, yy), FONT, 0.38, FG, 1,
                        cv2.LINE_AA)
        if ps is not None and ps.notes:
            for n in ps.notes[:2]:
                yy += 18
                cv2.putText(c, n[:44], (x + pw + 6, yy), FONT, 0.34,
                            (120, 190, 230), 1, cv2.LINE_AA)

    def _profile(self, c, x, y, w, h, p):
        x, y, w, h = int(x), int(y), int(w), int(h)
        cv2.rectangle(c, (x, y), (x + w, y + h), (30, 34, 40), -1)
        cv2.putText(c, "clearance along the path", (x + 6, y + 14), FONT,
                    0.4, FG, 1, cv2.LINE_AA)
        if p is None or not p.profile_s:
            return
        s = np.asarray(p.profile_s)
        v = np.asarray(p.profile_c)
        vmax = max(15.0, float(np.nanmax(v[np.isfinite(v)])) if
                   np.isfinite(v).any() else 15.0)
        vmax = min(vmax, 40.0)
        X = x + 30 + (s / max(s.max(), 1e-6)) * (w - 40)
        Y = y + h - 18 - np.clip(v / vmax, 0, 1) * (h - 40)
        m = cfg.VESSEL_MARGIN_MM if p.min_clearance_struct == "vessel" else \
            cfg.CHESTWALL_CLEARANCE_MM
        ym = int(y + h - 18 - m / vmax * (h - 40))
        cv2.line(c, (x + 30, ym), (x + w - 8, ym), (60, 60, 235), 1)
        cv2.putText(c, f"margin {m:.0f}", (x + w - 70, ym - 4), FONT, 0.34,
                    (60, 60, 235), 1, cv2.LINE_AA)
        pts = np.column_stack([X, Y]).astype(np.int32)
        cv2.polylines(c, [pts], False, PATH_COLORS.get(p.kind, C_REC), 2,
                      cv2.LINE_AA)
        cv2.putText(c, "0", (x + 20, y + h - 14), FONT, 0.32, DIM, 1)
        cv2.putText(c, f"{vmax:.0f} mm", (x + 2, y + 34), FONT, 0.32, DIM, 1)
        cv2.putText(c, f"{s.max():.0f} mm along needle", (x + w - 130,
                    y + h - 4), FONT, 0.32, DIM, 1)
        cv2.putText(c, f"min {p.min_clearance_mm:.1f} mm ({p.min_clearance_struct})",
                    (x + 34, y + 30), FONT, 0.36, FG, 1, cv2.LINE_AA)

    # ------------------------------------------------------------ table
    def _table(self, c, x, y, w, h, proc):
        _panel(c, x, y, w, h, "SHORTEST vs SAFEST vs RECOMMENDED")
        ps = (proc.plan_set_full or proc.plan_set) if proc else None
        cols = ["shortest", "safest", "recommended"]
        rows = [("insertion (mm)", "insertion_mm", 1),
                ("angle chest wall", "angle_chestwall_deg", 1),
                ("min clear. (mm)", "min_clearance_mm", 1),
                ("ligaments", "ligaments_crossed", 0),
                ("motion sens. (mm)", "motion_sensitivity_mm", 1),
                ("pred. miss (mm)", "predicted_miss_mm", 1),
                ("cost", "total_cost", 2)]
        cx = [x + 10, x + 170, x + 250, x + 330]
        yy = y + 44
        for i, k in enumerate(cols):
            cv2.putText(c, k[:5] if k != "recommended" else "recom.",
                        (cx[i + 1], yy), FONT, 0.42, PATH_COLORS[k], 1,
                        cv2.LINE_AA)
        for lab, key, nd in rows:
            yy += 22
            cv2.putText(c, lab, (cx[0], yy), FONT, 0.42, DIM, 1, cv2.LINE_AA)
            for i, k in enumerate(cols):
                p = ps.get(k) if ps else None
                v = getattr(p, key, None) if p is not None else None
                s = "-" if v is None else (f"{v:.{nd}f}" if nd else f"{v}")
                if key == "min_clearance_mm" and p is not None:
                    s += p.min_clearance_struct[0]
                cv2.putText(c, s, (cx[i + 1], yy), FONT, 0.42, FG, 1,
                            cv2.LINE_AA)

    def _events(self, c, x, y, w, h, proc, info):
        _panel(c, x, y, w, h, "EVENTS / LAST COMMAND")
        yy = y + 42
        lc = info.get("last_cmd")
        if lc:
            cv2.putText(c, f"last command: {lc}", (x + 12, yy), FONT, 0.44,
                        ACCENT, 1, cv2.LINE_AA)
            yy += 20
        ev = proc.events[-6:] if proc else []
        for t, kind, msg in ev:
            col = (60, 60, 235) if kind in ("safety",) and "DANGER" in msg \
                else (40, 190, 250) if kind == "safety" else FG
            s = f"{t:6.1f}  {msg}"
            for k in range(0, min(len(s), 102), 51):
                cv2.putText(c, s[k:k + 51], (x + 12, yy), FONT, 0.38, col, 1,
                            cv2.LINE_AA)
                yy += 16
            if yy > y + h - 8:
                break

    def _footer(self, c, info):
        cv2.rectangle(c, (0, H - 34), (W, H), (10, 11, 14), -1)
        cv2.putText(c, cfg.DISCLAIMER, (14, H - 12), FONT, 0.5,
                    (120, 190, 240), 1, cv2.LINE_AA)
        cv2.putText(c, "1/2/3 path  p plan  enter start  c compare  space "
                    "pause  z E-stop  r resume  +/- speed  [/] weights  L "
                    "legacy  a anatomy  t slow-mo  s shot  q quit",
                    (720, H - 12), FONT, 0.42, DIM, 1, cv2.LINE_AA)
