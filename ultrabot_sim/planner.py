"""
Anatomy-aware needle path planning: shortest, safest and recommended.

The biopsy needle is RIGID, so inside the tissue its path is a straight line
from a skin entry point. Everything here is computed in millimetres in the
probe frame (in-plane technique: the needle stays in the image plane,
y = 0) and then mapped to the world for the robot.

Target
    T_now   tracked lesion centre (YOLO + tracker)
    T_pred  LSTM autoregressive rollout to the expected firing time
            (k = ceil(t_fire / frame_dt), capped at LSTM_MAX_ROLLOUT).
    The throw is centred on the aim point: pre-fire tip = aim - THROW/2,
    fired tip = aim + THROW/2 along the needle axis.

Entries
    On the skin, OUTSIDE the probe footprint, on both sides, from
    ENTRY_MIN_OFFSET_MM to ENTRY_MAX_OFFSET_MM beyond the probe edge, every
    ENTRY_STEP_MM. Never from the image bottom / through the chest wall.

Hard constraints (a candidate failing any of them is rejected, with reason)
    entry outside the nipple-areola region; puncture not tangential to the
    skin; angle to the chest wall <= MAX_ANGLE_TO_CHESTWALL_DEG; insertion
    >= MIN_INSERTION_MM; insertion + throw <= needle - grip margin; gun body
    clear of the probe and of the skin; minimum clearance of the WHOLE
    segment (throw included) to vessels / chest wall / ribs / pleura >= its
    margin; the robot can reach it (IK < 0.1 mm, joint margin, link
    clearance), checked on the selected paths.

Selection
    shortest     minimum in-tissue length (entry -> pre-fire tip), ties:
                 smaller angle to the chest wall
    safest       maximum MINIMUM clearance along the whole segment
                 (max-min), ties: angle, ligaments crossed, motion
                 sensitivity
    recommended  weighted cost of min-max normalised terms (same
                 transparent style as ultrabot.biopsy):
                 w_len*length + w_clear*(1 - clearance) + w_angle*angle
                 + w_lig*ligaments + w_motion*motion + w_target*target_err
The best grid candidates are refined with scipy.optimize (SLSQP, bounded,
on entry offset and a small angle change) and kept only if still feasible
and better.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ultrabot_sim import config as cfg
from ultrabot_sim.anatomy import Anatomy

FORMULA = ("cost = w_len*len + w_clear*(1-clear) + w_angle*angle + "
           "w_lig*lig + w_motion*motion + w_target*target_err  "
           "(each term min-max normalised over feasible candidates)")

REASON_ORDER = ["nipple region", "tangential puncture", "angle to chest wall",
                "too short", "needle too short", "gun hits probe",
                "gun touches skin", "vessel clearance", "chest wall clearance",
                "rib clearance", "pleura clearance", "robot reach (IK)"]


@dataclass
class NeedlePlan:
    kind: str
    side: int
    offset_mm: float
    entry_mm: tuple
    direction: tuple
    aim_mm: tuple
    prefire_mm: tuple
    fired_mm: tuple
    lesion_now_mm: tuple
    lesion_pred_mm: tuple
    insertion_mm: float
    angle_chestwall_deg: float
    skin_angle_deg: float
    min_clearance_mm: float
    min_clearance_struct: str
    clearance_by_family: dict
    profile_s: list
    profile_c: list
    ligaments_crossed: int
    motion_sensitivity_mm: float
    predicted_miss_mm: float
    target_error_mm: float
    aim_offset_mm: float
    core_radius_mm: float
    feasible: bool
    reasons: list
    cost_terms: dict = field(default_factory=dict)
    total_cost: Optional[float] = None
    ik: Optional[dict] = None
    refined: Optional[dict] = None
    entry_px: Optional[tuple] = None
    world: dict = field(default_factory=dict)

    @property
    def length_mm(self):
        return self.insertion_mm

    def point(self, s_mm):
        """Probe-frame point (x, z) at signed distance s from the entry."""
        return (self.entry_mm[0] + self.direction[0] * s_mm,
                self.entry_mm[1] + self.direction[1] * s_mm)

    def attach_world(self, pose, calib):
        d3 = np.array([self.direction[0], 0.0, self.direction[1]])
        dw = pose.dir_probe_to_world(d3)

        def w(p):
            return pose.probe_to_world(np.array([p[0], 0.0, p[1]]))
        pre_entry = (self.entry_mm[0] - self.direction[0] * cfg.PRE_ENTRY_MM,
                     self.entry_mm[1] - self.direction[1] * cfg.PRE_ENTRY_MM)
        self.world = {"entry": w(self.entry_mm), "prefire": w(self.prefire_mm),
                      "fired": w(self.fired_mm), "aim": w(self.aim_mm),
                      "pre_entry": w(pre_entry), "dir": dw / np.linalg.norm(dw)}
        self.entry_px = tuple(float(v) for v in
                              calib.mm_to_px(*self.entry_mm))
        return self

    def to_dict(self):
        d = {k: getattr(self, k) for k in (
            "kind", "side", "offset_mm", "entry_mm", "direction", "aim_mm",
            "prefire_mm", "fired_mm", "lesion_now_mm", "lesion_pred_mm",
            "insertion_mm", "angle_chestwall_deg", "skin_angle_deg",
            "min_clearance_mm", "min_clearance_struct", "clearance_by_family",
            "ligaments_crossed", "motion_sensitivity_mm", "predicted_miss_mm",
            "target_error_mm", "aim_offset_mm", "core_radius_mm",
            "feasible", "reasons", "cost_terms",
            "total_cost", "ik", "refined", "entry_px")}
        d["world"] = {k: np.asarray(v).round(6).tolist()
                      for k, v in self.world.items()}
        return _jsonable(d)


@dataclass
class PlanSet:
    shortest: Optional[NeedlePlan]
    safest: Optional[NeedlePlan]
    recommended: Optional[NeedlePlan]
    candidates: list
    n_candidates: int
    n_feasible: int
    reject_counts: dict
    weights: dict
    preset: str
    aim_source: str
    k_rollout: int
    rollout: object
    plan_ms: float
    anatomy: Anatomy
    frame: int
    track_id: Optional[int]
    legacy: Optional[dict] = None
    notes: list = field(default_factory=list)

    def get(self, kind):
        return {"shortest": self.shortest, "safest": self.safest,
                "recommended": self.recommended}.get(kind)

    def summary(self):
        return _jsonable({
            "frame": self.frame, "track_id": self.track_id,
            "aim_source": self.aim_source, "k_rollout": self.k_rollout,
            "plan_ms": self.plan_ms, "n_candidates": self.n_candidates,
            "n_feasible": self.n_feasible, "reject_counts": self.reject_counts,
            "weights": self.weights, "preset": self.preset, "formula": FORMULA,
            "notes": self.notes,
            "anatomy": self.anatomy.to_dict(),
            "shortest": self.shortest.to_dict() if self.shortest else None,
            "safest": self.safest.to_dict() if self.safest else None,
            "recommended": self.recommended.to_dict()
            if self.recommended else None,
        })


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


# ------------------------------------------------------------- evaluation
def _seg_rect_distance(a, b, rect, n=24):
    """Distance from segment a->b (N, 2 each) to an axis-aligned rectangle
    (x0, x1, z0, z1), sampled."""
    t = np.linspace(0, 1, n)[None, :, None]
    P = a[:, None, :] + (b - a)[:, None, :] * t
    x0, x1, z0, z1 = rect
    dx = np.maximum(np.maximum(x0 - P[..., 0], 0), P[..., 0] - x1)
    dz = np.maximum(np.maximum(z0 - P[..., 1], 0), P[..., 1] - z1)
    return np.hypot(dx, dz).min(axis=1)


def evaluate(an: Anatomy, E, d, aim, t_now, t_pred, motion_vec, sigma):
    """
    Evaluate N candidate lines (vectorised).
    E (N,2) entry, d (N,2) unit direction, aim (N,2) throw centre target
    (projected onto the line), t_now / t_pred (2,) lesion now / at firing,
    motion_vec (2,) predicted displacement, sigma rollout uncertainty (mm).
    Returns a dict of arrays + per-candidate reason lists.
    """
    E = np.atleast_2d(np.asarray(E, float))
    d = np.atleast_2d(np.asarray(d, float))
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    aim = np.atleast_2d(np.asarray(aim, float))
    N = len(E)
    T = cfg.THROW_MM
    proj = np.einsum("ij,ij->i", aim - E, d)
    C = E + d * proj[:, None]                       # throw centre on line
    P = C - d * (T / 2.0)                           # pre-fire tip
    F = C + d * (T / 2.0)                           # fired tip
    L = np.einsum("ij,ij->i", P - E, d)             # in-tissue insertion
    nperp = np.column_stack([-d[:, 1], d[:, 0]])
    # angles
    f = an.fascia_dir()
    ang_cw = np.degrees(np.arccos(np.clip(np.abs(d @ f), 0, 1)))
    slope = an.skin_slope(E[:, 0])
    n_in = np.column_stack([-slope, np.ones(N)])
    n_in /= np.linalg.norm(n_in, axis=1, keepdims=True)
    skin_ang = np.degrees(np.arcsin(np.clip(np.einsum("ij,ij->i", d, n_in),
                                            -1, 1)))
    # exact minimum clearance of the segment E -> F to each family:
    # half-planes (chest wall, pleura) are linear along the segment, so the
    # minimum is at an end point; tubes (vessels, ribs) use the exact
    # point-to-segment distance.
    s_max = np.maximum(L + T, 0.0)
    fam_min = segment_clearance(an, E, d, s_max)
    fams = list(fam_min.keys())
    M = np.vstack([fam_min[f] for f in fams])
    minc = M.min(axis=0)
    which = np.array(fams)[np.argmin(M, axis=0)]
    slack_all = np.vstack([fam_min[f] - an.margins[f] for f in fams]
                          ).min(axis=0)
    # gun / probe / skin
    hub = P - d * cfg.NEEDLE_LEN_MM
    gun_end = hub - d * (cfg.GUN_LEN * 1000.0)
    bx, _, bz = (np.array(cfg.PROBE_BODY) * 1000.0)
    rect = (-bx / 2, bx / 2, -bz, 0.0)
    gun_probe = _seg_rect_distance(hub, gun_end, rect) - cfg.GUN_RADIUS * 1000
    shaft_probe = _seg_rect_distance(E, hub, rect) - cfg.NEEDLE_OD_MM / 2
    tt = np.linspace(0, 1, 16)[None, :, None]
    G = hub[:, None, :] + (gun_end - hub)[:, None, :] * tt
    gun_skin = (an.skin_depth(G[..., 0]) - G[..., 1]).min(axis=1) - \
        cfg.GUN_RADIUS * 1000
    # lesion motion
    rel_now = t_now[None, :] - E
    rel_pred = t_pred[None, :] - E
    target_err = np.abs(np.einsum("ij,ij->i", rel_now, nperp))
    perp_pred = np.abs(np.einsum("ij,ij->i", rel_pred, nperp))
    axial_pred = np.einsum("ij,ij->i", t_pred[None, :] - C, d)
    over = np.maximum(np.abs(axial_pred) - T / 2.0, 0.0)
    aim_offset = np.hypot(perp_pred, over)
    # predicted miss = how far the throw passes outside the lesion CORE
    # (CORE_FRACTION x the lesion half-size across the needle)
    if an.lesion is not None:
        ga, gb = an.lesion.a, an.lesion.b
        r_core = cfg.CORE_FRACTION / np.sqrt((nperp[:, 0] / ga) ** 2 +
                                             (nperp[:, 1] / gb) ** 2)
    else:
        r_core = np.zeros(N)
    pred_miss = np.maximum(aim_offset - r_core, 0.0)
    motion = np.abs(nperp @ np.asarray(motion_vec, float)) + sigma
    lig = np.array([an.ligament_crossings(E[i], F[i]) for i in range(N)])
    nip = an.in_nipple(E[:, 0])
    reasons = []
    for i in range(N):
        r = []
        if nip[i]:
            r.append("nipple region")
        if skin_ang[i] < cfg.MIN_SKIN_ANGLE_DEG:
            r.append("tangential puncture")
        if ang_cw[i] > cfg.MAX_ANGLE_TO_CHESTWALL_DEG:
            r.append("angle to chest wall")
        if L[i] < cfg.MIN_INSERTION_MM:
            r.append("too short")
        if L[i] + T > cfg.NEEDLE_LEN_MM - cfg.GRIP_MARGIN_MM:
            r.append("needle too short")
        if gun_probe[i] < cfg.PROBE_CLEARANCE_MM or shaft_probe[i] < 1.0:
            r.append("gun hits probe")
        if gun_skin[i] < 2.0:
            r.append("gun touches skin")
        for fam in fams:
            if fam_min[fam][i] < an.margins[fam]:
                r.append(f"{fam} clearance")
        reasons.append(r)
    return {"E": E, "d": d, "C": C, "P": P, "F": F, "L": L, "ang_cw": ang_cw,
            "skin_ang": skin_ang, "minc": minc, "which": which,
            "fam_min": fam_min, "s_max": s_max,
            "target_err": target_err, "pred_miss": pred_miss,
            "aim_offset": aim_offset, "r_core": r_core,
            "motion": motion, "lig": lig, "reasons": reasons,
            "gun_probe": gun_probe, "gun_skin": gun_skin,
            "slack": slack_all}


def segment_clearance(an: Anatomy, E, d, length):
    """Exact min clearance (mm) of segments E + s d, s in [0, length], to
    each hard-structure family. Returns {family: (N,)}."""
    E = np.atleast_2d(E)
    d = np.atleast_2d(d)
    length = np.atleast_1d(length)
    F = E + d * length[:, None]
    out = {}
    k = an.fascia_slope
    nrm = math.sqrt(1 + k * k)

    def plane(z0, X, Z):
        return (z0 + k * X - Z) / nrm
    out["chest wall"] = np.minimum(plane(an.fascia_z0, E[:, 0], E[:, 1]),
                                   plane(an.fascia_z0, F[:, 0], F[:, 1]))

    def tubes(circles):
        Q = np.asarray([(c[0], c[1]) for c in circles], float)   # (M, 2)
        R = np.asarray([c[2] for c in circles], float)
        rel = Q[None, :, :] - E[:, None, :]                      # (N, M, 2)
        t = np.clip(np.einsum("nmk,nk->nm", rel, d), 0.0,
                    length[:, None])
        close = E[:, None, :] + d[:, None, :] * t[..., None]
        dist = np.linalg.norm(Q[None] - close, axis=2) - R[None, :]
        return dist.min(axis=1)
    if an.vessels:
        out["vessel"] = tubes([(v.x, v.z, v.r) for v in an.vessels])
    if an.ribs:
        out["rib"] = tubes(an.ribs)
    zp = an.pleura_depth(0.0)
    out["pleura"] = np.minimum(plane(zp, E[:, 0], E[:, 1]),
                               plane(zp, F[:, 0], F[:, 1]))
    return out


def clearance_profile(an: Anatomy, E, d, length, step=None):
    """Sampled clearance along one segment (for the console chart)."""
    step = step or cfg.SAMPLE_STEP_MM
    s = np.arange(0.0, float(length) + 1e-9, step)
    if s[-1] < length:
        s = np.append(s, length)
    pts = np.asarray(E)[None, :] + np.asarray(d)[None, :] * s[:, None]
    c, fam, _ = an.min_clearance(pts)
    return s, c, fam


# ---------------------------------------------------------------- planner
class Planner:
    def __init__(self, calib):
        self.calib = calib

    def entries(self, an: Anatomy):
        W2 = self.calib.half_width_mm
        offs = np.arange(cfg.ENTRY_MIN_OFFSET_MM,
                         cfg.ENTRY_MAX_OFFSET_MM + 1e-9, cfg.ENTRY_STEP_MM)
        out = []
        for s in (-1, 1):
            x = s * (W2 + offs)
            z = an.skin_depth(x)
            out += [(s, o, xi, zi) for o, xi, zi in zip(offs, x, z)]
        return out

    def plan(self, lesion_state, anatomy: Anatomy, lesion_tracker=None,
             pose=None, ik_checker=None, preset=None, t_before_insert=0.0,
             frame_dt=0.2, legacy=True, speed_mm_s=None, side_lock=None):
        t0 = time.perf_counter()
        self._an = anatomy
        preset = preset or cfg.WEIGHT_PRESET
        W = dict(cfg.WEIGHT_PRESETS[preset])
        speed = speed_mm_s or cfg.INSERT_SPEED_MM_S
        notes = []
        t_now = np.array(lesion_state.center_mm, float)
        # ---- time of arrival -> rollout length
        L_guess = 35.0
        t_fire = t_before_insert + (cfg.PRE_ENTRY_MM + L_guess) / speed
        k = int(math.ceil(t_fire / max(frame_dt, 1e-3)))
        k = max(1, min(k, cfg.LSTM_MAX_ROLLOUT))
        ro = lesion_tracker.rollout(k) if lesion_tracker is not None else None
        if ro is not None and ro.k > 0:
            t_pred = np.array(ro.points_mm[-1], float)
            sigma = ro.radii_mm[-1]
            aim_src = f"LSTM rollout k={ro.k}"
        else:
            t_pred = t_now.copy()
            sigma = (lesion_tracker.uncertainty_mm(1)
                     if lesion_tracker is not None else cfg.SAFETY_MARGIN_MM)
            aim_src = "current centre (LSTM not ready)"
            notes.append("LSTM not available: aiming at the current lesion "
                         "centre, motion sensitivity = uncertainty only")
            k = 0
        motion_vec = t_pred - t_now
        aim = t_pred
        # ---- grid of entries
        ent = self.entries(anatomy)
        if side_lock in (-1, 1):
            # committed to one side of the probe (needle already on its way)
            ent = [e for e in ent if e[0] == side_lock]
            notes.append(f"side locked to {'+x' if side_lock > 0 else '-x'}")
        sides = np.array([e[0] for e in ent])
        offs = np.array([e[1] for e in ent])
        E = np.array([[e[2], e[3]] for e in ent])
        d = aim[None, :] - E
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        ev = evaluate(anatomy, E, d, np.repeat(aim[None, :], len(E), 0),
                      t_now, t_pred, motion_vec, sigma)
        feas = np.array([len(r) == 0 for r in ev["reasons"]])
        reject_counts = {}
        for r in ev["reasons"]:
            if r:
                reject_counts[r[0]] = reject_counts.get(r[0], 0) + 1
        cand_light = [{"entry": tuple(ev["E"][i]), "fired": tuple(ev["F"][i]),
                       "feasible": bool(feas[i]),
                       "reason": ev["reasons"][i][0] if ev["reasons"][i]
                       else ""} for i in range(len(E))]
        plans = {"shortest": None, "safest": None, "recommended": None}
        norm = None
        idx_f = np.where(feas)[0]
        if len(idx_f):
            norm = self._norm_ranges(ev, idx_f)
            costs = self._costs(ev, norm, W)
            orders = {
                "shortest": sorted(idx_f, key=lambda i: (round(ev["L"][i], 6),
                                                         ev["ang_cw"][i])),
                "safest": sorted(idx_f, key=lambda i: (
                    -round(ev["minc"][i], 6), ev["ang_cw"][i], ev["lig"][i],
                    ev["motion"][i])),
                "recommended": sorted(idx_f, key=lambda i: (costs[i],
                                                            ev["L"][i])),
            }
            for kind, order in orders.items():
                chosen = None
                for i in order[:4]:
                    p = self._make_plan(kind, ev, i, sides[i], offs[i],
                                        t_now, t_pred, norm, W, costs)
                    p = self._refine(kind, p, anatomy, aim, t_now, t_pred,
                                     motion_vec, sigma, norm, W)
                    if pose is not None:
                        p.attach_world(pose, self.calib)
                    if ik_checker is not None and pose is not None:
                        p.ik = ik_checker(p)
                        if not p.ik.get("ok", False):
                            reject_counts["robot reach (IK)"] = \
                                reject_counts.get("robot reach (IK)", 0) + 1
                            cand_light[i]["feasible"] = False
                            cand_light[i]["reason"] = "robot reach (IK)"
                            continue
                    chosen = p
                    break
                plans[kind] = chosen
        n_feas = sum(1 for c in cand_light if c["feasible"])
        legacy_out = None
        if legacy:
            legacy_out = legacy_plan(lesion_state)
        ps = PlanSet(shortest=plans["shortest"], safest=plans["safest"],
                     recommended=plans["recommended"], candidates=cand_light,
                     n_candidates=len(E), n_feasible=n_feas,
                     reject_counts=reject_counts, weights=W, preset=preset,
                     aim_source=aim_src, k_rollout=k, rollout=ro,
                     plan_ms=(time.perf_counter() - t0) * 1000.0,
                     anatomy=anatomy, frame=lesion_state.frame,
                     track_id=lesion_state.track_id, legacy=legacy_out,
                     notes=notes + list(anatomy.events))
        return ps

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _norm_ranges(ev, idx):
        def rng(a):
            v = np.asarray(a, float)[idx]
            return float(v.min()), float(v.max())
        return {"len": rng(ev["L"]), "clear": rng(ev["minc"]),
                "angle": rng(ev["ang_cw"]), "lig": rng(ev["lig"]),
                "motion": rng(ev["motion"]), "target": rng(ev["target_err"])}

    @staticmethod
    def _nz(v, r):
        lo, hi = r
        if hi - lo < 1e-9:
            return np.zeros_like(np.asarray(v, float))
        return np.clip((np.asarray(v, float) - lo) / (hi - lo), 0.0, 1.5)

    def _terms(self, ev, norm, i=None):
        sl = slice(None) if i is None else i
        return {"len": self._nz(ev["L"][sl], norm["len"]),
                "clear": 1.0 - self._nz(ev["minc"][sl], norm["clear"]),
                "angle": self._nz(ev["ang_cw"][sl], norm["angle"]),
                "lig": self._nz(ev["lig"][sl], norm["lig"]),
                "motion": self._nz(ev["motion"][sl], norm["motion"]),
                "target": self._nz(ev["target_err"][sl], norm["target"])}

    def _costs(self, ev, norm, W):
        t = self._terms(ev, norm)
        return (W["w_len"] * t["len"] + W["w_clear"] * t["clear"] +
                W["w_angle"] * t["angle"] + W["w_lig"] * t["lig"] +
                W["w_motion"] * t["motion"] + W["w_target"] * t["target"])

    def _make_plan(self, kind, ev, i, side, off, t_now, t_pred, norm, W,
                   costs=None, an=None):
        terms = {k: float(v) for k, v in self._terms(ev, norm, i).items()} \
            if norm else {}
        cost = None
        if terms:
            cost = float(W["w_len"] * terms["len"] + W["w_clear"] *
                         terms["clear"] + W["w_angle"] * terms["angle"] +
                         W["w_lig"] * terms["lig"] + W["w_motion"] *
                         terms["motion"] + W["w_target"] * terms["target"])
        an = an if an is not None else self._an
        ps, pc, _ = clearance_profile(an, ev["E"][i], ev["d"][i],
                                      ev["s_max"][i])
        return NeedlePlan(
            kind=kind, side=int(side), offset_mm=float(off),
            entry_mm=tuple(map(float, ev["E"][i])),
            direction=tuple(map(float, ev["d"][i])),
            aim_mm=tuple(map(float, ev["C"][i])),
            prefire_mm=tuple(map(float, ev["P"][i])),
            fired_mm=tuple(map(float, ev["F"][i])),
            lesion_now_mm=tuple(map(float, t_now)),
            lesion_pred_mm=tuple(map(float, t_pred)),
            insertion_mm=float(ev["L"][i]),
            angle_chestwall_deg=float(ev["ang_cw"][i]),
            skin_angle_deg=float(ev["skin_ang"][i]),
            min_clearance_mm=float(ev["minc"][i]),
            min_clearance_struct=str(ev["which"][i]),
            clearance_by_family={k: float(v[i]) for k, v in
                                 ev["fam_min"].items()},
            profile_s=np.round(ps, 3).tolist(),
            profile_c=np.round(pc, 3).tolist(),
            ligaments_crossed=int(ev["lig"][i]),
            motion_sensitivity_mm=float(ev["motion"][i]),
            predicted_miss_mm=float(ev["pred_miss"][i]),
            target_error_mm=float(ev["target_err"][i]),
            aim_offset_mm=float(ev["aim_offset"][i]),
            core_radius_mm=float(ev["r_core"][i]),
            feasible=len(ev["reasons"][i]) == 0,
            reasons=list(ev["reasons"][i]), cost_terms=terms, total_cost=cost)

    def _refine(self, kind, p, an, aim, t_now, t_pred, motion_vec, sigma,
                norm, W):
        """Local continuous optimisation of (entry offset, angle change)."""
        from scipy.optimize import minimize
        W2 = self.calib.half_width_mm
        s = p.side

        def line(v):
            o, dth = float(v[0]), float(v[1])
            x = s * (W2 + o)
            E = np.array([x, float(an.skin_depth(x))])
            # direction: from E toward the aim, rotated by dth (degrees)
            d0 = aim - E
            a = math.atan2(d0[1], d0[0]) + math.radians(dth)
            return E, np.array([math.cos(a), math.sin(a)])

        cache = {}

        def ev_of(v):
            key = (round(float(v[0]), 9), round(float(v[1]), 9))
            if key not in cache:
                E, d = line(v)
                cache[key] = evaluate(an, E[None], d[None], aim[None], t_now,
                                      t_pred, motion_vec, sigma)
            return cache[key]

        def objective(v):
            e = ev_of(v)
            if kind == "shortest":
                return float(e["L"][0]) / 10.0
            if kind == "safest":
                return -float(e["minc"][0])
            t = self._terms(e, norm, 0)
            return float(W["w_len"] * t["len"] + W["w_clear"] * t["clear"] +
                         W["w_angle"] * t["angle"] + W["w_lig"] * t["lig"] +
                         W["w_motion"] * t["motion"] +
                         W["w_target"] * t["target"])

        def cons(v):
            e = ev_of(v)
            E, d = line(v)
            perp = abs((aim - E) @ np.array([-d[1], d[0]]))
            return np.array([
                float(e["slack"][0]) - cfg.PLAN_BUFFER_MM,
                cfg.MAX_ANGLE_TO_CHESTWALL_DEG - float(e["ang_cw"][0]),
                float(e["skin_ang"][0]) - cfg.MIN_SKIN_ANGLE_DEG,
                cfg.NEEDLE_LEN_MM - cfg.GRIP_MARGIN_MM - cfg.THROW_MM -
                float(e["L"][0]),
                float(e["L"][0]) - cfg.MIN_INSERTION_MM,
                cfg.TARGET_TOL_MM - perp,
                abs(E[0] - cfg.NIPPLE_X_MM) - cfg.NIPPLE_R_MM,
                float(e["gun_probe"][0]) - cfg.PROBE_CLEARANCE_MM,
                float(e["gun_skin"][0]) - 2.0,
            ])

        x0 = np.array([p.offset_mm, 0.0])
        f0 = objective(x0)
        bounds = [(cfg.ENTRY_MIN_OFFSET_MM, cfg.ENTRY_MAX_OFFSET_MM),
                  (-4.0, 4.0)]
        info = {"method": "SLSQP", "iterations": 0, "start": f0, "fun": f0,
                "success": False, "improved": False}
        try:
            res = minimize(objective, x0, method="SLSQP", bounds=bounds,
                           constraints=[{"type": "ineq", "fun": cons}],
                           options={"maxiter": 12, "ftol": 1e-6,
                                    "eps": 1e-3})
            info.update(iterations=int(res.nit), fun=float(res.fun),
                        success=bool(res.success))
            v = np.clip(res.x, [b[0] for b in bounds], [b[1] for b in bounds])
            if np.all(cons(v) >= -1e-9) and objective(v) < f0 - 1e-6:
                e = ev_of(v)
                if len(e["reasons"][0]) == 0:
                    q = self._make_plan(kind, e, 0, s, float(v[0]),
                                        t_now, t_pred, norm, W)
                    info["improved"] = True
                    info["delta_angle_deg"] = float(v[1])
                    q.refined = info
                    return q
        except Exception as exc:                           # noqa: BLE001
            info["error"] = str(exc)
        p.refined = info
        return p


def legacy_plan(lesion_state):
    """The ORIGINAL UltraBot 2D planner on the same frame, for comparison."""
    from ultrabot.biopsy import optimize_biopsy_paths
    try:
        opt = optimize_biopsy_paths(
            lesion=lesion_state.center_px, predicted=lesion_state.pred_px,
            frame_size=(cfg.PANE_W_PX, cfg.PANE_H_PX))
    except Exception as exc:                               # noqa: BLE001
        return {"error": str(exc)}

    def strip(row):
        if row is None:
            return None
        return {k: v for k, v in row.items() if k != "plan"}
    return {"recommended": strip(opt["recommended"]),
            "shortest": strip(opt["shortest"]),
            "safest": strip(opt["safest"]),
            "n_candidates": opt["n_candidates"],
            "n_feasible": opt["n_feasible"],
            "n_rejected": opt["n_rejected"]}
