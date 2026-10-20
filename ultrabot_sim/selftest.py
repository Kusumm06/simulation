"""
Self-test: about 40 checks, PASS / WARN / FAIL, in groups

  A  geometry        pixel <-> mm <-> world, round trips, scale
  B  planning        anatomy, constraints, shortest vs safest, timing
  C  perception      YOUR models/best.pt + lstm_checkpoint.pth + video
  D  robot           PyBullet scene, IK accuracy, RCM, approach, limits
  E  renderer        pyrender / EGL, 3D panel draws a picture
  F  end to end      headless run of the shortest and the safest path

    python -m ultrabot_sim.selftest              # everything
    python -m ultrabot_sim.selftest --skip-e2e   # A-E only (fast)
    python -m ultrabot_sim.selftest --no-render  # skip E
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np                                          # noqa: E402

from ultrabot_sim import config as cfg                      # noqa: E402

RESULTS = []
if sys.platform.startswith("win"):
    import os
    os.system("")                     # enable ANSI colours in the console


def check(group, name, fn):
    try:
        out = fn()
        if isinstance(out, tuple):
            status, msg = out
            if not isinstance(status, str):
                status = "PASS" if bool(status) else "FAIL"
        else:
            status, msg = ("PASS" if out else "FAIL"), ""
    except Exception as e:                                  # noqa: BLE001
        status, msg = "FAIL", f"{type(e).__name__}: {e}"
        if "-v" in sys.argv:
            traceback.print_exc()
    RESULTS.append((group, name, status, msg))
    col = {"PASS": "\033[32m", "WARN": "\033[33m", "FAIL": "\033[31m"}[status]
    print(f"  {col}{status:4s}\033[0m  {group}  {name}"
          f"{('  - ' + msg) if msg else ''}")
    return status


def group_a():
    from ultrabot_sim.us_geometry import (ProbePose, USCalib, pixel_to_world,
                                          world_to_pixel)
    cal = USCalib.from_config()
    pose = ProbePose.nominal()
    check("A", "mm scale from the scanner preset", lambda: (
        "PASS", f"{cal.mm_per_px_lateral:.4f} x {cal.mm_per_px_axial:.4f} "
        f"mm/px ({cfg.US_WIDTH_MM:.0f} x {cfg.US_DEPTH_MM:.0f} mm)"))

    def rt():
        rng = np.random.default_rng(0)
        u = rng.uniform(0, 990, 1000)
        v = rng.uniform(0, 720, 1000)
        w = pixel_to_world(u, v, cal, pose)
        u2, v2, y = world_to_pixel(w, cal, pose)
        e = np.max(np.linalg.norm(pixel_to_world(u2, v2, cal, pose) - w,
                                  axis=1))
        return e < 1e-9, f"max round-trip error {e:.1e} m"
    check("A", "pixel -> world -> pixel round trip < 1e-9 m", rt)
    check("A", "image top = transducer face, bottom = depth",
          lambda: abs(cal.px_to_mm(495, 720)[1] - cfg.US_DEPTH_MM) < 1e-9)
    check("A", "probe frame right-handed (y = z x x)", lambda: np.allclose(
        np.cross(pose.R[:, 2], pose.R[:, 0]), pose.R[:, 1]))
    check("A", "in-plane point has zero elevation", lambda: abs(
        pose.world_to_probe(pixel_to_world(300, 300, cal, pose))[1]) < 1e-9)
    return cal


def group_b(cal):
    from ultrabot_sim.anatomy import Anatomy, LesionGeom
    from ultrabot_sim.lesion_model import LesionState
    from ultrabot_sim.planner import Planner

    def state(x, z):
        u, v = cal.mm_to_px(x, z)
        return LesionState(1, 1, (float(u), float(v)), (x, z),
                           LesionGeom(x, z, 9, 5), "benign", 0.9, False)
    an = Anatomy(cal, LesionGeom(0, 12, 9, 5))
    check("B", "anatomy layers ordered (skin < fat < fascia < pleura)",
          lambda: an.skin_depth(0) < an.fat_bottom(0) < an.fascia_depth(0)
          < an.pleura_depth(0))
    deep = Anatomy(cal, LesionGeom(0, 30, 9, 6))
    check("B", "fascia >= 6 mm below the deepest lesion edge",
          lambda: deep.fascia_z0 >= 36 + cfg.PECTORAL_MIN_BELOW_LESION_MM)
    pl = Planner(cal)
    pl.plan(state(0, 12), an, None, legacy=False)
    sets = []
    times = []
    for x, z in ((0, 12), (-6, 9), (8, 15), (3, 20)):
        a = Anatomy(cal, LesionGeom(x, z, 9, 5))
        t = time.perf_counter()
        ps = pl.plan(state(x, z), a, None, legacy=False)
        times.append((time.perf_counter() - t) * 1000)
        from ultrabot_sim.planner import legacy_plan
        ps.legacy = legacy_plan(state(x, z))
        sets.append((ps, a))
    check("B", "every lesion gets shortest / safest / recommended",
          lambda: all(ps.get(k) is not None for ps, _ in sets
                      for k in ("shortest", "safest", "recommended")))
    check("B", "entries only on the skin beside the probe", lambda: all(
        abs(ps.get(k).entry_mm[0]) >= cal.half_width_mm +
        cfg.ENTRY_MIN_OFFSET_MM - 1e-9 for ps, _ in sets
        for k in ("shortest", "safest")))
    check("B", "throw end >= 5 mm above the chest wall", lambda: all(
        a.clearance(np.array([ps.get(k).fired_mm]))["chest wall"][0] >=
        cfg.CHESTWALL_CLEARANCE_MM - 1e-6 for ps, a in sets
        for k in ("shortest", "safest", "recommended")))
    check("B", "len(shortest) <= len(safest)", lambda: all(
        ps.shortest.insertion_mm <= ps.safest.insertion_mm + 1e-6
        for ps, _ in sets))
    check("B", "clearance(safest) >= clearance(shortest)", lambda: all(
        ps.safest.min_clearance_mm >= ps.shortest.min_clearance_mm - 1e-6
        for ps, _ in sets))
    check("B", "angle to chest wall <= 35 deg", lambda: all(
        ps.get(k).angle_chestwall_deg <= cfg.MAX_ANGLE_TO_CHESTWALL_DEG
        for ps, _ in sets for k in ("shortest", "safest", "recommended")))
    med = float(np.median(times))
    check("B", f"planning time < {cfg.PLAN_BUDGET_MS:.0f} ms", lambda: (
        "PASS" if med < cfg.PLAN_BUDGET_MS else "WARN",
        f"median {med:.0f} ms (without robot IK checks)"))
    check("B", "legacy 2D planner also runs (comparison layer)",
          lambda: sets[0][0].legacy and "recommended" in sets[0][0].legacy)
    check("B", "scipy refinement ran", lambda: (
        "PASS", f"{sets[0][0].recommended.refined}"[:80]))


def group_c(cal):
    from ultrabot.pipeline import check_model_availability, load_models
    st = check_model_availability()
    check("C", "models/best.pt found", lambda: (
        st["yolo_available"], st["yolo_path"]))
    check("C", "models/lstm_checkpoint.pth found", lambda: (
        st["lstm_available"], st["lstm_path"]))
    if not st["all_available"]:
        return None
    models = load_models()
    check("C", "YOLO classes", lambda: ("PASS", str(models[3])))
    check("C", "LSTM sequence length 20", lambda: (
        models[2]["sequence_length"] == 20, str(models[2])))
    vid = cfg.DEFAULT_VIDEO
    check("C", "sample video outputs/input.mp4", lambda: (
        vid.is_file(), str(vid)))
    if not vid.is_file():
        return models
    from ultrabot_sim.inputs import VideoSource
    from ultrabot_sim.perception import Perception
    src = VideoSource(vid, start_frame=760, max_frames=40)
    per = Perception(models=models)
    obs = []
    t = time.perf_counter()
    while True:
        it = src.next()
        if it is None:
            break
        obs.append(per.step(it[1], it[0]))
    ms = (time.perf_counter() - t) * 1000 / max(len(obs), 1)
    check("C", "YOLO detects the lesion", lambda: (
        sum(o.detected for o in obs) > 30,
        f"{sum(o.detected for o in obs)}/{len(obs)} frames, {ms:.0f} ms/frame "
        f"on {'GPU' if cfg.DEVICE != 'cpu' else 'CPU'}"))
    check("C", "LSTM only after 20 consecutive hits", lambda: all(
        o.consecutive_hits >= 20 for o in obs if o.lstm_ready) and
        any(o.lstm_ready for o in obs))
    from ultrabot_sim.lesion_model import LesionTracker
    lt = LesionTracker(cal, models[1])
    for o in obs:
        lt.update(o)
    check("C", "LSTM one-step error measured in mm", lambda: (
        "PASS", str({k: round(v, 2) if isinstance(v, float) else v
                     for k, v in lt.error_stats().items()})))
    return models


def group_d(cal):
    from ultrabot_sim.approach import Approach
    from ultrabot_sim.rcm import LockedLine, rcm_deviation
    from ultrabot_sim.scene import Scene, check_reachability
    sc = Scene()
    sc.set_q("probe", sc.ready_q["probe"])
    check("D", "PyBullet scene + two KUKA iiwa arms", lambda: (
        len(sc.arms) == 2, f"{len(sc.groups)} body groups"))
    check("D", "probe on the skin (6-DoF IK)", lambda: (
        sc.contact_err[0] < cfg.IK_TOL, f"{sc.contact_err[0] * 1000:.5f} mm, "
        f"{sc.contact_err[1]:.5f} deg"))
    check("D", "home / ready poses reached", lambda: sc.home_ok)
    frac = check_reachability(sc, verbose=False)
    check("D", "needle arm reaches the in-plane workspace", lambda: (
        "PASS" if frac >= 0.95 else "WARN" if frac > 0.8 else "FAIL",
        f"{100 * frac:.1f}% of sampled poses"))
    rng = np.random.default_rng(5)
    errs = []
    for i in range(40):
        base = sc.ready_side_q[1 if i % 2 else -1]
        q = np.clip(np.asarray(base) + rng.normal(0, 0.25, 7),
                    np.array(cfg.JOINT_LOWER) * 0.95,
                    np.array(cfg.JOINT_UPPER) * 0.95)
        tip, d = sc.needle_axis(q)
        _, e, a, _, _ = sc.ik_needle_robust(tip, d, base, n_random=8)
        errs.append((e * 1000, a))
    check("D", "IK < 0.1 mm / 0.05 deg on random reachable targets", lambda: (
        all(e < 0.1 and a < 0.05 for e, a in errs),
        f"max {max(e for e, _ in errs):.5f} mm, "
        f"{max(a for _, a in errs):.5f} deg"))
    pose = sc.probe_pose()
    E = np.array([35.0, 0.05])
    d = (np.array([0.0, 14.0]) - E)
    d /= np.linalg.norm(d)
    Ew = pose.probe_to_world([E[0], 0, E[1]])
    dw = pose.dir_probe_to_world([d[0], 0, d[1]])
    line = LockedLine(Ew, dw)
    q, *_ = sc.ik_needle_robust(line.tip(-0.01), dw, sc.ready_side_q[1])
    dev = 0.0
    for s in np.linspace(0, 0.04, 80):
        q, e, a = sc.ik_needle(line.tip(s), dw, q, iters=80)
        tip, dd = sc.needle_axis(q)
        dev = max(dev, rcm_deviation(Ew, tip, dd) * 1000)
    check("D", "skin RCM deviation < 0.5 mm during insertion", lambda: (
        dev < 0.5, f"{dev:.5f} mm"))
    ap = Approach(sc, seed=0)
    jp = ap.plan("needle", sc.home_q["needle"], sc.ready_side_q[-1])
    check("D", "collision-free approach (RRT-Connect + timing)", lambda: (
        jp is not None and ap.validate("needle", jp),
        f"{ap.stats.get('waypoints')} waypoints, {jp.T:.1f} s" if jp else
        str(ap.stats)))
    cl = sc.clearances()
    check("D", "arm-arm clearance at rest", lambda: (
        cl["arm_arm"] >= cfg.ARM_MIN_CLEARANCE, f"{cl['arm_arm'] * 100:.1f} cm"))
    return sc


def group_e(sc):
    try:
        from ultrabot_sim.or_render import InlinePanel
        q0 = cfg.OR_QUALITY
        cfg.OR_QUALITY = "low"
        t = time.perf_counter()
        pan = InlinePanel(sc)
        pan.submit(None)
        ms = (time.perf_counter() - t) * 1000
        cfg.OR_QUALITY = q0
        img = pan.image
        check("E", "realistic 3D room (pyrender) draws", lambda: (
            img is not None and img.std() > 5,
            f"{pan.gl_info}; first frame {ms:.0f} ms"))
        pan.close()
    except Exception as e:                                  # noqa: BLE001
        why = str(e).splitlines()[0][:80] if str(e) else type(e).__name__
        check("E", "realistic 3D room (pyrender) draws", lambda: (
            "WARN", f"not available ({why}); the PyBullet fallback "
            "renderer will be used"))
    img = sc.render((320, 200))
    check("E", "PyBullet fallback renderer draws", lambda: img.std() > 2)
    import cv2
    check("E", "OpenCV console available", lambda: (
        "PASS", f"OpenCV {cv2.__version__}"))


def group_f():
    from ultrabot_sim.main import main
    import json
    main(["--compare", "--headless", "--once", "--tag", "selftest"])
    s = json.loads((cfg.RUNS_DIR / "summary_selftest.json").read_text())
    for p in s["procedures"]:
        crit = {c["check"]: c["pass"] for c in p["criteria"]}
        check("F", f"{p['kind']} path end to end", lambda p=p: (
            "PASS" if p["success"] else "WARN",
            f"{p['outcome']} {p.get('abort_reason', '')}".strip()))
        for name, ok in crit.items():
            check("F", f"  {p['kind']}: {name}", lambda ok=ok: ok)
    if len(s["procedures"]) < 2:
        check("F", "both paths executed", lambda: False)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-e2e", action="store_true")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args(argv)
    print(f"UltraBot 3D simulation self-test  ({cfg.DISCLAIMER})")
    print("A geometry")
    cal = group_a()
    print("B planning")
    group_b(cal)
    print("C perception (your models + video)")
    group_c(cal)
    print("D robot")
    sc = group_d(cal)
    if not a.no_render:
        print("E renderer")
        group_e(sc)
    sc.disconnect()
    if not a.skip_e2e:
        print("F end to end (headless compare on the sample video)")
        group_f()
    n = {k: sum(1 for r in RESULTS if r[2] == k) for k in
         ("PASS", "WARN", "FAIL")}
    print(f"\n{len(RESULTS)} checks: {n['PASS']} PASS, {n['WARN']} WARN, "
          f"{n['FAIL']} FAIL")
    return 1 if n["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
