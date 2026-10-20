"""Anatomy-aware planning: shortest / safest / recommended."""
import time

import numpy as np

from ultrabot_sim import config as cfg
from ultrabot_sim.anatomy import Anatomy, Vessel
from ultrabot_sim.planner import Planner, segment_clearance

from .conftest import make_state

LESIONS = [(0.0, 12.0), (-6.0, 9.0), (8.0, 15.0), (3.0, 20.0), (-10.0, 14.0)]


def _plan(calib, x, z, **kw):
    st = make_state(calib, x, z)
    an = Anatomy(calib, st.geom)
    return Planner(calib).plan(st, an, None, legacy=False, **kw), an


def test_entries_on_skin_beside_probe(calib):
    an = Anatomy(calib, make_state(calib, 0, 12).geom)
    for s, off, x, z in Planner(calib).entries(an):
        assert abs(x) >= calib.half_width_mm + cfg.ENTRY_MIN_OFFSET_MM - 1e-9
        assert np.isclose(z, an.skin_depth(x))       # on the skin, never deeper
        assert z < an.fascia_depth(x)                # never from the chest wall
        assert z < 0.5 * calib.depth_mm              # never the image bottom


def test_feasible_paths_keep_chestwall_margin(calib):
    for x, z in LESIONS:
        ps, an = _plan(calib, x, z)
        for kind in ("shortest", "safest", "recommended"):
            p = ps.get(kind)
            assert p is not None and p.feasible
            fc = an.clearance(np.array([p.fired_mm]))["chest wall"][0]
            assert fc >= cfg.CHESTWALL_CLEARANCE_MM - 1e-6
            assert p.angle_chestwall_deg <= cfg.MAX_ANGLE_TO_CHESTWALL_DEG
            assert not an.in_nipple(p.entry_mm[0])
            for fam, v in p.clearance_by_family.items():
                assert v >= an.margins[fam] - 1e-6


def test_shortest_vs_safest(calib):
    for x, z in LESIONS:
        ps, _ = _plan(calib, x, z)
        assert ps.shortest.insertion_mm <= ps.safest.insertion_mm + 1e-6
        assert ps.safest.min_clearance_mm >= ps.shortest.min_clearance_mm - 1e-6


def test_vessel_on_line_is_refused(calib):
    ps, an = _plan(calib, 0.0, 12.0)
    p = ps.safest
    mid = np.array(p.entry_mm) + np.array(p.direction) * p.insertion_mm / 2
    st = make_state(calib, 0.0, 12.0)
    an2 = Anatomy(calib, st.geom)
    an2.vessels.append(Vessel(float(mid[0]), float(mid[1]), 1.0))
    c = segment_clearance(an2, np.array([p.entry_mm]), np.array([p.direction]),
                          np.array([p.insertion_mm + cfg.THROW_MM]))
    assert c["vessel"][0] < 0                        # the old line hits it
    ps2 = Planner(calib).plan(st, an2, None, legacy=False)
    for kind in ("shortest", "safest", "recommended"):
        q = ps2.get(kind)
        assert q is None or q.clearance_by_family["vessel"] >= \
            cfg.VESSEL_MARGIN_MM - 1e-6
    assert ps2.reject_counts.get("vessel clearance", 0) > \
        ps.reject_counts.get("vessel clearance", 0)


def test_planning_time(calib):
    _plan(calib, 0, 12)                              # warm-up
    ts = []
    for x, z in LESIONS:
        t = time.perf_counter()
        _plan(calib, x, z)
        ts.append((time.perf_counter() - t) * 1000)
    assert np.median(ts) < cfg.PLAN_BUDGET_MS, ts


def test_deterministic(calib):
    a, _ = _plan(calib, 2.0, 13.0)
    b, _ = _plan(calib, 2.0, 13.0)
    for k in ("shortest", "safest", "recommended"):
        assert a.get(k).to_dict() == b.get(k).to_dict()


def test_side_lock(calib):
    ps, _ = _plan(calib, 0.0, 12.0, side_lock=-1)
    for k in ("shortest", "safest", "recommended"):
        assert ps.get(k).side == -1
