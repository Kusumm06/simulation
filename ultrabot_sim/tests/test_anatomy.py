"""Parametric anatomy: layers, signed distances, constraints."""
import numpy as np

from ultrabot_sim import config as cfg
from ultrabot_sim.anatomy import Anatomy, LesionGeom, estimate_chestwall


def test_layer_order_and_skin(calib):
    an = Anatomy(calib, LesionGeom(0, 12, 8, 4))
    hw = calib.half_width_mm
    assert np.allclose(an.skin_depth(np.linspace(-hw, hw, 50)), 0.0)
    xs = np.linspace(hw + 1, hw + 80, 40)
    d = an.skin_depth(xs)
    assert np.all(np.diff(d) > 0)                       # falls away outside
    for x in (-60.0, 0.0, 40.0):
        seq = [an.skin_depth(x), an.fat_bottom(x), an.fascia_depth(x),
               an.muscle_bottom(x), an.pleura_depth(x)]
        assert all(a < b for a, b in zip(seq, seq[1:]))


def test_sdf_signs(calib):
    an = Anatomy(calib, LesionGeom(0, 12, 8, 4))
    above = an.clearance(np.array([[0.0, an.fascia_z0 - 4.0]]))
    below = an.clearance(np.array([[0.0, an.fascia_z0 + 2.0]]))
    assert np.isclose(above["chest wall"][0], 4.0)
    assert below["chest wall"][0] < 0
    v = an.vessels[0]
    inside = an.clearance(np.array([[v.x, v.z]]))["vessel"][0]
    assert np.isclose(inside, -v.r)
    on_surface = an.clearance(np.array([[v.x + v.r, v.z]]))["vessel"][0]
    assert abs(on_surface) < 1e-9
    P3 = np.array([[v.x, 7.0, v.z]])                    # tube along y
    assert np.isclose(an.clearance(P3)["vessel"][0], -v.r)


def test_fascia_pushed_below_deep_lesion(calib):
    deep = LesionGeom(0, 30, 8, 6)                      # bottom at 36 mm
    an = Anatomy(calib, deep)
    assert an.fascia_z0 >= 36 + cfg.PECTORAL_MIN_BELOW_LESION_MM - 1e-9
    assert any("pushed" in e for e in an.events)


def test_vessel_moved_off_lesion(calib):
    x, z, r = cfg.VESSELS[0]
    an = Anatomy(calib, LesionGeom(x, z, 6, 4))
    v = an.vessels[0]
    assert v.moved_mm != 0
    assert an.lesion.boundary_distance(v.x, v.z) >= r + cfg.VESSEL_MARGIN_MM \
        - 1e-6
    assert any("overlapped" in e for e in an.events)


def test_meshes_exist(calib):
    an = Anatomy(calib, LesionGeom(0, 12, 8, 4))
    names = {n for n, _, _ in an.meshes()}
    assert {"breast-skin", "fascia", "muscle", "vessel", "rib",
            "lesion"} <= names


def test_auto_chestwall_on_synthetic_image(calib):
    img = np.full((720, 990, 3), 40, np.uint8)
    row = 560                                           # 31.1 mm
    img[row:row + 18] = 220
    est = estimate_chestwall(img, calib)
    assert est.used, est.message
    assert abs(est.z0 - row * calib.mm_per_px_axial) < 1.5
    flat = np.full((720, 990, 3), 60, np.uint8)
    assert not estimate_chestwall(flat, calib).used
    an = Anatomy(calib, LesionGeom(0, 30, 8, 6), img, auto_chestwall=True)
    assert an.fascia_z0 >= 36 + cfg.PECTORAL_MIN_BELOW_LESION_MM - 1e-6
