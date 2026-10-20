"""Ultrasound calibration: pixel <-> probe frame <-> world."""
import numpy as np

from ultrabot_sim import config as cfg
from ultrabot_sim.us_geometry import (ProbePose, USCalib, out_of_plane_mm,
                                      pixel_to_world, world_to_pixel)


def _random_pose(rng):
    from scipy.spatial.transform import Rotation
    R = Rotation.from_rotvec(rng.normal(size=3)).as_matrix()
    return ProbePose(R, rng.uniform(-1, 1, 3))


def test_scale_from_config(calib):
    assert np.isclose(calib.mm_per_px_axial, cfg.US_DEPTH_MM / cfg.PANE_H_PX)
    assert np.isclose(calib.mm_per_px_lateral, cfg.US_WIDTH_MM / cfg.PANE_W_PX)
    x, z = calib.px_to_mm(cfg.PANE_W_PX / 2, 0)
    assert abs(x) < 1e-12 and abs(z) < 1e-12        # top centre = probe face
    x, z = calib.px_to_mm(cfg.PANE_W_PX, cfg.PANE_H_PX)
    assert np.isclose(x, cfg.US_WIDTH_MM / 2) and np.isclose(z, cfg.US_DEPTH_MM)


def test_pixel_world_round_trip(calib):
    rng = np.random.default_rng(0)
    for _ in range(20):
        pose = _random_pose(rng)
        u = rng.uniform(0, calib.pane_w, 100)
        v = rng.uniform(0, calib.pane_h, 100)
        w = pixel_to_world(u, v, calib, pose)
        u2, v2, y = world_to_pixel(w, calib, pose)
        w2 = pixel_to_world(u2, v2, calib, pose)
        assert np.max(np.linalg.norm(w2 - w, axis=1)) < 1e-9      # metres
        assert np.max(np.abs(u2 - u)) < 1e-6
        assert np.max(np.abs(y)) < 1e-9                  # on the image plane


def test_probe_world_round_trip():
    rng = np.random.default_rng(1)
    pose = _random_pose(rng)
    P = rng.uniform(-80, 80, (500, 3))
    assert np.max(np.abs(pose.world_to_probe(pose.probe_to_world(P)) - P)) \
        < 1e-6                                           # 1e-9 m in mm
    W = rng.uniform(-1, 1, (500, 3))
    assert np.max(np.abs(pose.probe_to_world(pose.world_to_probe(W)) - W)) \
        < 1e-9


def test_out_of_plane_and_axes():
    pose = ProbePose.nominal()
    R = pose.R
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0)
    assert np.allclose(np.cross(R[:, 2], R[:, 0]), R[:, 1])   # y = z x x
    w = pose.probe_to_world([3.0, 0.4, 10.0])
    assert np.isclose(out_of_plane_mm(w, pose), 0.4)


def test_depth_override_changes_scale():
    c = USCalib(30.0, 40.0)
    assert np.isclose(c.px_to_mm(0, 720)[1], 30.0)
