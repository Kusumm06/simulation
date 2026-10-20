"""
Ultrasound calibration: pixel <-> probe frame (mm) <-> world (m).

A B-mode image is a slice through the body: its horizontal axis is the
lateral position along the transducer and its VERTICAL axis IS DEPTH below
the transducer face. So every pixel of the 990 x 720 pane is a real point in
the probe's image plane once the scanner's depth and width are known.

Probe frame (right-handed):
    origin  centre of the transducer face (on the skin)
    +x      lateral, image right
    +z      depth into the body, image down
    +y      elevation (out of the image plane) = z x x

Pixel (u, v)  ->  probe (x, 0, z):
    x = (u - W/2) * mm_per_px_lateral
    z =  v        * mm_per_px_axial

The world pose of the probe frame comes from the probe arm's flange
(PyBullet forward kinematics) plus the fixed flange -> face offset.
The mm scale is the CONFIGURED scanner preset (config.US_DEPTH_MM,
US_WIDTH_MM); it is not measured from the video.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ultrabot_sim import config as cfg


@dataclass
class USCalib:
    depth_mm: float
    width_mm: float
    pane_w: int = 990
    pane_h: int = 720

    @classmethod
    def from_config(cls):
        return cls(float(cfg.US_DEPTH_MM), float(cfg.US_WIDTH_MM),
                   int(cfg.PANE_W_PX), int(cfg.PANE_H_PX))

    @property
    def mm_per_px_lateral(self):
        return self.width_mm / self.pane_w

    @property
    def mm_per_px_axial(self):
        return self.depth_mm / self.pane_h

    @property
    def half_width_mm(self):
        return self.width_mm / 2.0

    def px_to_mm(self, u, v):
        """Pixel -> probe-plane (x_mm, z_mm). Works on scalars or arrays."""
        u = np.asarray(u, float)
        v = np.asarray(v, float)
        return ((u - self.pane_w / 2.0) * self.mm_per_px_lateral,
                v * self.mm_per_px_axial)

    def mm_to_px(self, x, z):
        x = np.asarray(x, float)
        z = np.asarray(z, float)
        return (x / self.mm_per_px_lateral + self.pane_w / 2.0,
                z / self.mm_per_px_axial)

    def len_px_to_mm(self, du, dv):
        """A pixel displacement -> mm displacement (lateral, axial)."""
        return (np.asarray(du, float) * self.mm_per_px_lateral,
                np.asarray(dv, float) * self.mm_per_px_axial)

    def in_image(self, x, z):
        return (np.abs(x) <= self.half_width_mm) & (z >= 0) & \
            (z <= self.depth_mm)


class ProbePose:
    """World pose of the probe frame. R columns = probe x, y, z axes in the
    world; t = transducer-face centre (m)."""

    def __init__(self, R, t):
        R = np.asarray(R, float)
        self.R = R
        self.t = np.asarray(t, float)

    @classmethod
    def from_axes(cls, x_axis, z_axis, origin):
        x = np.asarray(x_axis, float)
        z = np.asarray(z_axis, float)
        x = x / np.linalg.norm(x)
        z = z - x * np.dot(z, x)
        z = z / np.linalg.norm(z)
        y = np.cross(z, x)
        return cls(np.column_stack([x, y, z]), origin)

    @classmethod
    def from_flange(cls, flange_pos, flange_R, probe_len=None):
        """Probe frame from the arm flange: probe axes = flange axes, face
        = flange + z * PROBE_LEN."""
        L = cfg.PROBE_LEN if probe_len is None else probe_len
        Rf = np.asarray(flange_R, float)
        return cls(Rf, np.asarray(flange_pos, float) + Rf[:, 2] * L)

    @classmethod
    def nominal(cls):
        return cls.from_axes(cfg.PROBE_X_AXIS, cfg.PROBE_Z_AXIS,
                             cfg.PROBE_SITE)

    def flange_target(self, probe_len=None):
        L = cfg.PROBE_LEN if probe_len is None else probe_len
        return self.t - self.R[:, 2] * L

    # mm in the probe frame <-> metres in the world
    def probe_to_world(self, p_mm):
        p = np.asarray(p_mm, float) / 1000.0
        return p @ self.R.T + self.t

    def world_to_probe(self, w):
        w = np.asarray(w, float)
        return ((w - self.t) @ self.R) * 1000.0

    def dir_probe_to_world(self, d):
        return np.asarray(d, float) @ self.R.T

    def dir_world_to_probe(self, d):
        return np.asarray(d, float) @ self.R

    def copy(self):
        return ProbePose(self.R.copy(), self.t.copy())


def pixel_to_world(u, v, calib: USCalib, pose: ProbePose):
    x, z = calib.px_to_mm(u, v)
    x = np.asarray(x, float)
    p = np.stack([x, np.zeros_like(x), np.asarray(z, float)], axis=-1)
    return pose.probe_to_world(p)


def world_to_pixel(w, calib: USCalib, pose: ProbePose):
    """World point(s) -> (u, v, out_of_plane_mm)."""
    p = pose.world_to_probe(w)
    u, v = calib.mm_to_px(p[..., 0], p[..., 2])
    return u, v, p[..., 1]


def probe_to_world(p_mm, pose: ProbePose):
    return pose.probe_to_world(p_mm)


def world_to_probe(w, pose: ProbePose):
    return pose.world_to_probe(w)


def out_of_plane_mm(w, pose: ProbePose):
    """Elevational distance of world point(s) from the image plane (mm)."""
    return np.abs(pose.world_to_probe(w)[..., 1])


def visible_in_slice(w, pose: ProbePose, slice_mm=None):
    s = cfg.US_SLICE_MM if slice_mm is None else slice_mm
    return out_of_plane_mm(w, pose) <= s / 2.0
