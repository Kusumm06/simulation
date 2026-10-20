"""
Rigid 14 G core-biopsy needle on a spring-loaded biopsy gun.

Tool chain on the needle arm. The gun sits on a bracket on the flange face
with its axis tilted NEEDLE_MOUNT_DEG from the flange z axis (toward the
flange x axis), like real robotic biopsy end-effectors; this keeps the
KUKA wrist far from its joint limits for near-horizontal needles:

    gun rear (flange face) --GUN_LEN--> needle hub --NEEDLE_LEN--> tip

Firing: the inner stylet (with the 19 mm specimen notch) shoots forward by
THROW_MM, then the outer cutting cannula follows over it, in about
FIRE_TIME_S. The gun itself does not move. Rigid model: no deflection,
no tissue deformation.
"""
from __future__ import annotations

import numpy as np

from ultrabot_sim import config as cfg

TOOL_FLANGE_OFFSET = 0.048          # m, iiwa link-7 frame -> flange face


def tool_length():
    """Flange face -> cocked needle tip (m)."""
    return cfg.GUN_LEN + cfg.NEEDLE_LEN_MM / 1000.0


def mount_axis_local():
    """Needle axis in the iiwa link-7 frame."""
    a = np.radians(cfg.NEEDLE_MOUNT_DEG)
    return np.array([np.sin(a), 0.0, np.cos(a)])


def tool_local():
    """(gun rear point, needle axis, cocked tip point) in the link-7 frame."""
    face = np.array([0.0, 0.0, TOOL_FLANGE_OFFSET])
    a = mount_axis_local()
    return face, a, face + a * tool_length()


def gun_rotation_local():
    """Rotation of the gun body (its z = needle axis) in the link-7 frame."""
    a = np.radians(cfg.NEEDLE_MOUNT_DEG)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


class NeedleModel:
    """Throw state of the gun. ext_* are mm beyond the cocked position."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.stylet_ext = 0.0
        self.cannula_ext = 0.0
        self.firing = False
        self.fired = False
        self._t = 0.0

    def fire(self):
        self.firing = True
        self.fired = False
        self._t = 0.0

    def update(self, dt, slowmo=1.0):
        """Advance the firing animation. Returns True when complete."""
        if not self.firing:
            return self.fired
        T = cfg.FIRE_TIME_S * max(slowmo, 1.0)
        self._t += dt
        u = min(self._t / T, 1.0)
        # stylet in the first half, cannula in the second half
        self.stylet_ext = cfg.THROW_MM * min(u / 0.5, 1.0)
        self.cannula_ext = cfg.THROW_MM * max(0.0, (u - 0.5) / 0.5)
        if u >= 1.0:
            self.firing = False
            self.fired = True
        return self.fired

    def tips(self, gun_rear, d):
        """World tips (stylet, cannula) for the gun rear point."""
        base = np.asarray(gun_rear, float) + np.asarray(d) * tool_length()
        return (base + np.asarray(d) * self.stylet_ext / 1000.0,
                base + np.asarray(d) * self.cannula_ext / 1000.0)

    def notch(self, gun_rear, d):
        """Specimen notch segment (world) on the stylet: the NOTCH_MM just
        behind the stylet's trocar point (about 2 mm)."""
        s, _ = self.tips(gun_rear, d)
        d = np.asarray(d, float)
        a = s - d * 0.002
        return a - d * cfg.NOTCH_MM / 1000.0, a
