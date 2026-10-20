"""
Skin-entry remote centre of motion (RCM), adapted from the reference
project's trocar RCM.

The reference parameterises a laparoscopic instrument BY its trocar so the
pivot is exact by construction. A biopsy needle is simpler: once the tip
has passed the skin, a rigid needle must not move sideways at the skin at
all (it cannot steer), so the only remaining degree of freedom is the
insertion depth s along the locked line:

    tip(s) = E + d * s        E = skin entry point, d = locked direction

The arm is placed to match (flange = tip - d * tool length); the RCM
deviation is then measured from the ARM'S ACTUAL forward kinematics, not
from the command, so it shows the real sideways motion at the skin.
"""
from __future__ import annotations

import math

import numpy as np


def point_line_distance(p, a, d):
    """Distance from point p to the infinite line a + t d (d unit)."""
    p, a, d = (np.asarray(v, float) for v in (p, a, d))
    v = p - a
    return float(np.linalg.norm(v - d * np.dot(v, d)))


def rcm_deviation(entry, axis_point, axis_dir):
    """Sideways distance (m) between the skin entry point and the needle
    axis as the arm actually holds it."""
    return point_line_distance(entry, axis_point, axis_dir / np.linalg.norm(
        axis_dir))


def axis_angle_deg(a, b):
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    return math.degrees(math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0))))


class LockedLine:
    """A needle line locked at skin puncture."""

    def __init__(self, entry, d):
        self.entry = np.asarray(entry, float)
        d = np.asarray(d, float)
        self.d = d / np.linalg.norm(d)

    def tip(self, s):
        """Tip position at insertion s (m, negative = before the skin)."""
        return self.entry + self.d * s


class TrapezoidProfile:
    """Acceleration-limited 1-D move s0 -> s1 (units consistent, e.g. mm,
    mm/s, mm/s^2). position(t) for t >= 0."""

    def __init__(self, s0, s1, vmax, amax):
        self.s0, self.s1 = float(s0), float(s1)
        self.sign = 1.0 if s1 >= s0 else -1.0
        D = abs(s1 - s0)
        self.vmax, self.amax = float(vmax), float(amax)
        ta = self.vmax / self.amax
        da = 0.5 * self.amax * ta * ta
        if 2 * da > D:                    # triangular
            ta = math.sqrt(D / self.amax)
            self.vpk = self.amax * ta
            self.ta, self.tc = ta, 0.0
        else:
            self.vpk = self.vmax
            self.ta = ta
            self.tc = (D - 2 * da) / self.vmax
        self.T = 2 * self.ta + self.tc
        self.D = D

    def position(self, t):
        t = min(max(t, 0.0), self.T)
        a, ta, tc, v = self.amax, self.ta, self.tc, self.vpk
        if t < ta:
            s = 0.5 * a * t * t
        elif t < ta + tc:
            s = 0.5 * a * ta * ta + v * (t - ta)
        else:
            td = self.T - t
            s = self.D - 0.5 * a * td * td
        return self.s0 + self.sign * s

    def velocity(self, t):
        if t <= 0 or t >= self.T:
            return 0.0
        if t < self.ta:
            v = self.amax * t
        elif t < self.ta + self.tc:
            v = self.vpk
        else:
            v = self.amax * (self.T - t)
        return self.sign * v

    def done(self, t):
        return t >= self.T
