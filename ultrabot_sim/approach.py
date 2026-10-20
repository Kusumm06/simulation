"""
Collision-free robot motion outside the body.

RRT-Connect in joint space with PyBullet collision checks on the ghost
world (arm links, gun and needle against the patient, table, breast and
the other arm), then random shortcutting, then a minimum-jerk (quintic)
time law per segment that respects MAX_JOINT_VEL and MAX_TCP_SPEED.

The planner is deterministic for a given seed. Every returned path is
re-checked at RRT_CHECK_STEP resolution before it is accepted.
"""
from __future__ import annotations

import math
import time

import numpy as np

from ultrabot_sim import config as cfg

MINJERK_PEAK = 1.875            # peak velocity of s(t)=10t^3-15t^4+6t^5


def minjerk(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * u * (10 - 15 * u + 6 * u * u)


class JointPath:
    """Waypoints + per-segment durations; q(t) by quintic blending."""

    def __init__(self, waypoints, durations):
        self.wp = [np.asarray(w, float) for w in waypoints]
        self.dur = list(durations)
        self.T = float(sum(self.dur))

    def q(self, t):
        if not self.dur:
            return self.wp[0].copy()
        t = min(max(t, 0.0), self.T)
        acc = 0.0
        for i, d in enumerate(self.dur):
            if t <= acc + d or i == len(self.dur) - 1:
                u = (t - acc) / d if d > 0 else 1.0
                s = minjerk(u)
                return self.wp[i] + (self.wp[i + 1] - self.wp[i]) * s
            acc += d
        return self.wp[-1].copy()

    def done(self, t):
        return t >= self.T


class SplinePath:
    """C2 cubic spline through joint-space waypoints, u in [0, 1]."""

    def __init__(self, waypoints):
        from scipy.interpolate import CubicSpline
        W = np.array(waypoints, float)
        seg = np.linalg.norm(np.diff(W, axis=0), axis=1)
        lam = np.concatenate([[0.0], np.cumsum(seg)])
        self.length = float(lam[-1])
        lam = lam / max(lam[-1], 1e-9)
        self.cs = CubicSpline(lam, W, axis=0, bc_type="clamped")
        self.lo = np.array(cfg.JOINT_LOWER)
        self.hi = np.array(cfg.JOINT_UPPER)

    def at(self, u):
        return np.clip(self.cs(min(max(u, 0.0), 1.0)), self.lo, self.hi)


class SplineJointPath:
    """Same interface as JointPath for a spline + min-jerk time law. With a
    reparametrisation table (u as a function of normalised tool arc length)
    the tool point follows the min-jerk profile ALONG the curve, so no part
    of the path dictates a slow overall time."""

    def __init__(self, spline, T, sigma=None, u_of_sigma=None):
        self.sp = spline
        self.T = float(T)
        self.sig = sigma
        self.uos = u_of_sigma
        self.wp = [spline.at(u) for u in np.linspace(0, 1, 9)]

    def _u(self, t):
        m = minjerk(t / self.T) if self.T > 0 else 1.0
        if self.sig is None:
            return m
        return float(np.interp(m, self.sig, self.uos))

    def q(self, t):
        return self.sp.at(self._u(t))

    def done(self, t):
        return t >= self.T


class Approach:
    def __init__(self, scene, seed=0):
        self.scene = scene
        self.rng = np.random.default_rng(seed)
        self.lo = np.array(cfg.JOINT_LOWER)
        self.hi = np.array(cfg.JOINT_UPPER)
        self.stats = {}

    # ------------------------------------------------------------ checks
    def valid(self, name, q, final_zone=False):
        return self.scene.collision_free(name, q, allow_needle_breast=True)

    def edge_ok(self, name, a, b, step=None):
        step = step or cfg.RRT_CHECK_STEP
        n = max(1, int(math.ceil(np.max(np.abs(b - a)) / step)))
        for k in range(1, n + 1):
            if not self.valid(name, a + (b - a) * (k / n)):
                return False
        return True

    # ------------------------------------------------------------ RRT
    def plan(self, name, q_start, q_goal, max_iters=None, timeout=8.0):
        """RRT-Connect. Returns a JointPath or None."""
        t0 = time.perf_counter()
        qs, qg = np.asarray(q_start, float), np.asarray(q_goal, float)
        self.stats = {"iters": 0, "nodes": 0, "ms": 0.0, "direct": False,
                      "ok": False}
        self.stats["start_valid"] = bool(self.valid(name, qs))
        if not self.valid(name, qg):
            self.stats["reason"] = "goal configuration in collision"
            return None
        if self.edge_ok(name, qs, qg):
            self.stats.update(direct=True, ok=True)
            path = [qs, qg]
        else:
            path = self._rrt_connect(name, qs, qg,
                                     max_iters or cfg.RRT_MAX_ITERS, timeout)
            if path is None:
                self.stats["ms"] = (time.perf_counter() - t0) * 1000
                self.stats["reason"] = "RRT-Connect found no path"
                return None
            path = self._shortcut(name, path)
            self.stats["ok"] = True
        jp = self.time_parametrise(name, path)
        self.stats["ms"] = (time.perf_counter() - t0) * 1000
        self.stats["waypoints"] = len(path)
        self.stats["duration_s"] = jp.T
        return jp

    def _nearest(self, tree, q):
        Q = np.array([n[0] for n in tree])
        return int(np.argmin(np.linalg.norm(Q - q, axis=1)))

    def _extend(self, name, tree, q, step):
        i = self._nearest(tree, q)
        qn = tree[i][0]
        d = q - qn
        dist = np.linalg.norm(d)
        new = q.copy() if dist <= step else qn + d / dist * step
        if not self.edge_ok(name, qn, new):
            return None, "trapped"
        tree.append((new, i))
        return len(tree) - 1, ("reached" if dist <= step else "advanced")

    def _connect(self, name, tree, q, step):
        while True:
            k, st = self._extend(name, tree, q, step)
            if st != "advanced":
                return k, st

    def _rrt_connect(self, name, qs, qg, max_iters, timeout):
        t0 = time.perf_counter()
        A, B = [(qs, -1)], [(qg, -1)]
        step = cfg.RRT_STEP
        a_is_start = True
        for it in range(max_iters):
            self.stats["iters"] = it + 1
            if time.perf_counter() - t0 > timeout:
                return None
            # 10 % goal bias: sample the other tree's root
            qr = self.rng.uniform(self.lo, self.hi) \
                if self.rng.random() > 0.1 else B[0][0]
            k, st = self._extend(name, A, qr, step)
            if st != "trapped":
                k2, st2 = self._connect(name, B, A[k][0], step)
                if st2 == "reached":
                    pa = self._trace(A, k)
                    pb = self._trace(B, k2)
                    path = pa + pb[::-1]
                    if not a_is_start:
                        path = path[::-1]
                    self.stats["nodes"] = len(A) + len(B)
                    return path
            A, B = B, A
            a_is_start = not a_is_start
        return None

    @staticmethod
    def _trace(tree, k):
        out = []
        while k != -1:
            out.append(tree[k][0])
            k = tree[k][1]
        return out[::-1]

    def _shortcut(self, name, path, tries=80):
        path = [np.asarray(q, float) for q in path]
        for _ in range(tries):
            if len(path) <= 2:
                break
            i, j = sorted(self.rng.choice(len(path), 2, replace=False))
            if j - i < 2:
                continue
            if self.edge_ok(name, path[i], path[j]):
                path = path[:i + 1] + path[j:]
        return path

    # ------------------------------------------------------------ timing
    def time_parametrise(self, name, path):
        """
        One smooth motion through all waypoints: a C2 cubic spline in joint
        space (chord-length parameter, clamped ends) driven by a single
        minimum-jerk time law, so the arm does not stop at every waypoint.
        Duration keeps the peak joint speed <= MAX_JOINT_VEL and the peak
        tool-point speed (needle tip / transducer face, by forward
        kinematics) <= MAX_TCP_SPEED. If the spline is not collision-free
        (it bulges between waypoints) the safe per-segment version is used.
        """
        path = [np.asarray(q, float) for q in path]
        if len(path) > 2:
            sp = SplinePath(path)
            if all(self.valid(name, sp.at(u)) for u in
                   np.linspace(0, 1, max(40, int(sp.length / 0.02)))):
                return self._arc_timed(name, sp)
        durs = []
        for a, b in zip(path[:-1], path[1:]):
            durs.append(self._duration(name,
                                       lambda u, a=a, b=b: a + (b - a) * u))
        return JointPath(path, durs)

    def _tool_point(self, name, q):
        if name == "needle":
            return self.scene.needle_axis(q)[0]
        fl, R = self.scene.flange_face(name, q)
        return fl + R[:, 2] * cfg.PROBE_LEN

    def _arc_timed(self, name, sp):
        """Time a spline by tool arc length (uniform min-jerk along it)."""
        us = np.linspace(0, 1, 201)
        Q = np.array([sp.at(u) for u in us])
        P = np.array([self._tool_point(name, q) for q in Q])
        dl = np.linalg.norm(np.diff(P, axis=0), axis=1)
        dj = np.abs(np.diff(Q, axis=0)).max(axis=1)
        # blend tool and joint arc so pure wrist rotations also progress
        w = dl / max(cfg.MAX_TCP_SPEED, 1e-6) + dj / cfg.MAX_JOINT_VEL
        cum = np.concatenate([[0.0], np.cumsum(w)])
        sig = cum / max(cum[-1], 1e-12)
        # with u(sigma), tool speed = (dl/dsigma) * sigma_dot; peak of
        # sigma_dot = 1.875 / T. Choose T so both limits hold everywhere.
        dsig = np.maximum(np.diff(sig), 1e-12)
        T = MINJERK_PEAK * max(float(np.max(dl / dsig)) / cfg.MAX_TCP_SPEED,
                               float(np.max(dj / dsig)) / cfg.MAX_JOINT_VEL,
                               1e-3)
        return SplineJointPath(sp, max(T, 0.4), sig, us)

    def _duration(self, name, f):
        """Min-jerk duration for the curve f(u), u in [0, 1]."""
        us = np.linspace(0, 1, 41)
        Q = np.array([f(u) for u in us])
        dq = np.abs(np.diff(Q, axis=0)).max(axis=1) / (us[1] - us[0])
        pts = []
        for q in Q[::2]:
            if name == "needle":
                pts.append(self.scene.needle_axis(q)[0])
            else:
                fl, R = self.scene.flange_face(name, q)
                pts.append(fl + R[:, 2] * cfg.PROBE_LEN)
        pts = np.array(pts)
        dp = np.linalg.norm(np.diff(pts, axis=0), axis=1) / (2 * (us[1] - us[0]))
        # peak speed of min-jerk = 1.875 * (path derivative) / T
        t_j = MINJERK_PEAK * float(dq.max()) / cfg.MAX_JOINT_VEL
        t_c = MINJERK_PEAK * float(dp.max()) / cfg.MAX_TCP_SPEED
        return max(t_j, t_c, 0.4)

    def validate(self, name, jp, step=None):
        """Dense re-check of a timed path along TIME (used by the tests)."""
        n = max(60, int(jp.T * cfg.CONTROL_HZ))
        prev = None
        for t in np.linspace(0, jp.T, n + 1):
            q = jp.q(t)
            if not self.valid(name, q):
                return False
            if prev is not None and np.max(np.abs(q - prev)) > \
                    (step or cfg.RRT_CHECK_STEP):
                mid = (q + prev) / 2
                if not self.valid(name, mid):
                    return False
            prev = q
        return True
