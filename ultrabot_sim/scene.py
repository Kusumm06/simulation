"""
PyBullet scene: table, patient, breast, two KUKA LBR iiwa arms on carts,
the ultrasound probe on arm 1 and the biopsy gun + needle on arm 2.

Layout (looking down, +x toward the patient's head, patient's right = +y):

                 [needle-arm cart]            (+y, patient's right)
        .--------------------------------------.
        |  legs        torso  (o) breast  head  |   table
        '--------------------------------------'
                 [probe-arm cart]             (-y, patient's left)

Kinematic simulation, like the reference project: the arms are put exactly
on their joint targets (resetJointState); there are no contact forces, no
tissue physics. Two PyBullet clients:

  main  - the displayed state (mirrored by the realistic renderer);
  ghost - invisible copies of both arms, their tools and the patient with
          collision shapes, used for IK iterations, RRT collision checks
          and clearance queries, so solving never disturbs the shown arms.
"""
from __future__ import annotations

import math

import numpy as np
import pybullet as p
import pybullet_data

from ultrabot_sim import config as cfg
from ultrabot_sim.needle import (TOOL_FLANGE_OFFSET, NeedleModel,
                                 gun_rotation_local, tool_length, tool_local)
from ultrabot_sim.us_geometry import ProbePose

SKIN = [0.88, 0.70, 0.60, 1.0]
BREAST = [0.90, 0.72, 0.64, 0.55]
STEEL = [0.70, 0.72, 0.76, 1.0]
DARK = [0.10, 0.11, 0.13, 1.0]
GOWN = [0.62, 0.74, 0.82, 1.0]
PROBE_COL = [0.86, 0.88, 0.90, 1.0]
GUN_COL = [0.22, 0.42, 0.70, 1.0]


def quat_from_R(R):
    """Rotation matrix -> PyBullet quaternion (x, y, z, w)."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(np.asarray(R, float)).as_quat().tolist()


def R_from_quat(q):
    return np.array(p.getMatrixFromQuaternion(q)).reshape(3, 3)


def frame_from_z(z, hint_x=None):
    """Rotation matrix whose z column is `z` (x as close to hint_x)."""
    z = np.asarray(z, float)
    z = z / np.linalg.norm(z)
    hx = np.array([1.0, 0, 0]) if hint_x is None else np.asarray(hint_x, float)
    x = hx - z * np.dot(hx, z)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(z, [0, 1.0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def home_seed(base, target):
    yaw = float(np.arctan2(target[1] - base[1], target[0] - base[0]))
    return [yaw, 0.75, 0.0, -1.5, 0.0, 0.9, 0.0]


def dls_ik(cid, body, target, z_axis, seed, x_axis=None, home=None,
           iters=None, tcp_local=None, axis_local=None):
    """
    Damped-least-squares IK on position + tool axis (+ optional x axis for
    full 6-DoF, used for the probe whose image plane orientation matters).
    Ported from the reference project's dls_ik: iterate from the previous
    solution (stays on the same elbow branch), weak null-space pull to the
    home posture, joint limits respected; an unreachable pose returns the
    nearest pose and its error so the caller can refuse it.
    tcp_local / axis_local: the controlled point and tool axis in the
    link-7 frame (default: the link origin and its z axis). The Jacobian is
    taken at the tool point (PyBullet's COM Jacobian shifted with the
    angular part), so the solver converges to < 1 micron.
    Returns (q, position_error_m, axis_error_deg).
    """
    tl = np.zeros(3) if tcp_local is None else np.asarray(tcp_local, float)
    al = np.array([0.0, 0.0, 1.0]) if axis_local is None else \
        np.asarray(axis_local, float)
    lo, hi = np.array(cfg.JOINT_LOWER), np.array(cfg.JOINT_UPPER)
    q = np.clip(np.array(seed, dtype=float), lo, hi)
    target = np.asarray(target, float)
    za = np.asarray(z_axis, float)
    za = za / np.linalg.norm(za)
    xa = None
    if x_axis is not None:
        xa = np.asarray(x_axis, float)
        xa = xa - za * np.dot(xa, za)
        xa = xa / np.linalg.norm(xa)
    lam2 = cfg.IK_DAMPING ** 2
    zero = [0.0] * 7
    n = iters or cfg.IK_ITERS
    for it in range(n):
        for j in range(7):
            p.resetJointState(body, j, q[j], physicsClientId=cid)
        ls = p.getLinkState(body, cfg.EE_LINK, computeForwardKinematics=True,
                            physicsClientId=cid)
        R = R_from_quat(ls[5])
        pos = np.array(ls[4]) + R @ tl
        e_pos = target - pos
        e_rot = _rot_error(R, za, xa, al)
        if np.linalg.norm(e_pos) < 2e-6 and np.linalg.norm(e_rot) < 2e-6:
            break
        jv, jw = p.calculateJacobian(body, cfg.EE_LINK, [0, 0, 0], list(q),
                                     zero, zero, physicsClientId=cid)
        jv, jw = np.array(jv), np.array(jw)
        r = pos - np.array(ls[0])                 # COM -> tool point
        sk = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
        J = np.vstack([jv - sk @ jw, jw])
        # near convergence use less damping -> sub-micron final accuracy
        lam = lam2 if np.linalg.norm(e_pos) > 1e-3 else lam2 * 0.01
        Jp = J.T @ np.linalg.inv(J @ J.T + lam * np.eye(6))
        dq = Jp @ np.hstack([e_pos, e_rot])
        if home is not None and cfg.IK_POSTURE_GAIN > 0:
            dq += (np.eye(7) - Jp @ J) @ (cfg.IK_POSTURE_GAIN *
                                          (np.asarray(home) - q))
        q = np.clip(q + np.clip(dq, -0.25, 0.25), lo, hi)
    for j in range(7):
        p.resetJointState(body, j, q[j], physicsClientId=cid)
    ls = p.getLinkState(body, cfg.EE_LINK, computeForwardKinematics=True,
                        physicsClientId=cid)
    R = R_from_quat(ls[5])
    err = float(np.linalg.norm(target - (np.array(ls[4]) + R @ tl)))
    ang = math.degrees(math.acos(float(np.clip((R @ al) @ za, -1, 1))))
    if xa is not None:
        ang = max(ang, math.degrees(math.acos(float(np.clip(R[:, 0] @ xa,
                                                            -1, 1)))))
    return list(q), err, ang


def _rot_error(R, za, xa=None, al=None):
    """Orientation error as a world-frame rotation vector (exact angle, not
    the small-angle cross product, so large errors do not stall the
    solver). 5-DoF: only the tool axis; 6-DoF: full frame (x given)."""
    if xa is not None:
        from scipy.spatial.transform import Rotation
        Rt = np.column_stack([xa, np.cross(za, xa), za])
        return Rotation.from_matrix(Rt @ R.T).as_rotvec()
    z = R[:, 2] if al is None else R @ al
    v = np.cross(z, za)
    s = float(np.linalg.norm(v))
    c = float(np.dot(z, za))
    ang = math.atan2(s, c)
    if s < 1e-12:
        if c > 0:
            return np.zeros(3)
        v = np.cross(z, R[:, 0])          # antiparallel: any perpendicular
        s = float(np.linalg.norm(v))
    return v / s * ang


def joint_margin_deg(q):
    q = np.asarray(q, float)
    lo, hi = np.array(cfg.JOINT_LOWER), np.array(cfg.JOINT_UPPER)
    return float(np.degrees(np.min(np.minimum(q - lo, hi - q))))


# --------------------------------------------------------------- geometry
def _box(cid, half, pos, colour, orn=None, collide=True):
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=colour,
                              physicsClientId=cid)
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half,
                                 physicsClientId=cid) if collide else -1
    return p.createMultiBody(0, col, vis, list(pos), orn or [0, 0, 0, 1],
                             physicsClientId=cid)


def _cyl(cid, r, h, pos, colour, orn=None, collide=True):
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=h,
                              rgbaColor=colour, physicsClientId=cid)
    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=h,
                                 physicsClientId=cid) if collide else -1
    return p.createMultiBody(0, col, vis, list(pos), orn or [0, 0, 0, 1],
                             physicsClientId=cid)


def _sphere(cid, r, pos, colour, collide=True):
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=r, rgbaColor=colour,
                              physicsClientId=cid)
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=r,
                                 physicsClientId=cid) if collide else -1
    return p.createMultiBody(0, col, vis, list(pos), physicsClientId=cid)


def _capsule_between(cid, a, b, r, colour, collide=True):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    L = float(np.linalg.norm(d))
    R = frame_from_z(d / L)
    orn = quat_from_R(R)
    vis = p.createVisualShape(p.GEOM_CAPSULE, radius=r, length=L,
                              rgbaColor=colour, physicsClientId=cid)
    col = p.createCollisionShape(p.GEOM_CAPSULE, radius=r, height=L,
                                 physicsClientId=cid) if collide else -1
    return p.createMultiBody(0, col, vis, ((a + b) / 2).tolist(), orn,
                             physicsClientId=cid)


def _mesh_body(cid, tm, colour, collide=True):
    V = np.asarray(tm.vertices, float)
    F = np.asarray(tm.faces, int).ravel().tolist()
    vis = p.createVisualShape(p.GEOM_MESH, vertices=V.tolist(), indices=F,
                              rgbaColor=colour, physicsClientId=cid)
    col = p.createCollisionShape(p.GEOM_MESH, vertices=V.tolist(),
                                 physicsClientId=cid) if collide else -1
    return p.createMultiBody(0, col, vis, [0, 0, 0], physicsClientId=cid)


def breast_world_mesh():
    """The breast solid in world coordinates (nominal probe pose)."""
    from ultrabot_sim.anatomy import breast_surface_mesh
    tm = breast_surface_mesh(cfg.US_WIDTH_MM / 2.0)
    pose = ProbePose.nominal()
    tm = tm.copy()
    tm.vertices = pose.probe_to_world(tm.vertices)
    return tm


def patient_boxes():
    """Collision approximation of the patient (world, m):
    [(kind, args)] built identically in both clients."""
    pad = cfg.TABLE_TOP_Z + cfg.TABLE_PAD
    cz = cfg.CHEST_Z
    return [
        ("box", ([0.48, 0.17, (cz - pad) / 2 - 0.004],
                 [0.16, 0.0, (pad + cz) / 2 - 0.004])),             # torso
        ("box", ([0.30, 0.15, 0.06], [-0.58, 0.0, pad + 0.06])),    # legs
        ("sphere", (0.10, [0.74, 0.0, pad + 0.10])),                # head
        ("capsule", ([0.56, 0.20, pad + 0.07], [0.98, 0.17, pad + 0.07],
                     0.045)),                                       # R arm up
        ("capsule", ([0.50, -0.21, pad + 0.06], [0.05, -0.22, pad + 0.05],
                     0.045)),                                       # L arm
    ]


class Scene:
    ARMS = ("probe", "needle")

    def __init__(self, gui=False):
        self.gui = gui
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        cid = self.client
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=cid)
        if gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=cid)
        self.groups = {}
        self.needle = NeedleModel()
        self._breast_tm = breast_world_mesh()

        def grouped(name, fn, *a):
            n0 = p.getNumBodies(physicsClientId=cid)
            fn(cid, *a)
            self.groups.setdefault(name, []).extend(
                p.getBodyUniqueId(i, physicsClientId=cid)
                for i in range(n0, p.getNumBodies(physicsClientId=cid)))

        grouped("room", self._build_room)
        grouped("table", self._build_table)
        grouped("patient", self._build_patient)
        grouped("breast", self._build_breast)
        grouped("monitors", self._build_monitors)
        self.bases = {"probe": np.asarray(cfg.CART_PROBE, float),
                      "needle": np.asarray(cfg.CART_NEEDLE, float)}
        self.arms = {}
        for name in self.ARMS:
            grouped("carts", self._build_cart, self.bases[name])
            self.arms[name] = p.loadURDF(
                "kuka_iiwa/model.urdf", basePosition=self.bases[name].tolist(),
                useFixedBase=True, physicsClientId=cid)
        grouped("tools", self._build_tools)
        self.q = {n: [0.0] * 7 for n in self.ARMS}
        self._build_ghost()
        self.home_q = {}
        self.ready_q = {}
        self._home_arms()
        self._label_text = {}

    # ------------------------------------------------------------ building
    def _build_room(self, cid):
        _box(cid, [2.0, 2.0, 0.01], [0, 0, -0.01], [0.55, 0.60, 0.62, 1])

    def _build_table(self, cid):
        t = cfg.TABLE_TOP_Z
        _box(cid, [0.15, 0.12, (t - 0.1) / 2], [0.0, 0, (t - 0.1) / 2 + 0.05],
             [0.85, 0.86, 0.86, 1])
        _box(cid, [cfg.TABLE_LEN / 2, cfg.TABLE_HALF_W, cfg.TABLE_PAD / 2 + 0.02],
             [cfg.TABLE_CENTER_X, 0, t + cfg.TABLE_PAD / 2 - 0.02],
             [0.12, 0.14, 0.18, 1])

    def _build_patient(self, cid):
        for kind, a in patient_boxes():
            if kind == "box":
                _box(cid, a[0], a[1], GOWN)
            elif kind == "sphere":
                _sphere(cid, a[0], a[1], SKIN)
            else:
                _capsule_between(cid, a[0], a[1], a[2], SKIN)

    def _build_breast(self, cid):
        _mesh_body(cid, self._breast_tm, BREAST)

    def _build_monitors(self, cid):
        x, y, z = cfg.MONITOR_POS
        yaw = cfg.MONITOR_YAW
        q = p.getQuaternionFromEuler([0, 0, yaw])
        _cyl(cid, 0.05, z, [x, y, z / 2], STEEL, collide=False)
        c, s = math.cos(yaw), math.sin(yaw)
        for k in (-1, 1):
            ox, oy = -k * 0.27 * s, k * 0.27 * c
            _box(cid, [0.02, 0.25, 0.16], [x + ox, y + oy, z], DARK, orn=q,
                 collide=False)

    def _build_cart(self, cid, base):
        x, y, z = base
        _box(cid, [0.15, 0.15, z / 2], [x, y, z / 2], [0.90, 0.91, 0.92, 1])

    def _build_tools(self, cid):
        bx, by, bz = cfg.PROBE_BODY
        self.probe_body = _box(cid, [bx / 2, by / 2, bz / 2], [0, 0, 2],
                               PROBE_COL, collide=False)
        self.gun_body = _cyl(cid, cfg.GUN_RADIUS, cfg.GUN_LEN, [0, 0, 2],
                             GUN_COL, collide=False)
        L = cfg.NEEDLE_LEN_MM / 1000.0
        self.cannula_body = _cyl(cid, cfg.NEEDLE_OD_MM / 2000.0, L, [0, 0, 2],
                                 [0.80, 0.82, 0.86, 1], collide=False)
        self.stylet_body = _cyl(cid, cfg.NEEDLE_OD_MM / 2000.0 * 0.7,
                                cfg.THROW_MM / 1000.0 + 0.004, [0, 0, 2],
                                [0.95, 0.80, 0.25, 1], collide=False)

    def _build_ghost(self):
        g = self._gid = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=g)
        self.ghost = {}
        for name in self.ARMS:
            self.ghost[name] = p.loadURDF(
                "kuka_iiwa/model.urdf", basePosition=self.bases[name].tolist(),
                useFixedBase=True, physicsClientId=g)
        self.obstacles = {}
        t = cfg.TABLE_TOP_Z
        self.obstacles["table"] = _box(
            g, [cfg.TABLE_LEN / 2, cfg.TABLE_HALF_W, cfg.TABLE_PAD / 2 + 0.02],
            [cfg.TABLE_CENTER_X, 0, t + cfg.TABLE_PAD / 2 - 0.02], DARK)
        pats = []
        for kind, a in patient_boxes():
            if kind == "box":
                pats.append(_box(g, a[0], a[1], GOWN))
            elif kind == "sphere":
                pats.append(_sphere(g, a[0], a[1], SKIN))
            else:
                pats.append(_capsule_between(g, a[0], a[1], a[2], SKIN))
        self.obstacles["patient"] = pats
        self.obstacles["breast"] = _mesh_body(g, self._breast_tm, BREAST)
        bx, by, bz = cfg.PROBE_BODY
        self.g_probe = _box(g, [bx / 2, by / 2, bz / 2], [0, 0, 3], PROBE_COL)
        self.g_gun = _cyl(g, cfg.GUN_RADIUS, cfg.GUN_LEN, [0, 0, 3], GUN_COL)
        self.g_needle = _cyl(g, cfg.NEEDLE_OD_MM / 2000.0,
                             cfg.NEEDLE_LEN_MM / 1000.0, [0, 0, 3], STEEL)
        # the arms are kinematic: no contacts between anything
        for a in self.ghost.values():
            p.setCollisionFilterGroupMask(a, -1, 0, 0, physicsClientId=g)

    # ------------------------------------------------------------ homing
    def _pose_cache_key(self):
        import hashlib
        import json
        keys = ["CART_PROBE", "CART_NEEDLE", "PROBE_SITE", "PROBE_X_AXIS",
                "PROBE_Z_AXIS", "PROBE_LEN", "GUN_LEN", "NEEDLE_LEN_MM",
                "NEEDLE_MOUNT_DEG", "US_WIDTH_MM", "PRE_ENTRY_MM",
                "STANDOFF_MM", "NEEDLE_HOME_POS", "BREAST_HEIGHT_MM",
                "BREAST_FALLOFF_MM", "IK_TOL", "JOINT_MARGIN_DEG"]
        raw = json.dumps({k: np.asarray(getattr(cfg, k)).round(9).tolist()
                          for k in keys}, sort_keys=True) + "v3"
        return hashlib.sha1(raw.encode()).hexdigest()[:16]

    def _load_pose_cache(self):
        """Solved home / ready poses from a previous start (same config),
        re-verified by forward kinematics before they are trusted."""
        import json
        path = cfg.RUNS_DIR / "cache" / f"arm_poses_{self._pose_cache_key()}.json"
        if not path.is_file():
            return False
        try:
            d = json.loads(path.read_text())
        except Exception:                                   # noqa: BLE001
            return False
        pp = ProbePose.nominal()
        pr = self.probe_pose(d["probe_ready"])
        if np.linalg.norm(pr.t - pp.t) > cfg.IK_TOL or \
                np.degrees(np.arccos(np.clip(pr.R[:, 0] @ pp.R[:, 0], -1, 1))) \
                > cfg.IK_AXIS_TOL_DEG:
            return False
        self.ready_q["probe"] = d["probe_ready"]
        self.home_q["probe"] = d["probe_home"]
        self.contact_err = tuple(d["contact_err"])
        self.ready_side_q = {1: d["needle_ready_p"], -1: d["needle_ready_m"]}
        self.ready_q["needle"] = d["needle_ready_p"]
        self.home_q["needle"] = d["needle_home"]
        self.home_err = d["home_err"]
        self.home_ok = bool(d["home_ok"])
        self.set_q("probe", self.home_q["probe"])
        self.set_q("needle", self.home_q["needle"])
        return True

    def _save_pose_cache(self):
        import json
        path = cfg.RUNS_DIR / "cache" / f"arm_poses_{self._pose_cache_key()}.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "probe_ready": list(map(float, self.ready_q["probe"])),
                "probe_home": list(map(float, self.home_q["probe"])),
                "contact_err": list(map(float, self.contact_err)),
                "needle_ready_p": list(map(float, self.ready_side_q[1])),
                "needle_ready_m": list(map(float, self.ready_side_q[-1])),
                "needle_home": list(map(float, self.home_q["needle"])),
                "home_err": {k: (list(map(float, v)) if k == "probe" else
                                 [bool(x) for x in v])
                             for k, v in self.home_err.items()},
                "home_ok": bool(self.home_ok)}))
        except Exception:                                   # noqa: BLE001
            pass

    def _home_arms(self):
        if self._load_pose_cache():
            return
        self._solve_homes()
        if self.home_ok:
            self._save_pose_cache()

    def _solve_homes(self):
        site = np.asarray(cfg.PROBE_SITE, float)
        # probe arm: contact pose first (6-DoF, image plane orientation
        # matters), then the home pose 22 cm straight above it
        pp = ProbePose.nominal()
        qc, ec, ac, mc, okc = self.ik_probe_robust(
            pp, home_seed(self.bases["probe"], site), n_random=40, iters=400)
        self.ready_q["probe"] = qc
        self.contact_err = (ec, ac)
        up = pp.copy()
        up.t = pp.t + np.array([0, 0, 0.22])
        q, e, a, m, ok = self.ik_probe_robust(up, qc, n_random=24, iters=400)
        self.home_q["probe"] = q
        self.set_q("probe", q)
        # needle arm: gun parked above the patient's right side
        tip = np.asarray(cfg.NEEDLE_HOME_POS, float)
        d = site - tip
        d /= np.linalg.norm(d)
        q2, e2, a2, m2, ok2 = self.ik_needle_robust(
            tip, d, home_seed(self.bases["needle"], site), n_random=40,
            iters=400)
        self.home_q["needle"] = q2
        self.set_q("needle", q2)
        # ready poses, one per side of the probe: the needle already pointing
        # along a nominal in-plane path, 11 cm back from the skin. They are
        # derived by WALKING the tool pose continuously (IK seeded step by
        # step) through a pose above the breast, so the parked, +x and -x
        # poses share one arm configuration branch and moving between them
        # is a short, smooth motion.
        pp_c = ProbePose.nominal()

        def ready_pose(side):
            W2 = cfg.US_WIDTH_MM / 2.0
            E = np.array([side * (W2 + 15.0), 0.3])
            A = np.array([0.0, 12.0])
            dd = (A - E) / np.linalg.norm(A - E)
            back = E - dd * (cfg.PRE_ENTRY_MM + cfg.STANDOFF_MM + 40.0)
            return (pp_c.probe_to_world([back[0], 0.0, back[1]]),
                    pp_c.dir_probe_to_world([dd[0], 0.0, dd[1]]))
        self.set_q("probe", qc)
        rtip, rd = ready_pose(1)
        q_r, e_r, a_r, m_r, ok_r = self.ik_needle_robust(
            rtip, rd, q2, n_random=60, iters=400)
        q_m, ok_m = self.best_branch(ready_pose(-1), q_r)
        home = (rtip - rd * 0.05 + np.array([0.0, 0.0, 0.03]), rd)
        q_h, ok_h = self.ik_walk(q_r, (rtip, rd), home)
        rerr = []
        for q_, ok_ in ((q_r, ok_r), (q_m, ok_m), (q_h, ok_h)):
            ok_ = ok_ and self.collision_free("needle", q_)
            rerr.append(ok_)
        self.set_q("probe", q)
        self.ready_side_q = {1: q_r, -1: q_m}
        self.ready_q["needle"] = q_r
        self.home_q["needle"] = q_h
        self.set_q("needle", q_h)
        self.home_err = {"probe": (e, a), "needle_ready_ok": rerr}
        self.home_ok = ok and okc and all(rerr)

    # ------------------------------------------------------------ kinematics
    def set_q(self, name, q):
        """Put an arm exactly on q (kinematic) and move its tool with it."""
        q = list(map(float, q))
        self.q[name] = q
        arm = self.arms[name]
        for j, v in enumerate(q):
            p.resetJointState(arm, j, v, physicsClientId=self.client)
        self._place_tool(name, self.client)

    def fk(self, name, q=None, client=None):
        """(link-7 position, rotation) for q (ghost) or the shown arm."""
        if q is None:
            ls = p.getLinkState(self.arms[name], cfg.EE_LINK,
                                computeForwardKinematics=True,
                                physicsClientId=self.client)
        else:
            g = self.ghost[name]
            for j, v in enumerate(q):
                p.resetJointState(g, j, v, physicsClientId=self._gid)
            ls = p.getLinkState(g, cfg.EE_LINK, computeForwardKinematics=True,
                                physicsClientId=self._gid)
        return np.array(ls[4]), R_from_quat(ls[5])

    def flange_face(self, name, q=None):
        pos, R = self.fk(name, q)
        return pos + R[:, 2] * TOOL_FLANGE_OFFSET, R

    def probe_pose(self, q=None):
        """Probe frame from arm 1's forward kinematics (the US image plane)."""
        f, R = self.flange_face("probe", q)
        return ProbePose.from_flange(f, R)

    def needle_frame(self, q=None):
        """(gun rear point, needle direction, link-7 rotation) as the arm
        holds them (forward kinematics)."""
        pos, R = self.fk("needle", q)
        face, a, _ = tool_local()
        return pos + R @ face, R @ a, R

    def needle_axis(self, q=None):
        """(cocked tip, direction) of the needle as the arm holds it."""
        g, d, _ = self.needle_frame(q)
        return g + d * tool_length(), d

    def ik_probe(self, pose: ProbePose, seed=None, iters=None):
        target = pose.t - pose.R[:, 2] * (cfg.PROBE_LEN + TOOL_FLANGE_OFFSET)
        seed = seed if seed is not None else self.q["probe"]
        return dls_ik(self._gid, self.ghost["probe"], target, pose.R[:, 2],
                      seed, x_axis=pose.R[:, 0],
                      home=self.home_q.get("probe"), iters=iters)

    def ik_needle(self, tip, d, seed=None, iters=None):
        _, a, tcp = tool_local()
        seed = seed if seed is not None else self.q["needle"]
        return dls_ik(self._gid, self.ghost["needle"], tip, d, seed,
                      home=self.home_q.get("needle"), iters=iters,
                      tcp_local=tcp, axis_local=a)

    def ik_robust(self, name, solve, seed=None, n_random=16, need_margin=None):
        """
        Multi-start IK. `solve(seed)` -> (q, err, ang). Tries the given seed
        (continuity), the home / ready postures, then deterministic random
        seeds, and keeps the best solution: accurate first (IK_TOL,
        IK_AXIS_TOL_DEG), then largest joint-limit margin. Returns
        (q, err_m, axis_deg, margin_deg, ok).
        """
        need = cfg.JOINT_MARGIN_DEG if need_margin is None else need_margin
        rng = np.random.default_rng(1234)
        seeds = []
        if seed is not None:
            seeds.append(list(seed))
        for qd in (self.q.get(name), self.ready_q.get(name),
                   self.home_q.get(name)):
            if qd is not None:
                seeds.append(list(qd))
        lo, hi = np.array(cfg.JOINT_LOWER), np.array(cfg.JOINT_UPPER)
        for _ in range(n_random):
            seeds.append((rng.uniform(lo, hi) * 0.8).tolist())
        best = None
        for sd in seeds:
            q, e, a = solve(sd)
            m = joint_margin_deg(q)
            acc = e < cfg.IK_TOL and a < cfg.IK_AXIS_TOL_DEG
            key = (acc and m >= need, acc, m if acc else -e)
            if best is None or key > best[0]:
                best = (key, q, e, a, m)
            if acc and m >= need and sd is seeds[0]:
                break                     # continuity seed is good: keep it
            if acc and m >= 15.0:
                break
        _, q, e, a, m = best
        return q, e, a, m, (e < cfg.IK_TOL and a < cfg.IK_AXIS_TOL_DEG
                            and m >= need)

    def best_branch(self, pose, q_from, n_seeds=60, min_margin=10.0):
        """Among many IK solutions of a needle pose, the collision-free one
        (margin >= min_margin) reachable from q_from by the shortest direct
        joint motion. Deterministic (fixed seeds)."""
        tip, d = pose
        rng = np.random.default_rng(77)
        lo, hi = np.array(cfg.JOINT_LOWER), np.array(cfg.JOINT_UPPER)
        best = None
        for k in range(n_seeds):
            sd = q_from if k == 0 else (rng.uniform(lo, hi) * 0.85).tolist()
            q, e, a = self.ik_needle(tip, d, sd, iters=300)
            m = joint_margin_deg(q)
            if e >= cfg.IK_TOL or a >= cfg.IK_AXIS_TOL_DEG or m < min_margin:
                continue
            if not self.collision_free("needle", q):
                continue
            cost = float(np.max(np.abs(np.asarray(q) - np.asarray(q_from))))
            if best is None or cost < best[0]:
                best = (cost, q)
        if best is None:
            q, e, a, m, ok = self.ik_needle_robust(tip, d, q_from, n_random=40)
            return q, ok
        return best[1], True

    def ik_walk(self, q0, pose0, pose1, step_m=0.01, step_deg=3.0):
        """Move the needle tool pose from pose0 to pose1 (tip, direction)
        in small steps, IK seeded by the previous step: stays on the same
        arm branch. Returns (q, ok)."""
        t0, d0 = (np.asarray(v, float) for v in pose0)
        t1, d1 = (np.asarray(v, float) for v in pose1)
        ang = math.degrees(math.acos(float(np.clip(d0 @ d1 / (
            np.linalg.norm(d0) * np.linalg.norm(d1)), -1, 1))))
        n = int(max(np.linalg.norm(t1 - t0) / step_m, ang / step_deg, 1)) + 1
        q = list(q0)
        from ultrabot_sim.control import _slerp_dir
        e = a = 0.0
        for k in range(1, n + 1):
            u = k / n
            tip = t0 + (t1 - t0) * u
            d = _slerp_dir(d0, d1, u)
            q, e, a = self.ik_needle(tip, d, q, iters=150)
        ok = e < cfg.IK_TOL and a < cfg.IK_AXIS_TOL_DEG and \
            joint_margin_deg(q) >= cfg.JOINT_MARGIN_DEG
        return q, ok

    def ik_needle_robust(self, tip, d, seed=None, n_random=16, iters=200):
        return self.ik_robust("needle",
                              lambda sd: self.ik_needle(tip, d, sd, iters),
                              seed, n_random)

    def ik_probe_robust(self, pose, seed=None, n_random=16, iters=300):
        return self.ik_robust("probe",
                              lambda sd: self.ik_probe(pose, sd, iters),
                              seed, n_random)

    # ------------------------------------------------------------ tools
    def _place_tool(self, name, cid, q=None):
        if cid == self.client:
            f, R = self.flange_face(name)
        else:
            g = self.ghost[name]
            ls = p.getLinkState(g, cfg.EE_LINK, computeForwardKinematics=True,
                                physicsClientId=cid)
            R = R_from_quat(ls[5])
            f = np.array(ls[4]) + R[:, 2] * TOOL_FLANGE_OFFSET
        orn = quat_from_R(R)
        z = R[:, 2]
        if name == "probe":
            bz = cfg.PROBE_BODY[2]
            face = f + z * cfg.PROBE_LEN
            pos = face - z * (bz / 2)
            body = self.probe_body if cid == self.client else self.g_probe
            p.resetBasePositionAndOrientation(body, pos.tolist(), orn,
                                              physicsClientId=cid)
            return
        face, a, _ = tool_local()
        f = (f - R[:, 2] * TOOL_FLANGE_OFFSET) + R @ face
        Rg = R @ gun_rotation_local()
        orn = quat_from_R(Rg)
        z = Rg[:, 2]
        gun_c = f + z * (cfg.GUN_LEN / 2)
        hub = f + z * cfg.GUN_LEN
        if cid == self.client:
            p.resetBasePositionAndOrientation(self.gun_body, gun_c.tolist(),
                                              orn, physicsClientId=cid)
            L = cfg.NEEDLE_LEN_MM / 1000.0
            ce = self.needle.cannula_ext / 1000.0
            se = self.needle.stylet_ext / 1000.0
            p.resetBasePositionAndOrientation(
                self.cannula_body, (hub + z * (L / 2 + ce)).tolist(), orn,
                physicsClientId=cid)
            sl = cfg.THROW_MM / 1000.0 + 0.004
            # stylet visible beyond the cannula tip when fired ahead of it
            stip = hub + z * (L + se)
            p.resetBasePositionAndOrientation(
                self.stylet_body, (stip - z * sl / 2).tolist(), orn,
                physicsClientId=cid)
        else:
            p.resetBasePositionAndOrientation(self.g_gun, gun_c.tolist(), orn,
                                              physicsClientId=cid)
            L = cfg.NEEDLE_LEN_MM / 1000.0
            p.resetBasePositionAndOrientation(
                self.g_needle, (hub + z * L / 2).tolist(), orn,
                physicsClientId=cid)

    def refresh_tools(self):
        for n in self.ARMS:
            self._place_tool(n, self.client)

    # ------------------------------------------------------------ collisions
    def _ghost_set(self, name, q):
        g = self.ghost[name]
        for j, v in enumerate(q):
            p.resetJointState(g, j, v, physicsClientId=self._gid)
        self._place_tool(name, self._gid)

    def sync_ghost(self):
        for n in self.ARMS:
            self._ghost_set(n, self.q[n])

    def _min_dist(self, a, b, query, link_a=None):
        kw = {}
        if link_a is not None:
            kw["linkIndexA"] = link_a
        pts = p.getClosestPoints(a, b, distance=query,
                                 physicsClientId=self._gid, **kw)
        if not pts:
            return query, None
        best = min(pts, key=lambda c: c[8])
        return float(best[8]), best[3]

    def clearances(self, needle_q=None, probe_q=None, query=None):
        """Distances (m) used by the safety layer and the planner, on the
        ghost with the given (or current) joint positions:
            arm_arm          probe arm + probe vs needle arm + gun
            needle_links     needle-arm links 0..5 vs patient / table
            needle_wrist     needle-arm link 6 (tool mount) vs patient
            gun_patient      gun body vs patient + breast
            gun_probe        gun vs probe housing
            probe_links      probe-arm links vs patient
        """
        Q = query or cfg.ARM_CLEAR_QUERY
        self._ghost_set("probe", probe_q if probe_q is not None
                        else self.q["probe"])
        self._ghost_set("needle", needle_q if needle_q is not None
                        else self.q["needle"])
        gA, gB = self.ghost["probe"], self.ghost["needle"]
        pats = self.obstacles["patient"] + [self.obstacles["table"]]
        out = {}
        aa = [self._min_dist(gA, gB, Q)[0], self._min_dist(self.g_probe, gB, Q)[0],
              self._min_dist(gA, self.g_gun, Q)[0]]
        out["arm_arm"] = min(aa)
        links = []
        for lk in range(0, 6):
            for ob in pats + [self.obstacles["breast"]]:
                links.append(self._min_dist(gB, ob, Q, link_a=lk)[0])
        out["needle_links"] = min(links)
        out["needle_wrist"] = min(self._min_dist(gB, ob, Q, link_a=6)[0]
                                  for ob in pats + [self.obstacles["breast"]])
        out["gun_patient"] = min(self._min_dist(self.g_gun, ob, Q)[0]
                                 for ob in pats + [self.obstacles["breast"]])
        out["gun_probe"] = self._min_dist(self.g_gun, self.g_probe, Q)[0]
        pl = []
        for lk in range(0, 7):
            for ob in pats:
                pl.append(self._min_dist(gA, ob, Q, link_a=lk)[0])
        out["probe_links"] = min(pl)
        return out

    def collision_free(self, name, q, link_margin=None, tool_margin=None,
                       allow_needle_breast=True):
        """RRT validity check for one arm configuration (ghost)."""
        lm = cfg.RRT_LINK_MARGIN if link_margin is None else link_margin
        tm = cfg.RRT_TOOL_MARGIN if tool_margin is None else tool_margin
        self._ghost_set(name, q)
        arm = self.ghost[name]
        other = self.ghost["probe" if name == "needle" else "needle"]
        obs = self.obstacles["patient"] + [self.obstacles["table"],
                                           self.obstacles["breast"]]
        for ob in obs + [other]:
            if p.getClosestPoints(arm, ob, distance=lm,
                                  physicsClientId=self._gid):
                return False
        if name == "needle":
            for ob in obs + [other, self.g_probe]:
                if p.getClosestPoints(self.g_gun, ob, distance=tm,
                                      physicsClientId=self._gid):
                    return False
            for ob in obs + [other, self.g_probe]:
                if allow_needle_breast and ob == self.obstacles["breast"]:
                    continue
                if p.getClosestPoints(self.g_needle, ob, distance=0.002,
                                      physicsClientId=self._gid):
                    return False
            if allow_needle_breast:
                # the needle may approach the skin, never cross into it
                if p.getClosestPoints(self.g_needle, self.obstacles["breast"],
                                      distance=-0.0005,
                                      physicsClientId=self._gid):
                    return False
        else:
            if p.getClosestPoints(self.g_probe, other, distance=tm,
                                  physicsClientId=self._gid):
                return False
        return True

    # ------------------------------------------------------------ fallback view
    def render(self, size=(744, 440), view=None):
        """PyBullet's own renderer (fallback when pyrender is unavailable)."""
        w, h = size
        name, dist, yaw, pitch, tgt = view or cfg.OR_VIEWS[0]
        V = p.computeViewMatrixFromYawPitchRoll(tgt, dist, yaw, pitch, 0, 2,
                                                physicsClientId=self.client)
        P = p.computeProjectionMatrixFOV(cfg.OR_FOV, w / h, 0.03, 12.0,
                                         physicsClientId=self.client)
        _, _, rgb, _, _ = p.getCameraImage(w, h, V, P,
                                           renderer=p.ER_TINY_RENDERER,
                                           physicsClientId=self.client)
        return np.reshape(np.array(rgb, dtype=np.uint8), (h, w, 4))[:, :, :3]

    def disconnect(self):
        for c in (getattr(self, "_gid", None), self.client):
            try:
                if c is not None:
                    p.disconnect(c)
            except Exception:                               # noqa: BLE001
                pass


def check_reachability(scene: Scene, verbose=True):
    """Sweep entry points around the probe and insertion depths, solve the
    real IK on the ghost and report what fraction the needle arm reaches."""
    from ultrabot_sim.us_geometry import USCalib
    from ultrabot_sim.anatomy import Anatomy
    cal = USCalib.from_config()
    an = Anatomy(cal, None)
    pose = scene.probe_pose(scene.ready_q.get("probe"))
    tot = ok = 0
    seed = scene.home_q["needle"]
    for s in (-1, 1):
        for off in np.linspace(cfg.ENTRY_MIN_OFFSET_MM,
                               cfg.ENTRY_MAX_OFFSET_MM, 7):
            x = s * (cal.half_width_mm + off)
            E = np.array([x, float(an.skin_depth(x))])
            for depth in (8.0, 16.0, 24.0):
                A = np.array([0.0, depth])
                d = (A - E) / np.linalg.norm(A - E)
                for sdep in (0.0, np.linalg.norm(A - E)):
                    tip = E + d * sdep
                    tw = pose.probe_to_world([tip[0], 0.0, tip[1]])
                    dw = pose.dir_probe_to_world([d[0], 0.0, d[1]])
                    q, e, a, m, good = scene.ik_needle_robust(tw, dw, seed)
                    tot += 1
                    if good:
                        ok += 1
                        seed = q
    frac = ok / max(tot, 1)
    if verbose:
        print(f"  needle arm reaches {100 * frac:.1f}% of {tot} sampled "
              f"in-plane needle poses")
    return frac
