"""
Realistic posed humans for the OR panel, built from the MakeHuman base mesh
(adapted from the reference project's humans.py: same rig, skinning and
two-bone IK; new poses for a supine breast-biopsy patient with the right
arm raised, and a standing radiologist).

Source (all CC0 1.0, MakeHuman Community / Data Collection AB):
    base.obj             the MakeHuman base body (19k vertices)
    default.mhskel       163-bone skeleton
    default_weights.mhw  skinning weights
    high-poly eyes       eye mesh + texture
They are fetched from github.com/makehumancommunity/makehuman by
tools/get_assets.py into assets/src/makehuman/.

The body is posed with linear-blend skinning. Arms and legs are placed with
analytic two-bone IK so the hands, knees and feet land EXACTLY where the
scene expects them (the surgeon's hands on his instruments, the patient's
legs in the low-lithotomy stirrups). Clothing is done by material regions on
the skinned body: scrubs, gloves, surgical cap and mask for the surgeon; the
patient is under drapes (built in props.py) with a cap and taped eyes.
"""
import json
import os

import numpy as np
import trimesh

from ultrabot_sim import config as cfg

MH = os.path.join(cfg.ASSET_DIR, "src", "makehuman")
META = {}                 # extra points other modules need (patient mouth)


# ------------------------------------------------------------------ loading
def _load_obj(path):
    V, F, G = [], [], []
    group = None
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                V.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("g "):
                group = line.split()[1]
            elif line.startswith("f "):
                idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
                for k in range(1, len(idx) - 1):        # quads -> tris
                    F.append([idx[0], idx[k], idx[k + 1]])
                    G.append(group)
    return np.array(V), np.array(F), np.array(G)


class _Rig:
    _cache = None

    def __init__(self):
        V, F, G = _load_obj(os.path.join(MH, "base.obj"))
        self.V = V
        self.F = F[G == "body"]
        sk = json.load(open(os.path.join(MH, "default.mhskel")))
        self.bones = sk["bones"]
        jpos = {k: V[v].mean(axis=0) for k, v in sk["joints"].items()}
        self.head = {b: jpos[d["head"]] for b, d in self.bones.items()}
        self.tail = {b: jpos[d["tail"]] for b, d in self.bones.items()}
        w = json.load(open(os.path.join(MH, "default_weights.mhw")))["weights"]
        self.weights = {b: np.array(v) for b, v in w.items()}
        # topological order (parents first)
        order, seen = [], set()

        def visit(b):
            if b in seen:
                return
            par = self.bones[b]["parent"]
            if par:
                visit(par)
            seen.add(b)
            order.append(b)
        for b in self.bones:
            visit(b)
        self.order = order
        # dominant bone per vertex (for clothing regions)
        n = len(V)
        best = np.zeros(n)
        self.dom = np.array([""] * n, dtype=object)
        for b, arr in self.weights.items():
            idx, wt = arr[:, 0].astype(int), arr[:, 1]
            m = wt > best[idx]
            best[idx[m]] = wt[m]
            self.dom[idx[m]] = b
        # eyes (CC0 proxy fitted to the base mesh)
        self.eyes = self._fit_eyes()

    @classmethod
    def get(cls):
        if cls._cache is None:
            cls._cache = cls()
        return cls._cache

    def _fit_eyes(self):
        d = os.path.join(MH, "eyes", "high-poly")
        objp, clo = os.path.join(d, "high-poly.obj"), \
            os.path.join(d, "high-poly.mhclo")
        if not (os.path.isfile(objp) and os.path.isfile(clo)):
            return None
        EV, EF, _ = _load_obj(objp)
        scale = {}
        refs, rows, in_verts = {}, [], False
        for line in open(clo):
            t = line.split()
            if not t or t[0].startswith("#"):
                continue
            if t[0] in ("x_scale", "y_scale", "z_scale"):
                a, b, ref = int(t[1]), int(t[2]), float(t[3])
                ax = "xyz".index(t[0][0])
                scale[ax] = abs(self.V[a][ax] - self.V[b][ax]) / ref
            elif t[0] == "verts":
                in_verts = True
            elif in_verts and len(t) == 9:
                rows.append([float(x) for x in t])
            elif in_verts and len(t) == 1:
                rows.append([float(t[0]), 0, 0, 1, 0, 0, 0, 0, 0])
        rows = np.array(rows)
        i = rows[:, :3].astype(int)
        w = rows[:, 3:6]
        off = rows[:, 6:9] * np.array([scale.get(0, 1), scale.get(1, 1),
                                       scale.get(2, 1)])
        P = (self.V[i] * w[:, :, None]).sum(axis=1) + off
        return P[:len(EV)], EF


# --------------------------------------------------------------- math bits
def _rot_between(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(np.dot(a, b))
    if c < -0.9999:
        ax = np.cross(a, [1, 0, 0])
        if np.linalg.norm(ax) < 1e-6:
            ax = np.cross(a, [0, 1, 0])
        ax /= np.linalg.norm(ax)
        return 2 * np.outer(ax, ax) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def _axis_angle(axis, ang):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def _two_bone(root, mid, end, target, pole):
    """New mid position so |root-mid|, |mid-end| keep their lengths."""
    a = np.linalg.norm(mid - root)
    b = np.linalg.norm(end - mid)
    d = target - root
    L = float(np.clip(np.linalg.norm(d), abs(a - b) + 1e-6, a + b - 1e-6))
    u = d / np.linalg.norm(d)
    x = (a * a - b * b + L * L) / (2 * L)
    h = np.sqrt(max(a * a - x * x, 0.0))
    pv = pole - root
    perp = pv - u * np.dot(pv, u)
    perp /= max(np.linalg.norm(perp), 1e-9)
    return root + u * x + perp * h, root + u * L


# ------------------------------------------------------------------ posing
class Posed:
    """
    Pose = world-space rotation per bone (unlisted bones inherit their
    parent's rotation). Rest geometry is first placed in the world by
    `to_world` (scale + rotation + translation).
    """

    def __init__(self, to_world_R, to_world_s, to_world_t, zflat=None):
        self.rig = _Rig.get()
        self.R0, self.s, self.t = to_world_R, to_world_s, to_world_t
        self.zflat = zflat              # (base_z, k): thin the body front-back
        self.V = self._w(self.rig.V)
        self.head = {b: self._w(h) for b, h in self.rig.head.items()}
        self.tail = {b: self._w(t) for b, t in self.rig.tail.items()}
        self.G = {}

    def _w(self, P):
        W = (np.asarray(P) * self.s) @ self.R0.T + self.t
        if self.zflat is not None:
            base, k = self.zflat
            W = W.copy()
            W[..., 2] = base + (W[..., 2] - base) * k
        return W

    def rotate(self, bone, R):
        self.G[bone] = R

    def _transforms(self):
        M = {}
        for b in self.rig.order:
            par = self.rig.bones[b]["parent"]
            if par is None:
                Gp, Mp = np.eye(3), (np.eye(3), np.zeros(3))
            else:
                Gp = M[par][0]
                Mp = M[par]
            newhead = Mp[0] @ self.head[b] + Mp[1]
            Gb = self.G.get(b, Gp if par else np.eye(3))
            # v' = Gb (v - head) + newhead
            M[b] = (Gb, newhead - Gb @ self.head[b])
        return M

    def joint(self, bone, M=None, which="head"):
        M = M or self._transforms()
        par = self.rig.bones[bone]["parent"]
        if which == "head":
            A, c = M[par] if par else (np.eye(3), np.zeros(3))
            return A @ self.head[bone] + c
        A, c = M[bone]
        return A @ self.tail[bone] + c

    def limb_ik(self, upper, lower, end_bone, target, pole, extra=()):
        """Rotate upper+lower chain so `end_bone`'s head reaches target."""
        M = self._transforms()
        root = self.joint(upper, M)
        mid = self.joint(lower, M)
        end = self.joint(end_bone, M)
        new_mid, new_end = _two_bone(root, mid, end, target, pole)
        Gu = _rot_between(mid - root, new_mid - root) @ M[upper][0]
        for b in (upper,) + tuple(extra[:1]):
            self.G[b] = Gu
        M = self._transforms()
        mid2 = self.joint(lower, M)
        end2 = self.joint(end_bone, M)
        Gl = _rot_between(end2 - mid2, new_end - mid2) @ M[lower][0]
        for b in (lower,) + tuple(extra[1:]):
            self.G[b] = Gl
        return new_mid, new_end

    def skin(self, V=None):
        M = self._transforms()
        V = self.V if V is None else V
        out = np.zeros_like(V)
        wsum = np.zeros(len(V))
        for b, arr in self.rig.weights.items():
            if b not in M:
                continue
            idx, wt = arr[:, 0].astype(int), arr[:, 1]
            A, c = M[b]
            out[idx] += wt[:, None] * (V[idx] @ A.T + c)
            wsum[idx] += wt
        lone = wsum < 1e-6
        out[~lone] /= wsum[~lone, None]
        out[lone] = V[lone]
        self.M = M
        return out


# --------------------------------------------------------------- materials
def _regions(rig, Vrest, posed_V, who):
    """
    Material per face. Surgeon / radiologist: scrubs, gloves, cap, mask.
    Patient: hospital gown over the trunk, arms and legs; bare face, hands
    and feet; a cap; skin prep (iodine) around the drape window over the
    right breast.
    """
    y, z, x = Vrest[:, 1], Vrest[:, 2], Vrest[:, 0]
    dom = rig.dom
    lab = np.array(["skin"] * len(Vrest), dtype=object)
    hand = np.array([str(b).startswith(("wrist", "finger", "metacarpal",
                                        "thumb")) for b in dom])
    foot = np.array([str(b).startswith(("foot", "toe")) for b in dom])
    cap = y > 7.62 - 0.55 * np.clip((0.9 - z) / 1.4, 0.0, 1.0)
    if who == "surgeon":
        lab[:] = "scrubs"
        lab[y > 6.10] = "skin"
        vneck = (y > 5.45) & (z > 0.55) & (np.abs(x) < 0.55 - (6.1 - y))
        lab[vneck] = "skin"
        lab[hand] = "glove"
        lab[foot] = "shoe"
        mask = (y > 6.50) & (y < 7.13) & (z > 0.90) & (np.abs(x) < 0.80)
        lab[mask] = "mask"
        lab[cap] = "cap"
    else:
        lab[:] = "gown"
        lab[y > 6.02] = "skin"                  # neck and head
        lab[hand | foot] = "skin"
        lab[cap] = "cap"
        P = posed_V
        site = np.asarray(cfg.PROBE_SITE, float)
        prep = (np.hypot(P[:, 0] - site[0], P[:, 1] - site[1]) < 0.095) & \
            (P[:, 2] > cfg.CHEST_Z - 0.06)
        lab[prep] = "prep"
    pri = {"skin": 0, "prep": 1, "scrubs": 1, "shoe": 1, "legging": 1,
           "pants": 1, "gown": 1, "glove": 2, "mask": 3, "cap": 4}
    F = rig.F
    fl = lab[F]
    best = np.argmax(np.vectorize(pri.get)(fl), axis=1)
    out = fl[np.arange(len(F)), best]
    c = Vrest[F].mean(axis=1)
    cy, cz, cx = c[:, 1], c[:, 2], c[:, 0]
    capc = cy > 7.62 - 0.55 * np.clip((0.9 - cz) / 1.4, 0.0, 1.0)
    head_zone = cy > 6.9
    out[head_zone & capc] = "cap"
    out[head_zone & ~capc & (out == "cap")] = "skin"
    if who == "surgeon":
        neck = (cy > 5.0) & (cy < 6.4)
        vn = (cy > 5.45) & (cz > 0.55) & (np.abs(cx) < 0.55 - (6.1 - cy))
        skin_c = (cy > 6.10) | vn
        out[neck & skin_c & (out == "scrubs")] = "skin"
        out[neck & ~skin_c & (out == "skin")] = "scrubs"
    return out


# garments sit slightly proud of the skin (metres, along the normal)
GARMENT_OFFSET = {"cap": 0.006, "mask": 0.005, "glove": 0.0015,
                  "scrubs": 0.003, "legging": 0.012, "pants": 0.010,
                  "gown": 0.006}


MATERIALS = {
    # name: (rgba, roughness, metallic)
    "skin":   ([0.86, 0.66, 0.55, 1.0], 0.55, 0.0),
    "scrubs": ([0.22, 0.44, 0.40, 1.0], 0.88, 0.0),
    "glove":  ([0.86, 0.88, 0.80, 1.0], 0.40, 0.0),
    "cap":    ([0.24, 0.47, 0.52, 1.0], 0.90, 0.0),
    "mask":   ([0.62, 0.78, 0.86, 1.0], 0.92, 0.0),
    "shoe":   ([0.10, 0.11, 0.13, 1.0], 0.50, 0.0),
    "eye":    ([0.92, 0.92, 0.90, 1.0], 0.15, 0.0),
    "iris":   ([0.30, 0.20, 0.12, 1.0], 0.15, 0.0),
    "pupil":  ([0.02, 0.02, 0.02, 1.0], 0.10, 0.0),
    "tape":   ([0.95, 0.95, 0.93, 1.0], 0.70, 0.0),
    "legging": ([0.24, 0.47, 0.60, 1.0], 0.95, 0.0),
    "pants":  ([0.47, 0.61, 0.71, 1.0], 0.93, 0.0),       # hospital cotton
    "gown":   ([0.64, 0.76, 0.84, 1.0], 0.93, 0.0),
    "prep":   ([0.42, 0.20, 0.075, 1.0], 0.45, 0.0),     # dried iodine
}


def _hem(part, radius, sections=8):
    """Rolled hem (thin tube) along the open edges of a garment piece."""
    try:
        outline = part.outline()
    except Exception:                                       # noqa: BLE001
        return None
    tubes = []
    for ent in outline.entities:
        pts = outline.vertices[ent.points]
        if len(pts) < 4:
            continue
        # smooth the saw-tooth edge a little before sweeping
        k = 3
        pad = np.vstack([pts[-k:], pts, pts[:k]]) if ent.closed else \
            np.vstack([np.repeat(pts[:1], k, 0), pts, np.repeat(pts[-1:], k, 0)])
        sm = np.array([pad[i:i + 2 * k + 1].mean(0) for i in range(len(pts))])
        ring = np.array([[np.cos(a), np.sin(a)] for a in
                         np.linspace(0, 2 * np.pi, sections, endpoint=False)])
        verts, faces = [], []
        n = len(sm)
        for i in range(n):
            t = sm[(i + 1) % n] - sm[i - 1]
            t /= max(np.linalg.norm(t), 1e-9)
            a = np.cross(t, [0, 0, 1.0])
            if np.linalg.norm(a) < 1e-6:
                a = np.cross(t, [1.0, 0, 0])
            a /= np.linalg.norm(a)
            b = np.cross(t, a)
            for c, s_ in ring:
                verts.append(sm[i] + radius * (c * a + s_ * b))
        last = n if ent.closed else n - 1
        for i in range(last):
            j = (i + 1) % n
            for r in range(sections):
                r2 = (r + 1) % sections
                p0, p1 = i * sections + r, i * sections + r2
                q0, q1 = j * sections + r, j * sections + r2
                faces += [[p0, q0, p1], [p1, q0, q1]]
        tubes.append(trimesh.Trimesh(np.array(verts), np.array(faces),
                                     process=False))
    return trimesh.util.concatenate(tubes) if tubes else None


HEM = {"cap": 0.0045, "scrubs": 0.0030, "pants": 0.0060, "gown": 0.0045}


def _split(V, F, mats):
    full = trimesh.Trimesh(V, F, process=False)
    N = full.vertex_normals
    parts = []
    for m in np.unique(mats):
        sel = F[mats == m]
        Vm = V + N * GARMENT_OFFSET.get(m, 0.0)
        tm = trimesh.Trimesh(Vm, sel, process=False)
        tm.remove_unreferenced_vertices()
        parts.append((m, tm))
        if m in HEM:
            # only the hem around the head / neck, not cuffs at wrists etc.
            sub = tm
            if m == "scrubs":
                keep = tm.vertices[tm.faces].mean(axis=1)
                hi = np.percentile(tm.vertices[:, 2], 80)
                sub = trimesh.Trimesh(tm.vertices,
                                      tm.faces[keep[:, 2] > hi], process=False)
            h = _hem(sub, HEM[m])
            if h is not None:
                if m == "scrubs":
                    # keep only neckline loops (the highest ones)
                    pass
                parts.append((m, h))
    return parts


# ------------------------------------------------------------------ people
def _place(R, height, pos):
    rig = _Rig.get()
    h_rest = rig.V[:, 1].max() - rig.V[:, 1].min()
    s = height / h_rest
    return R, s, np.asarray(pos, float)


def build_radiologist():
    """Standing on the patient's left at cfg.RADIOLOGIST_POS, facing +y
    (toward the table), hands forward over the table edge."""
    x, y = cfg.RADIOLOGIST_POS
    # MH: +z forward, +y up, +x = character's left
    # world: forward +y, up +z, character's left = -x
    R = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], float)
    rig = _Rig.get()
    R, s, _ = _place(R, cfg.SURGEON_HEIGHT, (0, 0, 0))
    foot_y = rig.V[:, 1].min()
    t = np.array([x, y, -foot_y * s])
    P = Posed(R, s, t)
    lean = _axis_angle([1, 0, 0], np.radians(-6))
    P.rotate("spine03", lean)
    for side, sg in (("L", -1), ("R", 1)):
        hand = np.array([x + sg * 0.16, y + 0.40, 1.02])
        pole = np.array([x + sg * 0.45, y - 0.05, 0.95])
        P.limb_ik(f"upperarm01.{side}", f"lowerarm01.{side}",
                  f"wrist.{side}", hand, pole,
                  extra=(f"upperarm02.{side}", f"lowerarm02.{side}"))
        P.rotate(f"wrist.{side}", P.G[f"lowerarm01.{side}"])
    return _finish(P, "surgeon")


def build_patient():
    """
    Supine on the table, head toward +x, RIGHT arm raised above the head
    (ipsilateral arm up for a right-breast biopsy), left arm at the side,
    legs straight. The body is thinned front-to-back so the chest skin at
    the scan site sits at cfg.CHEST_Z (the breast mound is added on top by
    the renderer from the same geometry the planner uses).
    """
    rig = _Rig.get()
    # MH up (+y) -> world +x (head end), MH front (+z) -> world +z (face up)
    R = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], float)
    R, s, _ = _place(R, cfg.PATIENT_HEIGHT, (0, 0, 0))
    back_z = rig.V[:, 2].min()
    root_y = rig.head["root"][1]
    pad = cfg.TABLE_TOP_Z + cfg.TABLE_PAD
    t = np.array([cfg.PATIENT_PELVIS_X - root_y * s, 0.0, pad - back_z * s])
    probe = Posed(R, s, t)
    V = probe.V
    site = np.asarray(cfg.PROBE_SITE, float)
    near = np.hypot(V[:, 0] - site[0], V[:, 1] - site[1]) < 0.05
    top = V[near, 2].max() if near.any() else V[:, 2].max()
    k = (cfg.CHEST_Z - pad) / max(top - pad, 1e-6)
    P = Posed(R, s, t, zflat=(pad, k))
    z = pad
    for side, sg in (("L", -1), ("R", 1)):
        if side == "R":       # raised above the head, resting on the board
            wrist = np.array([0.98, 0.17, z + 0.07])
            pole = np.array([0.74, 0.34, z + 0.10])
        else:                 # along the left side
            wrist = np.array([0.05, -0.22, z + 0.05])
            pole = np.array([0.30, -0.30, z + 0.06])
        P.limb_ik(f"upperarm01.{side}", f"lowerarm01.{side}",
                  f"wrist.{side}", wrist, pole,
                  extra=(f"upperarm02.{side}", f"lowerarm02.{side}"))
    parts = _finish(P, "patient")
    META["patient_chest_k"] = float(k)
    return parts


def _mask_sheet(Vrest):
    """
    Pleated surgical mask as its own surface (rest-space, dm): a sheet laid
    over the face's front outline from the nose bridge to under the chin,
    with three horizontal pleats. Built separately so it bridges the nose and
    lips like real fabric instead of following them.
    """
    cz = 0.25
    face = Vrest[(Vrest[:, 1] > 6.45) & (Vrest[:, 1] < 7.18) &
                 (Vrest[:, 2] > 0.55) & (np.abs(Vrest[:, 0]) < 0.95)]
    th_f = np.arctan2(face[:, 0], face[:, 2] - cz)
    r_f = np.hypot(face[:, 0], face[:, 2] - cz)
    nth, ny = 40, 22
    ths = np.linspace(-1.05, 1.05, nth)
    env = np.array([r_f[np.abs(th_f - t) < 0.06].max()
                    if np.any(np.abs(th_f - t) < 0.06) else 0.0
                    for t in ths])
    env = np.convolve(np.pad(env, 3, mode="edge"), np.ones(7) / 7, "valid")
    ys = np.linspace(6.50, 7.10, ny)
    G = np.zeros((ny, nth, 3))
    for i, yv in enumerate(ys):
        u = (yv - ys[0]) / (ys[-1] - ys[0])          # 0 chin .. 1 nose bridge
        tuck = 0.10 * (1 - np.sin(np.pi * u))          # hugs at top/bottom
        pleat = 0.025 * np.sin(u * 3 * 2 * np.pi)
        for j, t in enumerate(ths):
            side = abs(t) / 1.05
            r = env[j] + 0.06 - tuck * (0.5 + 0.5 * side) + pleat
            G[i, j] = [r * np.sin(t), yv - 0.10 * side ** 2, cz + r * np.cos(t)]
    V = G.reshape(-1, 3)
    F = []
    for i in range(ny - 1):
        for j in range(nth - 1):
            a0 = i * nth + j
            F += [[a0, a0 + 1, a0 + nth], [a0 + 1, a0 + nth + 1, a0 + nth]]
    return V, np.array(F)


def _finish(P, who):
    rig = P.rig
    V = P.skin()
    mats = _regions(rig, rig.V, V, who)
    if who == "surgeon":
        mats = np.where(mats == "mask", "skin", mats)     # face under it
    parts = _split(V, rig.F, mats)
    if who == "surgeon":
        MV, MF = _mask_sheet(rig.V)
        Mh = P.M["head"]
        sheet = trimesh.Trimesh(P._w(MV) @ Mh[0].T + Mh[1], MF, process=False)
        # two-sided so it reads from any angle
        back = trimesh.Trimesh(sheet.vertices, sheet.faces[:, ::-1],
                               process=False)
        parts.append(("mask", trimesh.util.concatenate([sheet, back])))
    if rig.eyes is not None:
        EV, EF = rig.eyes
        Mh = P.M["head"]
        Ew = P._w(EV) @ Mh[0].T + Mh[1]
        tm = trimesh.Trimesh(Ew, EF, process=False)
        if who == "patient":
            # eyes taped shut for surgery: a tape strip over each eye
            for c in (tm.vertices[tm.vertices[:, 1] > Ew[:, 1].mean()].mean(0),
                      tm.vertices[tm.vertices[:, 1] <= Ew[:, 1].mean()].mean(0)):
                tape = trimesh.creation.box(extents=[0.032, 0.040, 0.004])
                tape.apply_translation(c + [0, 0, 0.010])
                parts.append(("tape", tape))
        else:
            # sclera / iris / pupil by angle from each eyeball's gaze axis
            fwd = Mh[0] @ (P.R0 @ np.array([0.0, 0.0, 1.0]))
            fwd /= np.linalg.norm(fwd)
            lab = np.array(["eye"] * len(tm.faces), dtype=object)
            side = (Ew[:, :] - Ew.mean(0)) @ (Mh[0] @ (P.R0 @ [1.0, 0, 0]))
            for sel in (side > 0, side <= 0):
                c = Ew[sel].mean(axis=0)
                d = Ew - c
                r = np.linalg.norm(d[sel], axis=1).max()
                cosang = (d @ fwd) / np.maximum(np.linalg.norm(d, axis=1),
                                                1e-9)
                fc = cosang[tm.faces].mean(axis=1)
                fs = sel[tm.faces].all(axis=1)
                lab[fs & (fc > 0.86)] = "iris"
                lab[fs & (fc > 0.965)] = "pupil"
            for m in ("eye", "iris", "pupil"):
                sub = trimesh.Trimesh(Ew, tm.faces[lab == m], process=False)
                sub.remove_unreferenced_vertices()
                parts.append((m, sub))
    return parts
