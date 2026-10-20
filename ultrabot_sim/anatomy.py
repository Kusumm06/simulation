"""
Parametric breast + chest-wall phantom registered to the ultrasound plane.

My YOLO model only detects LESIONS. There is no segmentation of the other
tissues, so everything in this file is an explicit, configurable MODEL
(config.py), drawn and labelled as a model - never as detected anatomy.

Layers, from the skin (top of the image) down, all in the probe frame (mm):

    skin                 0 .. 2 mm below the skin surface
    subcutaneous fat     2 .. 8 mm                     low risk
    fibroglandular       8 mm .. pectoral fascia       where lesions live
    Cooper's ligaments   thin oblique bands            soft penalty
    pectoral fascia      PECTORAL_DEPTH_MM (pushed >= 6 mm below the lesion)
                                                       HARD no-go
    pectoralis major     MUSCLE_THICK_MM below the fascia
    ribs / pleura        below the muscle              HARD no-go
    superficial vessels  tubes through the plane       HARD no-go + margin
    nipple-areola        region beside the probe       entry forbidden

Every structure is available as a signed-distance function (fast clearance
maths, positive = outside / above) and as a trimesh mesh (rendering and
PyBullet collision).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ultrabot_sim import config as cfg
from ultrabot_sim.us_geometry import USCalib


@dataclass
class LesionGeom:
    """Lesion ellipse in the probe frame (mm). The elevational half-size is
    ASSUMED equal to the lateral one (single-plane ultrasound)."""
    x: float
    z: float
    a: float          # lateral half-axis
    b: float          # axial half-axis

    @property
    def c(self):
        return self.a

    @classmethod
    def from_box_px(cls, box, calib: USCalib):
        x1, y1, x2, y2 = [float(v) for v in box]
        cx, cz = calib.px_to_mm((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        a = abs(x2 - x1) / 2.0 * calib.mm_per_px_lateral
        b = abs(y2 - y1) / 2.0 * calib.mm_per_px_axial
        return cls(float(cx), float(cz), float(max(a, 0.5)),
                   float(max(b, 0.5)))

    def contains(self, x, z, scale=1.0):
        return ((np.asarray(x) - self.x) / (self.a * scale)) ** 2 + \
            ((np.asarray(z) - self.z) / (self.b * scale)) ** 2 <= 1.0

    def boundary_distance(self, x, z):
        """Approximate distance (mm) from a point to the ellipse boundary,
        positive outside (radial scaling approximation, exact for circles)."""
        dx, dz = np.asarray(x) - self.x, np.asarray(z) - self.z
        r = np.hypot(dx, dz)
        k = np.sqrt((dx / self.a) ** 2 + (dz / self.b) ** 2)
        with np.errstate(invalid="ignore", divide="ignore"):
            rb = np.where(k > 1e-9, r / k, min(self.a, self.b))
        return r - rb


@dataclass
class Vessel:
    x: float
    z: float
    r: float
    moved_mm: float = 0.0


@dataclass
class ChestwallEstimate:
    used: bool
    confidence: float
    z0: float
    slope: float
    message: str


HARD_MARGINS = {}      # filled in Anatomy.__init__ from config


def _skin_profile(d):
    H, D = cfg.BREAST_HEIGHT_MM, cfg.BREAST_FALLOFF_MM
    d = np.clip(np.asarray(d, float), 0.0, None)
    return H * (1.0 - np.cos(np.minimum(d / D, 1.0) * np.pi / 2.0))


def _skin_profile_slope(d):
    H, D = cfg.BREAST_HEIGHT_MM, cfg.BREAST_FALLOFF_MM
    d = np.clip(np.asarray(d, float), 0.0, None)
    inside = d < D
    return np.where(inside, H * np.sin(np.minimum(d / D, 1.0) * np.pi / 2.0)
                    * (np.pi / 2.0) / D, 0.0)


class Anatomy:
    def __init__(self, calib: USCalib, lesion: LesionGeom | None = None,
                 frame_bgr=None, auto_chestwall=None):
        self.calib = calib
        self.lesion = lesion
        self.events: list[str] = []
        self.margins = {"vessel": cfg.VESSEL_MARGIN_MM,
                        "chest wall": cfg.CHESTWALL_CLEARANCE_MM,
                        "rib": cfg.RIB_MARGIN_MM,
                        "pleura": cfg.PLEURA_MARGIN_MM}
        HARD_MARGINS.update(self.margins)
        # ---- pectoral fascia (a line z = z0 + slope * x in the probe plane)
        lesion_bottom = (lesion.z + lesion.b) if lesion else 0.0
        floor = lesion_bottom + cfg.PECTORAL_MIN_BELOW_LESION_MM
        self.fascia_z0 = max(cfg.PECTORAL_DEPTH_MM, floor)
        self.fascia_slope = 0.0
        if self.fascia_z0 > cfg.PECTORAL_DEPTH_MM + 1e-9:
            self.events.append(
                f"pectoral fascia pushed to {self.fascia_z0:.1f} mm "
                f"(>= {cfg.PECTORAL_MIN_BELOW_LESION_MM:.0f} mm below the "
                f"lesion edge)")
        use_auto = cfg.AUTO_CHESTWALL if auto_chestwall is None \
            else auto_chestwall
        self.chestwall = ChestwallEstimate(False, 0.0, self.fascia_z0, 0.0,
                                           "configured depth")
        if use_auto and frame_bgr is not None:
            self.chestwall = estimate_chestwall(frame_bgr, calib)
            if self.chestwall.used:
                z0, k = self.chestwall.z0, self.chestwall.slope
                # never above the lesion: shift down (keep the slope)
                hw = calib.half_width_mm
                zmin = min(z0 + k * (-hw), z0 + k * hw)
                if zmin < floor:
                    z0 += floor - zmin
                    self.events.append(
                        "auto chest wall shifted below the lesion "
                        f"(+{floor - zmin:.1f} mm)")
                self.fascia_z0, self.fascia_slope = z0, k
                self.events.append(
                    f"auto chest wall used (confidence "
                    f"{self.chestwall.confidence:.2f})")
            else:
                self.events.append(f"auto chest wall not used: "
                                   f"{self.chestwall.message}; configured "
                                   f"depth kept")
        # ---- vessels, moved off the lesion if they overlap it
        self.vessels = []
        for (x, z, r) in cfg.VESSELS:
            v = Vessel(float(x), float(z), float(r))
            if lesion is not None:
                self._clear_vessel_of_lesion(v, lesion)
            # keep it in the fat / gland layers, above the fascia
            zmax = self.fascia_depth(v.x) - v.r - 1.0
            if v.z > zmax:
                v.z = zmax
            self.vessels.append(v)
        # ---- Cooper's ligaments (segments from the fascia to the skin)
        self.ligaments = []
        for xf, xs in cfg.LIGAMENTS:
            zf = float(self.fascia_depth(xf))
            zs = float(self.skin_depth(xs) + cfg.SKIN_THICK_MM)
            self.ligaments.append(((float(xf), zf), (float(xs), zs)))
        # ---- ribs (circles below the muscle, centres along x)
        self.ribs = []
        for k in range(-8, 9):
            x = cfg.RIB_OFFSET_MM + k * cfg.RIB_PITCH_MM
            if abs(x) > cfg.BREAST_FALLOFF_MM + 40:
                continue
            z = float(self.fascia_depth(x) + cfg.MUSCLE_THICK_MM +
                      cfg.RIB_RADIUS_MM)
            self.ribs.append((float(x), z, float(cfg.RIB_RADIUS_MM)))

    # ------------------------------------------------------------ layers
    def skin_depth(self, x):
        """Depth (mm, probe +z) of the skin surface at lateral x. Zero under
        the transducer (the probe flattens the skin), falling away
        outside it."""
        d = np.abs(np.asarray(x, float)) - self.calib.half_width_mm
        return _skin_profile(d)

    def skin_slope(self, x):
        x = np.asarray(x, float)
        d = np.abs(x) - self.calib.half_width_mm
        return np.sign(x) * _skin_profile_slope(d)

    def skin_inward_normal(self, x):
        """Unit normal of the skin curve at x, pointing into the body."""
        s = float(self.skin_slope(x))
        n = np.array([-s, 1.0])
        return n / np.linalg.norm(n)

    def fat_bottom(self, x):
        return self.skin_depth(x) + cfg.SKIN_THICK_MM + cfg.FAT_THICK_MM

    def fascia_depth(self, x):
        return self.fascia_z0 + self.fascia_slope * np.asarray(x, float)

    def muscle_bottom(self, x):
        return self.fascia_depth(x) + cfg.MUSCLE_THICK_MM

    def pleura_depth(self, x):
        return self.muscle_bottom(x) + 2 * cfg.RIB_RADIUS_MM + \
            cfg.PLEURA_GAP_MM

    def fascia_dir(self):
        d = np.array([1.0, self.fascia_slope])
        return d / np.linalg.norm(d)

    def in_nipple(self, x):
        return np.abs(np.asarray(x, float) - cfg.NIPPLE_X_MM) < cfg.NIPPLE_R_MM

    # ---------------------------------------------------------- distances
    def clearance(self, P):
        """
        Clearance (mm) of points P (N, 2 = x, z in the plane, or N, 3 =
        x, y, z) to every hard no-go structure. Vessels and ribs are tubes
        running along y, so the in-plane distance is exact for them;
        fascia and pleura are planes containing y.
        Returns dict family -> (N,) clearance (positive = outside).
        """
        P = np.atleast_2d(np.asarray(P, float))
        if P.shape[1] == 3:
            x, z = P[:, 0], P[:, 2]
        else:
            x, z = P[:, 0], P[:, 1]
        out = {}
        # chest wall: perpendicular distance above the fascia line
        k = self.fascia_slope
        out["chest wall"] = (self.fascia_z0 + k * x - z) / math.sqrt(1 + k * k)
        if self.vessels:
            dv = np.min([np.hypot(x - v.x, z - v.z) - v.r
                         for v in self.vessels], axis=0)
            out["vessel"] = dv
        if self.ribs:
            out["rib"] = np.min([np.hypot(x - rx, z - rz) - rr
                                 for rx, rz, rr in self.ribs], axis=0)
        zp0 = self.pleura_depth(0.0)
        out["pleura"] = (zp0 + k * x - z) / math.sqrt(1 + k * k)
        return out

    def min_clearance(self, P):
        """(min clearance over all hard structures, family name of the
        closest) per point, plus the margin-normalised slack (clearance -
        margin; negative = violation)."""
        c = self.clearance(P)
        fams = list(c.keys())
        M = np.vstack([c[f] for f in fams])
        slack = np.vstack([c[f] - self.margins[f] for f in fams])
        i = np.argmin(M, axis=0)
        return M[i, np.arange(M.shape[1])], np.array(fams)[i], \
            slack.min(axis=0)

    def ligament_crossings(self, E, F):
        """Number of Cooper's ligaments the segment E -> F crosses."""
        n = 0
        for (a, b) in self.ligaments:
            if _segments_intersect(E, F, a, b):
                n += 1
        return n

    def vessel_at(self, i):
        return self.vessels[i]

    # ------------------------------------------------------------- helpers
    def _clear_vessel_of_lesion(self, v: Vessel, lesion: LesionGeom):
        need = v.r + cfg.VESSEL_MARGIN_MM
        d = float(lesion.boundary_distance(v.x, v.z))
        if d >= need:
            return
        sgn = 1.0 if v.x >= lesion.x else -1.0
        x0 = v.x
        for _ in range(400):
            v.x += sgn * 0.5
            if float(lesion.boundary_distance(v.x, v.z)) >= need:
                break
        v.moved_mm = v.x - x0
        self.events.append(
            f"vessel at x={x0:.0f} mm overlapped the lesion; moved "
            f"{v.moved_mm:+.1f} mm (model vessel)")

    def to_dict(self):
        return {
            "fascia_z0": self.fascia_z0, "fascia_slope": self.fascia_slope,
            "vessels": [(v.x, v.z, v.r) for v in self.vessels],
            "ribs": self.ribs, "ligaments": self.ligaments,
            "lesion": None if self.lesion is None else
            (self.lesion.x, self.lesion.z, self.lesion.a, self.lesion.b),
            "chestwall": self.chestwall.message,
            "events": list(self.events),
            "depth_mm": self.calib.depth_mm, "width_mm": self.calib.width_mm,
        }

    # -------------------------------------------------------------- meshes
    def meshes(self):
        """Meshes in the PROBE frame, millimetres:
        [(name, trimesh, material_key)]. Callers convert to the world."""
        return build_meshes(self)


def _segments_intersect(p1, p2, q1, q2):
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    p1, p2, q1, q2 = map(lambda t: (float(t[0]), float(t[1])),
                         (p1, p2, q1, q2))
    o1, o2 = orient(p1, p2, q1), orient(p1, p2, q2)
    o3, o4 = orient(q1, q2, p1), orient(q1, q2, p2)
    return (o1 * o2 < 0) and (o3 * o4 < 0)


# ---------------------------------------------------------------- chest wall
def estimate_chestwall(frame_bgr, calib: USCalib, rng_seed=0):
    """
    Pectoral fascia from the image: a bright, roughly horizontal echogenic
    band in the lower part of the frame. For each of 33 column strips the
    row of strongest dark->bright transition in the lower 45 % is taken,
    then a RANSAC line is fitted. Confidence = inlier share x contrast.
    """
    import cv2
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 3.0)
    H, W = g.shape
    v0, v1 = int(0.55 * H), int(0.97 * H)
    grad = np.zeros_like(g)
    grad[1:-1] = g[2:] - g[:-2]                    # brighter below = +
    pts = []
    strengths = []
    for k in range(33):
        u0, u1 = int(k * W / 33), int((k + 1) * W / 33)
        col = grad[v0:v1, u0:u1].mean(axis=1)
        j = int(np.argmax(col))
        pts.append(((u0 + u1) / 2.0, v0 + j))
        strengths.append(col[j])
    pts = np.array(pts, float)
    strengths = np.array(strengths, float)
    rng = np.random.default_rng(rng_seed)
    best_in, best = None, None
    for _ in range(200):
        i, j = rng.choice(len(pts), 2, replace=False)
        (ua, va), (ub, vb) = pts[i], pts[j]
        if abs(ub - ua) < 60:
            continue
        m = (vb - va) / (ub - ua)
        c = va - m * ua
        res = np.abs(pts[:, 1] - (m * pts[:, 0] + c))
        inl = res < 10.0
        if best_in is None or inl.sum() > best_in.sum():
            best_in, best = inl, (m, c)
    if best is None or best_in.sum() < 4:
        return ChestwallEstimate(False, 0.0, cfg.PECTORAL_DEPTH_MM, 0.0,
                                 "no consistent bright band")
    # least squares on the inliers
    A = np.column_stack([pts[best_in, 0], np.ones(best_in.sum())])
    m, c = np.linalg.lstsq(A, pts[best_in, 1], rcond=None)[0]
    inlier_share = best_in.mean()
    contrast = float(np.clip(np.median(strengths[best_in]) / 12.0, 0, 1))
    conf = float(inlier_share * contrast)
    # to the probe frame: z = z0 + slope * x
    x_a, z_a = calib.px_to_mm(0.0, c)
    x_b, z_b = calib.px_to_mm(float(W), m * W + c)
    slope = float((z_b - z_a) / (x_b - x_a))
    z0 = float(z_a - slope * x_a)
    if abs(math.degrees(math.atan(slope))) > 20:
        return ChestwallEstimate(False, conf, z0, slope,
                                 "estimated line too steep (> 20 deg)")
    if conf < cfg.CHESTWALL_MIN_CONFIDENCE:
        return ChestwallEstimate(False, conf, z0, slope,
                                 f"low confidence {conf:.2f}")
    return ChestwallEstimate(True, conf, z0, slope, "image estimate")


# ------------------------------------------------------------------- meshes
def build_meshes(an: Anatomy):
    """Probe-frame (mm) meshes of the phantom for the 3D view."""
    import trimesh
    out = []
    W2 = an.calib.half_width_mm
    D = cfg.BREAST_FALLOFF_MM
    out.append(("breast-skin", breast_surface_mesh(W2), "skin"))
    span = W2 + D
    # fascia slab and muscle (thin boxes following the fascia line)
    for name, top, thick, mat in (
            ("fascia", 0.0, 1.2, "fascia"),
            ("muscle", 1.2, cfg.MUSCLE_THICK_MM - 1.2, "muscle")):
        L = 2 * min(span, 95.0)
        b = trimesh.creation.box(extents=[L, 70.0, thick])
        ang = math.atan(an.fascia_slope)
        T = trimesh.transformations.rotation_matrix(-ang, [0, 1, 0])
        b.apply_transform(T)
        b.apply_translation([0, 0, an.fascia_z0 + top + thick / 2])
        out.append((name, b, mat))
    for (x, z, r) in an.ribs:
        if abs(x) > 95:
            continue
        c = trimesh.creation.cylinder(radius=r, height=70.0, sections=24)
        c.apply_transform(trimesh.transformations.rotation_matrix(
            np.pi / 2, [1, 0, 0]))
        c.apply_translation([x, 0, z])
        out.append(("rib", c, "bone"))
    for v in an.vessels:
        c = trimesh.creation.cylinder(radius=v.r, height=70.0, sections=20)
        c.apply_transform(trimesh.transformations.rotation_matrix(
            np.pi / 2, [1, 0, 0]))
        c.apply_translation([v.x, 0, v.z])
        out.append(("vessel", c, "vessel"))
    if an.lesion is not None:
        L = an.lesion
        s = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        s.apply_scale([L.a, L.c, L.b])
        s.apply_translation([L.x, 0, L.z])
        out.append(("lesion", s, "lesion"))
    for (a, b) in an.ligaments:
        p0 = np.array([a[0], 0, a[1]])
        p1 = np.array([b[0], 0, b[1]])
        d = p1 - p0
        Lg = np.linalg.norm(d)
        q = trimesh.creation.box(extents=[0.8, 60.0, Lg])
        Rm = trimesh.geometry.align_vectors([0, 0, 1], d / Lg)
        q.apply_transform(Rm)
        q.apply_translation((p0 + p1) / 2)
        out.append(("ligament", q, "ligament"))
    return out


def breast_surface_mesh(half_width_mm, n=56):
    """Breast skin as a closed convex solid in the probe frame (mm): the
    top surface follows the skin profile used by the planner (flat under
    the transducer), the base is closed at the chest (depth H)."""
    import trimesh
    D = cfg.BREAST_FALLOFF_MM
    H = cfg.BREAST_HEIGHT_MM
    yhalf = 8.0
    R = D + 2.0
    xs = np.linspace(-(half_width_mm + R), half_width_mm + R, n)
    ys = np.linspace(-(yhalf + R), yhalf + R, n)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    dx = np.clip(np.abs(X) - half_width_mm, 0, None)
    dy = np.clip(np.abs(Y) - yhalf, 0, None)
    d = np.hypot(dx, dy)
    Z = _skin_profile(d)
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    pts = pts[d.ravel() <= D + 1e-6]
    base = pts.copy()
    base[:, 2] = H
    hull = trimesh.convex.convex_hull(np.vstack([pts, base]))
    return hull
