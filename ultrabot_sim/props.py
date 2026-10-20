"""
Procedure-room props for the realistic OR panel, built procedurally
(CC0). Adapted from the reference project's props.py (room, textures,
lights, trolley, IV pole, robot carts and monitor tower are reused; the
table, drape and ultrasound machine are new for breast biopsy):

    room        walls, vinyl floor, ceiling panels, doors
    table       flat procedure table + arm board for the raised right arm
    drapes      sterile drape moulded over the patient with a window over
                the right breast
    lights      two ceiling-mounted procedure lamps aimed at the breast
    us machine  ultrasound scanner cart (live image on its screen)
    trolley     biopsy back-table with a specimen pot
    iv          IV pole with a bag
    carts       white robot carts under both arms
    screens     monitor tower (raw ultrasound + AI overlay) and the scanner

Everything is placed from the SAME numbers the simulation uses (config.py),
so the realistic objects sit exactly where the primitive ones were.
Each item is (name, trimesh, material dict). Materials:
    {"color": rgba, "rough": r, "metal": m, "texture": key, "emissive": rgb,
     "uv_scale": (u, v)}
"""
import numpy as np
import trimesh
from trimesh.creation import box, cylinder, capsule, icosphere

from ultrabot_sim import config as cfg

M_STEEL = {"color": [0.80, 0.81, 0.83, 1.0], "rough": 0.28, "metal": 0.95}
M_STEEL_DARK = {"color": [0.42, 0.43, 0.46, 1.0], "rough": 0.35, "metal": 0.9}
M_WHITE_PLASTIC = {"color": [0.93, 0.94, 0.94, 1.0], "rough": 0.30,
                   "metal": 0.0}
M_GREY_PLASTIC = {"color": [0.55, 0.57, 0.60, 1.0], "rough": 0.45,
                  "metal": 0.0}
M_BLACK = {"color": [0.06, 0.07, 0.08, 1.0], "rough": 0.40, "metal": 0.0}
M_PAD = {"color": [0.10, 0.12, 0.16, 1.0], "rough": 0.65, "metal": 0.0}
M_DRAPE = {"color": [0.24, 0.47, 0.60, 1.0], "rough": 0.95, "metal": 0.0,
           "texture": "fabric", "uv_scale": (6, 6)}
M_DRAPE_GREEN = {"color": [0.26, 0.50, 0.44, 1.0], "rough": 0.95,
                 "metal": 0.0, "texture": "fabric", "uv_scale": (6, 6)}
M_INCISE = {"color": [0.85, 0.66, 0.56, 0.35], "rough": 0.15, "metal": 0.0}
M_LAMP_GLOW = {"color": [1.0, 1.0, 0.97, 1.0], "rough": 0.2, "metal": 0.0,
               "emissive": [1.0, 0.98, 0.92]}
M_SCREEN_GLOW = {"color": [0.10, 0.25, 0.30, 1.0], "rough": 0.1,
                 "metal": 0.0, "emissive": [0.25, 0.45, 0.5]}
# room surfaces get more fake bounce light (bounce_k): large light-coloured
# surfaces are lit mostly by light reflected from everything else
M_FLOOR = {"color": [1, 1, 1, 1], "rough": 0.32, "metal": 0.0,
           "texture": "floor", "bounce_k": 1.6}
M_FLOOR_BORDER = {"color": [1, 1, 1, 1], "rough": 0.32, "metal": 0.0,
                  "texture": "border", "bounce_k": 1.4}
M_RAIL = {"color": [0.78, 0.80, 0.81, 1.0], "rough": 0.35, "metal": 0.6}
M_GLASS = {"color": [0.70, 0.80, 0.82, 0.28], "rough": 0.05, "metal": 0.0}
M_LED_PANEL = {"color": [1, 1, 1, 1], "rough": 0.3, "metal": 0.0,
               "emissive": [0.95, 0.96, 0.98]}
M_WALL = {"color": [1, 1, 1, 1], "rough": 0.35, "metal": 0.0,
          "texture": "tiles", "bounce_k": 1.7}
M_CEIL = {"color": [0.95, 0.96, 0.96, 1.0], "rough": 0.8, "metal": 0.0,
          "texture": "ceiling", "bounce_k": 2.0}
M_BAG = {"color": [0.85, 0.90, 0.95, 0.45], "rough": 0.1, "metal": 0.0}
M_GAS_O2 = {"color": [0.92, 0.93, 0.93, 1.0], "rough": 0.3, "metal": 0.2}
M_GAS_N2O = {"color": [0.15, 0.25, 0.65, 1.0], "rough": 0.3, "metal": 0.2}
M_HOSE = {"color": [0.55, 0.70, 0.85, 0.85], "rough": 0.3, "metal": 0.0}
M_SLEEVE = {"color": [0.92, 0.95, 0.98, 0.22], "rough": 0.12, "metal": 0.0}


# ------------------------------------------------------------------ helpers
def _at(m, pos, R=None):
    m = m.copy()
    if R is not None:
        T = np.eye(4)
        T[:3, :3] = R
        m.apply_transform(T)
    m.apply_translation(pos)
    return m


def _rbox(ext, pos, radius=0.02, R=None):
    """Box with rounded vertical edges (looks manufactured, not CG)."""
    ex, ey, ez = ext
    r = min(radius, ex / 2 - 1e-3, ey / 2 - 1e-3)
    core = [box(extents=[ex - 2 * r, ey, ez]), box(extents=[ex, ey - 2 * r, ez])]
    for sx in (-1, 1):
        for sy in (-1, 1):
            c = cylinder(radius=r, height=ez, sections=16)
            c.apply_translation([sx * (ex / 2 - r), sy * (ey / 2 - r), 0])
            core.append(c)
    return _at(trimesh.util.concatenate(core), pos, R)


def _tube(points, radius, sections=12, flat=1.0, cap=False):
    """Swept tube along a polyline. `radius` may be one value or one per
    point; `flat` < 1 squashes the cross-section vertically (oval);
    `cap` closes both ends."""
    pts = np.asarray(points, float)
    rad = np.broadcast_to(np.asarray(radius, float), (len(pts),))
    ring = np.array([[np.cos(a), np.sin(a)] for a in
                     np.linspace(0, 2 * np.pi, sections, endpoint=False)])
    V, F = [], []
    n = len(pts)
    for i in range(n):
        t = pts[min(i + 1, n - 1)] - pts[max(i - 1, 0)]
        t /= max(np.linalg.norm(t), 1e-9)
        a = np.cross(t, [0, 0, 1.0])
        if np.linalg.norm(a) < 1e-6:
            a = np.cross(t, [1.0, 0, 0])
        a /= np.linalg.norm(a)
        b = np.cross(t, a)
        if flat != 1.0 and abs(t[2]) < 0.99:
            b = np.array([0, 0, 1.0]) - t[2] * t      # `b` = most vertical
            b /= np.linalg.norm(b)
            a = np.cross(b, t)
        for c, s in ring:
            V.append(pts[i] + rad[i] * (c * a + flat * s * b))
    for i in range(n - 1):
        for r in range(sections):
            r2 = (r + 1) % sections
            F += [[i * sections + r, (i + 1) * sections + r, i * sections + r2],
                  [i * sections + r2, (i + 1) * sections + r,
                   (i + 1) * sections + r2]]
    if cap:
        for i, sgn in ((0, 1), (n - 1, -1)):
            V.append(pts[i])
            c = len(V) - 1
            for r in range(sections):
                r2 = (r + 1) % sections
                tri = [c, i * sections + r2, i * sections + r]
                F.append(tri if sgn > 0 else tri[::-1])
    return trimesh.Trimesh(np.array(V), np.array(F), process=False)


def _bezier(p0, p1, p2, p3, n=24):
    t = np.linspace(0, 1, n)[:, None]
    return ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 +
            3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3)


def _uv_planar(m, axes=(0, 1), scale=1.0):
    uv = m.vertices[:, list(axes)] * scale
    m.visual = trimesh.visual.TextureVisuals(uv=uv)
    return m


# ------------------------------------------------------------------- room
def room():
    """
    Modern OR shell: seamless wall panels with a stainless crash rail,
    coved vinyl skirting, sheet-vinyl floor with a darker border band,
    600 mm ceiling grid with LED panels and the laminar-flow canopy, plus
    the usual wall equipment (doors, gas outlets, clock, PACS screen,
    glass-front storage cabinets, control panel).
    """
    hx, hy, hz = cfg.ROOM_SIZE
    H = 2.0 * hz
    out = []
    floor = box(extents=[2 * hx, 2 * hy, 0.02])
    floor.apply_translation([0, 0, -0.01])
    out.append(("floor", _uv_planar(floor, (0, 1), 1 / 1.2), M_FLOOR))
    # darker border band along the walls (welded-in vinyl strip)
    bw = 0.30
    for ext, pos in (([2 * hx, bw, 0.004], [0, hy - bw / 2, 0.002]),
                     ([2 * hx, bw, 0.004], [0, -hy + bw / 2, 0.002]),
                     ([bw, 2 * hy - 2 * bw, 0.004], [hx - bw / 2, 0, 0.002]),
                     ([bw, 2 * hy - 2 * bw, 0.004], [-hx + bw / 2, 0, 0.002])):
        out.append(("floor-border", _uv_planar(_at(box(extents=ext), pos),
                                               (0, 1), 1 / 1.2),
                    M_FLOOR_BORDER))
    ceil = box(extents=[2 * hx, 2 * hy, 0.02])
    ceil.apply_translation([0, 0, H + 0.01])
    out.append(("ceiling", _uv_planar(ceil, (0, 1), 1 / 1.2), M_CEIL))
    for name, ext, pos, axes in [
            ("wall-x", [0.02, 2 * hy, H], [-hx, 0, H / 2], (1, 2)),
            ("wall+x", [0.02, 2 * hy, H], [hx, 0, H / 2], (1, 2)),
            ("wall-y", [2 * hx, 0.02, H], [0, -hy, H / 2], (0, 2)),
            ("wall+y", [2 * hx, 0.02, H], [0, hy, H / 2], (0, 2))]:
        w = box(extents=ext)
        w.apply_translation(pos)
        out.append((name, _uv_planar(w, axes, 1 / 2.4), M_WALL))
    # coved skirting (vinyl turned up the wall) + stainless crash rail
    for axis, sgn in ((0, -1), (0, 1), (1, -1), (1, 1)):
        L = 2 * (hy if axis == 0 else hx)
        R = np.eye(3) if axis == 1 else np.array([[0, -1, 0], [1, 0, 0],
                                                  [0, 0, 1]], float)
        wall = (hx if axis == 0 else hy) * sgn
        inward = -sgn
        # cove: quarter round + upstand
        cove = cylinder(radius=0.05, height=L, sections=24)
        cove.apply_transform(trimesh.transformations.rotation_matrix(
            np.pi / 2, [0, 1, 0]))
        up = box(extents=[L, 0.012, 0.11])
        up.apply_translation([0, 0, 0.055])
        for m_, off, z in ((cove, 0.0, 0.0), (up, 0.006, 0.0)):
            pos = np.zeros(3)
            if axis == 0:
                pos[0] = wall + inward * off
            else:
                pos[1] = wall + inward * off
            out.append(("cove", _at(m_, pos, R),
                        {"color": [0.34, 0.39, 0.41, 1], "rough": 0.32,
                         "metal": 0.0, "bounce_k": 1.4}))
        rail = _rbox([L - 0.2, 0.03, 0.13], [0, 0, 0], 0.012)
        stripe = box(extents=[L - 0.2, 0.034, 0.025])
        pos = np.array([0.0, 0.0, 0.92])
        if axis == 0:
            pos[0] = wall + inward * 0.03
        else:
            pos[1] = wall + inward * 0.03
        out.append(("rail", _at(rail, pos, R), M_RAIL))
        out.append(("rail-stripe", _at(stripe, pos, R), M_STEEL_DARK))
    # ceiling LED panels around the laminar-flow canopy
    for x in (-2.1, -1.5, 1.5, 2.1):
        for y in (-1.8, 0.0, 1.8):
            out.append(("led-panel", _at(box(extents=[0.58, 0.58, 0.02]),
                                         [x, y, H - 0.012]), M_LED_PANEL))
    out.append(("laminar", _at(box(extents=[2.4, 2.0, 0.06]), [0, 0, H - 0.03]),
                {"color": [0.97, 0.98, 0.98, 1], "rough": 0.6, "metal": 0}))
    out.append(("laminar-glow", _at(box(extents=[2.2, 1.8, 0.01]),
                                    [0, 0, H - 0.065]),
                {"color": [1, 1, 1, 1], "rough": 0.4, "metal": 0,
                 "emissive": [0.80, 0.82, 0.84]}))
    out += wall_equipment()
    return out


def _on_wall(wall, along, z, ext, depth):
    """Box on a wall: wall in '+x','-x','+y','-y'; along = coordinate
    along the wall; ext = (width, height)."""
    hx, hy, _ = cfg.ROOM_SIZE
    w, h = ext
    if wall in ("+x", "-x"):
        s = 1 if wall == "+x" else -1
        return box(extents=[depth, w, h]), [s * (hx - depth / 2), along, z]
    s = 1 if wall == "+y" else -1
    return box(extents=[w, depth, h]), [along, s * (hy - depth / 2), z]


def wall_equipment():
    out = []
    hx, hy, _ = cfg.ROOM_SIZE
    # sliding hermetic doors with round windows
    for wall, along in (("+y", 1.4), ("-x", -1.6)):
        m, pos = _on_wall(wall, along, 1.10, (1.5, 2.2), 0.05)
        out.append(("door", _at(m, pos), M_RAIL))
        m, pos = _on_wall(wall, along, 1.10, (1.42, 2.12), 0.06)
        out.append(("door-leaf", _at(m, pos),
                    {"color": [0.86, 0.88, 0.88, 1], "rough": 0.3,
                     "metal": 0.3}))
        m, pos = _on_wall(wall, along, 1.55, (0.36, 0.36), 0.07)
        out.append(("door-window", _at(m, pos), M_GLASS))
        m, pos = _on_wall(wall, along + 0.55, 1.05, (0.05, 0.40), 0.09)
        out.append(("door-handle", _at(m, pos), M_STEEL))
    # medical gas outlet panel (O2 white, air black/white, vacuum yellow,
    # N2O blue) at the head end
    m, pos = _on_wall("+x", 1.05, 1.45, (0.80, 0.30), 0.05)
    out.append(("gas-panel", _at(m, pos), M_WHITE_PLASTIC))
    for i, col in enumerate(([0.95, 0.95, 0.95], [0.10, 0.10, 0.10],
                             [0.95, 0.80, 0.10], [0.15, 0.30, 0.75],
                             [0.20, 0.65, 0.30])):
        m, pos = _on_wall("+x", 0.75 + i * 0.15, 1.45, (0.07, 0.07), 0.07)
        out.append(("gas-outlet", _at(m, pos),
                    {"color": col + [1], "rough": 0.3, "metal": 0.1}))
    # wall clock above the gas panel
    clock = cylinder(radius=0.17, height=0.04, sections=48)
    clock.apply_transform(trimesh.transformations.rotation_matrix(
        np.pi / 2, [0, 1, 0]))
    out.append(("clock", _at(clock, [hx - 0.03, 0.0, 2.45]), M_WHITE_PLASTIC))
    face = cylinder(radius=0.15, height=0.01, sections=48)
    face.apply_transform(trimesh.transformations.rotation_matrix(
        np.pi / 2, [0, 1, 0]))
    out.append(("clock-face", _at(face, [hx - 0.052, 0.0, 2.45]),
                {"color": [0.97, 0.97, 0.95, 1], "rough": 0.5, "metal": 0}))
    for ang, L in ((0.0, 0.11), (2.1, 0.08)):           # hands
        hand = box(extents=[0.006, 0.008, L])
        hand.apply_translation([0, 0, L / 2])
        hand.apply_transform(trimesh.transformations.rotation_matrix(
            ang, [1, 0, 0]))
        out.append(("clock-hand", _at(hand, [hx - 0.06, 0.0, 2.45]),
                    M_BLACK))
    # PACS / imaging wall screen at the head end
    m, pos = _on_wall("+x", -1.35, 1.65, (1.25, 0.74), 0.05)
    out.append(("pacs-frame", _at(m, pos), M_BLACK))
    m, pos = _on_wall("+x", -1.35, 1.65, (1.17, 0.66), 0.055)
    out.append(("pacs-screen", _at(m, pos),
                {"color": [0.05, 0.06, 0.07, 1], "rough": 0.1, "metal": 0,
                 "emissive": [0.10, 0.13, 0.16]}))
    for i, (dx, dz, w_, h_) in enumerate(((-0.30, 0.05, 0.42, 0.50),
                                          (0.25, 0.05, 0.50, 0.50))):
        m, pos = _on_wall("+x", -1.35 + dx, 1.65 + dz, (w_, h_), 0.058)
        out.append(("pacs-image", _at(m, pos),
                    {"color": [0.1, 0.1, 0.1, 1], "rough": 0.2, "metal": 0,
                     "emissive": [0.42, 0.44, 0.46] if i == 0
                     else [0.30, 0.33, 0.35]}))
    # glass-front storage cabinets recessed into the long wall
    for x0 in (-1.9, -0.8, 0.3):
        m, pos = _on_wall("-y", x0, 1.25, (1.0, 1.70), 0.06)
        out.append(("cabinet-frame", _at(m, pos), M_RAIL))
        m, pos = _on_wall("-y", x0, 1.25, (0.92, 1.62), 0.065)
        out.append(("cabinet-back", _at(m, pos),
                    {"color": [0.82, 0.85, 0.85, 1], "rough": 0.4,
                     "metal": 0.2}))
        for z in (0.75, 1.15, 1.55, 1.95):
            m, pos = _on_wall("-y", x0, z, (0.90, 0.015), 0.09)
            out.append(("cabinet-shelf", _at(m, pos), M_STEEL))
            # boxes / packs on the shelves
            for k in range(4):
                bx = x0 - 0.33 + k * 0.22
                col = [[0.25, 0.45, 0.70], [0.90, 0.90, 0.88],
                       [0.35, 0.60, 0.45], [0.85, 0.75, 0.55]][(k + int(z * 10)) % 4]
                m, pos = _on_wall("-y", bx, z + 0.08, (0.16, 0.15), 0.07)
                out.append(("cabinet-item", _at(m, pos),
                            {"color": col + [1], "rough": 0.6, "metal": 0}))
        m, pos = _on_wall("-y", x0, 1.25, (0.92, 1.62), 0.10)
        out.append(("cabinet-glass", _at(m, pos), M_GLASS))
    # OR control touch panel by the foot-end door
    m, pos = _on_wall("-x", 0.9, 1.45, (0.55, 0.38), 0.04)
    out.append(("control-panel", _at(m, pos), M_GREY_PLASTIC))
    m, pos = _on_wall("-x", 0.9, 1.45, (0.49, 0.32), 0.045)
    out.append(("control-screen", _at(m, pos),
                {"color": [0.05, 0.1, 0.12, 1], "rough": 0.1, "metal": 0,
                 "emissive": [0.08, 0.30, 0.40]}))
    # X-ray film viewer on the +y wall
    m, pos = _on_wall("+y", -1.4, 1.55, (1.0, 0.55), 0.06)
    out.append(("viewer-frame", _at(m, pos), M_GREY_PLASTIC))
    m, pos = _on_wall("+y", -1.4, 1.55, (0.92, 0.47), 0.065)
    out.append(("viewer-light", _at(m, pos),
                {"color": [1, 1, 1, 1], "rough": 0.3, "metal": 0,
                 "emissive": [0.85, 0.88, 0.92]}))
    return out


# ------------------------------------------------------------------ table
def table():
    """Flat procedure table (patient supine) + arm board on the patient's
    right for the raised arm."""
    t = cfg.TABLE_TOP_Z
    L = cfg.TABLE_LEN
    cx = cfg.TABLE_CENTER_X
    hw = cfg.TABLE_HALF_W
    out = [("table-base", _rbox([0.80, 0.42, 0.08], [cx, 0, 0.04], 0.06),
            M_WHITE_PLASTIC),
           ("table-column", _rbox([0.32, 0.26, t - 0.20],
                                  [cx, 0, 0.08 + (t - 0.20) / 2], 0.05),
            M_WHITE_PLASTIC),
           ("table-frame", _rbox([L - 0.04, 2 * hw - 0.04, 0.06],
                                 [cx, 0, t - 0.05], 0.02), M_STEEL),
           ("table-pad", _rbox([L, 2 * hw, cfg.TABLE_PAD],
                               [cx, 0, t + cfg.TABLE_PAD / 2 - 0.01], 0.04),
            M_PAD)]
    for sy in (-1, 1):
        out.append(("rail", _at(box(extents=[L - 0.10, 0.012, 0.03]),
                                [cx, sy * (hw + 0.006), t - 0.04]), M_STEEL))
    # arm board above the head on the right (+y) for the raised arm
    pad = t + cfg.TABLE_PAD
    out.append(("armboard", _rbox([0.50, 0.16, 0.035],
                                  [0.86, 0.20, pad - 0.005], 0.03), M_PAD))
    out.append(("armboard-post", _at(cylinder(radius=0.015, height=0.06),
                                     [0.70, 0.22, t - 0.01]), M_STEEL))
    return out


# -------------------------------------------------------------- drapes
def window_rect():
    """Drape window around the scan site: (x0, x1, y0, y1)."""
    x, y = float(cfg.PROBE_SITE[0]), float(cfg.PROBE_SITE[1])
    return x - 0.15, x + 0.15, y - 0.12, y + 0.14


def drapes(patient_parts):
    """A sterile drape moulded over the patient (height field from the
    patient's mesh, smoothed like fabric) with the window over the right
    breast left open."""
    body = trimesh.util.concatenate([m for n, m in patient_parts
                                     if n in ("skin", "gown", "cap",
                                              "prep")])
    x0, x1 = -0.95, 0.70
    y0, y1 = -0.36, 0.36
    nx, ny = 110, 56
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    pad_top = cfg.TABLE_TOP_Z + cfg.TABLE_PAD
    origins = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, 2.0)])
    dirs = np.tile([0, 0, -1.0], (X.size, 1))
    Z = np.full(X.size, pad_top)
    try:
        loc, ray_i, _ = body.ray.intersects_location(origins, dirs,
                                                     multiple_hits=False)
        Z[ray_i] = np.maximum(Z[ray_i], loc[:, 2])
    except Exception:                                       # noqa: BLE001
        pass
    Z = Z.reshape(nx, ny)
    from scipy.ndimage import gaussian_filter, maximum_filter
    Z = maximum_filter(Z, size=5)
    Z = gaussian_filter(Z, 2.0) + 0.010
    edge = np.clip((np.abs(Y) - cfg.TABLE_HALF_W + 0.04) / 0.10, 0, 1)
    Z = Z * (1 - edge) + (pad_top - 0.02 - 0.06 * edge) * edge
    V = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    wx0, wx1, wy0, wy1 = window_rect()
    F = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a = i * ny + j
            cx, cy = (xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2
            if wx0 < cx < wx1 and wy0 < cy < wy1:
                continue
            F += [[a, a + ny, a + 1], [a + 1, a + ny, a + ny + 1]]
    sheet = trimesh.Trimesh(V, np.array(F), process=False)
    out = [("drape", _uv_planar(sheet, (0, 1), 1 / 0.5), M_DRAPE)]
    for sy in (-1, 1):
        zs = np.linspace(pad_top - 0.08, 0.40, 14)
        xx = np.linspace(x0, x1, 80)
        G, FF = [], []
        for k, zz in enumerate(zs):
            u = k / (len(zs) - 1)
            for xv in xx:
                fold = 0.018 * u * np.sin(xv * 38 + sy)
                G.append([xv, sy * (cfg.TABLE_HALF_W + 0.07 + 0.04 * u) +
                          sy * fold, zz])
        for k in range(len(zs) - 1):
            for i in range(len(xx) - 1):
                a = k * len(xx) + i
                FF += [[a, a + 1, a + len(xx)],
                       [a + 1, a + len(xx) + 1, a + len(xx)]]
        panel = trimesh.Trimesh(np.array(G), np.array(FF), process=False)
        back = trimesh.Trimesh(panel.vertices, panel.faces[:, ::-1],
                               process=False)
        out.append(("drape-side", _uv_planar(
            trimesh.util.concatenate([panel, back]), (0, 2), 1 / 0.5),
            M_DRAPE))
    return out


# ---------------------------------------------------------------- lights
def surgical_lights():
    hx, hy, hz = cfg.ROOM_SIZE
    H = 2 * hz
    out = []
    mount = np.array([0.30, 0.0, H])
    out.append(("light-mount", _at(cylinder(radius=0.09, height=0.12),
                                   mount - [0, 0, 0.06]), M_WHITE_PLASTIC))
    out.append(("light-post", _at(cylinder(radius=0.035, height=0.45),
                                  mount - [0, 0, 0.30]), M_WHITE_PLASTIC))
    heads = [np.array([0.20, 0.55, 2.05]), np.array([0.55, -0.40, 2.12])]
    joint = mount - [0, 0, 0.52]
    for h in heads:
        elbow = (joint + h) / 2 + [0, 0, 0.18]
        out.append(("light-arm", _tube([joint, elbow, h + [0, 0, 0.16]], 0.03),
                    M_WHITE_PLASTIC))
        # lamp head tilted toward the abdomen
        aim = np.asarray(cfg.PROBE_SITE, float) - h
        aim /= np.linalg.norm(aim)
        R = trimesh.geometry.align_vectors([0, 0, -1], aim)[:3, :3]
        dome = icosphere(subdivisions=3, radius=0.34)
        dome.vertices[:, 2] = np.clip(dome.vertices[:, 2], -0.02, None) * 0.35
        out.append(("light-head", _at(dome, h, R), M_WHITE_PLASTIC))
        glow = cylinder(radius=0.27, height=0.01, sections=48)
        out.append(("light-glow", _at(glow, h + aim * 0.012, R), M_LAMP_GLOW))
        handle = cylinder(radius=0.025, height=0.12)
        out.append(("light-handle", _at(handle, h + aim * 0.07, R),
                    M_GREY_PLASTIC))
    return out, heads


# ------------------------------------------------------------- equipment
def us_machine():
    """Ultrasound scanner cart at cfg.US_CART_POS: body, control panel with
    keyboard, articulated monitor (its screen shows the live image),
    probe holder and cable."""
    x, y = cfg.US_CART_POS
    c = np.array([x, y, 0.0])
    out = [("us-body", _rbox([0.48, 0.52, 0.72], c + [0, 0, 0.46], 0.05),
            M_WHITE_PLASTIC),
           ("us-plinth", _rbox([0.56, 0.60, 0.10], c + [0, 0, 0.07], 0.06),
            M_GREY_PLASTIC),
           ("us-panel", _rbox([0.50, 0.46, 0.05], c + [0, 0.02, 0.86], 0.03),
            M_GREY_PLASTIC),
           ("us-keys", _at(box(extents=[0.30, 0.16, 0.012]),
                           c + [0, 0.06, 0.892]), M_BLACK)]
    for sx in (-1, 1):
        for sy in (-1, 1):
            out.append(("caster", _at(cylinder(radius=0.035, height=0.03),
                                      c + [sx * 0.22, sy * 0.24, 0.035],
                                      trimesh.transformations.rotation_matrix(
                                          np.pi / 2, [1, 0, 0])[:3, :3]),
                        M_BLACK))
    out.append(("us-arm", _tube([c + [0, -0.10, 0.88], c + [0, -0.10, 1.20]],
                                0.02), M_GREY_PLASTIC))
    m = us_monitor_center()
    R = us_monitor_R()
    out.append(("us-monitor", _rbox([0.46, 0.035, 0.32], m, 0.02, R),
                M_BLACK))
    holder = c + [0.22, 0.10, 0.88]
    out.append(("us-holder", _at(cylinder(radius=0.02, height=0.08), holder),
                M_GREY_PLASTIC))
    return out


def us_monitor_center():
    x, y = cfg.US_CART_POS
    return np.array([x, y - 0.10, 1.38])


def us_monitor_R():
    """Scanner monitor faces the table (toward -x, +y)."""
    ang = np.radians(126.9)
    return trimesh.transformations.rotation_matrix(ang, [0, 0, 1])[:3, :3]


def us_machine_screen_quad():
    m = us_monitor_center()
    R = us_monitor_R()
    n = R @ np.array([0, -1.0, 0])          # screen normal (front face)
    u = R @ np.array([1.0, 0, 0])
    v = np.array([0, 0, 1.0])
    c = m + n * 0.019
    w, h = 0.42, 0.28
    V = np.array([c - u * w / 2 - v * h / 2, c + u * w / 2 - v * h / 2,
                  c + u * w / 2 + v * h / 2, c - u * w / 2 + v * h / 2])
    uv = np.array([[1, 0], [0, 0], [0, 1], [1, 1]], float)
    q = trimesh.Trimesh(V, np.array([[0, 2, 1], [0, 3, 2]]), process=False)
    q.visual = trimesh.visual.TextureVisuals(uv=uv)
    return q


def specimen_kit():
    """On the trolley: specimen pot (formalin) and spare biopsy needles."""
    c = np.array([-0.85, 0.80, 0.0])
    out = [("pot", _at(cylinder(radius=0.035, height=0.07),
                       c + [0.25, -0.12, 0.96]),
            {"color": [0.85, 0.88, 0.90, 0.6], "rough": 0.1, "metal": 0}),
           ("pot-lid", _at(cylinder(radius=0.037, height=0.015),
                           c + [0.25, -0.12, 1.0]),
            {"color": [0.95, 0.75, 0.15, 1], "rough": 0.4, "metal": 0})]
    for k in range(3):
        out.append(("spare-needle", _at(
            cylinder(radius=0.0012, height=0.12),
            c + [-0.25 + k * 0.03, -0.15, 0.925],
            trimesh.transformations.rotation_matrix(np.pi / 2,
                                                    [0, 1, 0])[:3, :3]),
            M_STEEL))
    return out


def trolley():
    """Instrument back-table on the patient's right, near the surgeon."""
    c = np.array([-0.85, 0.80, 0.0])
    out = [("trolley-top", _rbox([0.90, 0.55, 0.02], c + [0, 0, 0.90], 0.02),
            M_STEEL),
           ("trolley-shelf", _rbox([0.86, 0.51, 0.015], c + [0, 0, 0.30], 0.02),
            M_STEEL)]
    for sx in (-1, 1):
        for sy in (-1, 1):
            out.append(("trolley-leg", _at(cylinder(radius=0.014, height=0.88),
                                           c + [sx * 0.42, sy * 0.25, 0.46]),
                        M_STEEL))
            out.append(("caster", _at(icosphere(radius=0.035),
                                      c + [sx * 0.42, sy * 0.25, 0.035]),
                        M_BLACK))
    cloth = _rbox([0.92, 0.57, 0.006], c + [0, 0, 0.914], 0.02)
    out.append(("trolley-cloth", _uv_planar(cloth, (0, 1), 1 / 0.5),
                M_DRAPE_GREEN))
    # laid-out instruments
    rng = np.random.default_rng(3)
    for k in range(9):
        L = rng.uniform(0.14, 0.30)
        ins = cylinder(radius=0.004, height=L)
        R = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
        Rz = trimesh.transformations.rotation_matrix(rng.uniform(-0.15, 0.15),
                                                     [0, 0, 1])
        p = c + [-0.32 + k * 0.075, rng.uniform(-0.10, 0.12), 0.922]
        out.append(("instrument", _at(ins, p, (Rz @ R)[:3, :3]), M_STEEL))
    out.append(("kidney-dish", _at(icosphere(subdivisions=2, radius=0.09),
                                   c + [0.30, 0.15, 0.93]), M_STEEL))
    return out


def iv_stand():
    c = np.array([1.10, 0.58, 0.0])
    out = [("iv-pole", _at(cylinder(radius=0.012, height=2.0),
                           c + [0, 0, 1.0]), M_STEEL)]
    for k in range(5):
        a = 2 * np.pi * k / 5
        leg = _tube([c + [0, 0, 0.06], c + [0.28 * np.cos(a),
                                             0.28 * np.sin(a), 0.03]], 0.012)
        out.append(("iv-leg", leg, M_STEEL))
    out.append(("iv-hook", _tube([c + [-0.12, 0, 1.98], c + [0.12, 0, 1.98]],
                                 0.008), M_STEEL))
    bag = capsule(height=0.16, radius=0.06)
    bag.vertices[:, 1] *= 0.45
    out.append(("iv-bag", _at(bag, c + [0.10, 0, 1.78]), M_BAG))
    line = _bezier(c + [0.10, 0, 1.66], c + [0.10, 0.1, 1.2],
                   np.array([0.80, 0.40, 1.0]),
                   np.array([0.62, 0.22, cfg.TABLE_TOP_Z + 0.12]))
    out.append(("iv-line", _tube(line, 0.002), M_HOSE))
    return out


def robot_cart(base):
    """White robot cart under a KUKA arm (replaces the black pedestal)."""
    x, y, z = base
    out = [("cart-column", _rbox([0.34, 0.34, z - 0.12],
                                 [x, y, 0.12 + (z - 0.12) / 2], 0.08),
            M_WHITE_PLASTIC),
           ("cart-base", _rbox([0.62, 0.56, 0.12], [x, y, 0.08], 0.10),
            M_WHITE_PLASTIC),
           ("cart-trim", _rbox([0.36, 0.36, 0.03], [x, y, z - 0.015], 0.08),
            M_GREY_PLASTIC),
           ("cart-panel", _at(box(extents=[0.12, 0.004, 0.08]),
                              [x - 0.05, y - np.sign(y) * 0.172, z - 0.2]),
            M_SCREEN_GLOW)]
    for sx in (-1, 1):
        for sy in (-1, 1):
            out.append(("caster", _at(cylinder(radius=0.035, height=0.03),
                                      [x + sx * 0.25, y + sy * 0.22, 0.035],
                                      trimesh.transformations.rotation_matrix(
                                          np.pi / 2, [1, 0, 0])[:3, :3]),
                        M_BLACK))
    return out


def monitor_screens():
    """Monitor tower screens (raw ultrasound, AI overlay) + the scanner's
    own screen (front faces, for the live feed)."""
    x, y, z = cfg.MONITOR_POS
    c, sn = np.cos(cfg.MONITOR_YAW), np.sin(cfg.MONITOR_YAW)
    out = []
    for i, s in enumerate((-1, 1)):
        ox, oy = -s * 0.27 * sn, s * 0.27 * c
        cx = x + ox + 0.023 * c + 0.0085 * c
        cy = y + oy + 0.023 * sn + 0.0085 * sn
        # quad facing the surgeon (-x when yaw = pi)
        w, h = 0.44, 0.26
        V = np.array([[cx, cy - w / 2, z - h / 2], [cx, cy + w / 2, z - h / 2],
                      [cx, cy + w / 2, z + h / 2], [cx, cy - w / 2, z + h / 2]])
        if c < 0:                      # seen from -x: mirror u so image reads
            uv = np.array([[1, 0], [0, 0], [0, 1], [1, 1]], float)
        else:
            uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
        F = np.array([[0, 2, 1], [0, 3, 2]]) if c < 0 else \
            np.array([[0, 1, 2], [0, 2, 3]])
        q = trimesh.Trimesh(V, F, process=False)
        q.visual = trimesh.visual.TextureVisuals(uv=uv)
        out.append((f"screen{i}", q, {"color": [1, 1, 1, 1], "rough": 0.1,
                                      "metal": 0, "texture": f"screen{i}",
                                      "emissive_tex": True}))
    out.append(("screen2", us_machine_screen_quad(),
                {"color": [1, 1, 1, 1], "rough": 0.1, "metal": 0,
                 "texture": "screen2", "emissive_tex": True}))
    return out


# --------------------------------------------------------------- textures
def _pnoise(size, feature_px, rng, octaves=1):
    """Seamless (periodic) smooth noise in [-1, 1], features ~feature_px."""
    out = np.zeros((size, size), np.float32)
    amp, tot = 1.0, 0.0
    f = np.fft.fftfreq(size)
    fy, fx = np.meshgrid(f, f, indexing="ij")
    rr = fx ** 2 + fy ** 2
    for o in range(octaves):
        sig = 1.0 / max(feature_px / (2 ** o), 1.0)
        spec = np.fft.fft2(rng.normal(size=(size, size)))
        n = np.real(np.fft.ifft2(spec * np.exp(-rr / (2 * sig ** 2))))
        n /= max(np.abs(n).max(), 1e-9)
        out += amp * n
        tot += amp
        amp *= 0.5
    return out / tot


def _height_to_normal(h, strength):
    """Tangent-space normal map (RGB uint8) from a periodic height field."""
    dx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) * 0.5 * strength
    dy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) * 0.5 * strength
    n = np.dstack([-dx, dy, np.ones_like(h)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    return ((n * 0.5 + 0.5) * 255).astype(np.uint8)


def _rough_map(r):
    """pyrender reads roughness from R and metallic from G."""
    r = np.clip(r, 0, 1)
    return np.dstack([r * 255, np.full_like(r, 255), np.zeros_like(r)]
                     ).astype(np.uint8)


_TEX_SETS = {}


def cv2_blur(a, k):
    """Small periodic box blur (no OpenCV needed)."""
    out = a.copy()
    for ax in (0, 1):
        acc = np.zeros_like(out)
        for d in range(-k, k + 1):
            acc += np.roll(out, d, ax)
        out = acc / (2 * k + 1)
    return out


def make_texture_set(key, size=1024):
    """
    Procedural PBR texture set (all seamless): {"color": RGB uint8,
    "normal": RGB uint8 tangent-space or None, "rough": RG map or None}.
    Stand-ins for photo-scanned CC0 textures, which can't be downloaded in
    every environment; drop real ones in assets/textures/ to override.
    """
    if key in _TEX_SETS:
        return _TEX_SETS[key]
    import os
    cache = os.path.join(cfg.ASSET_DIR, "textures", ".cache",
                         f"{key}_{size}_v2.npz")
    if os.path.isfile(cache):
        try:
            z = np.load(cache)
            out = {k: (z[k] if k in z.files else None)
                   for k in ("color", "normal", "rough")}
            _TEX_SETS[key] = out
            return out
        except Exception:                                   # noqa: BLE001
            pass
    rng = np.random.default_rng(sum(map(ord, key)) * 7919)
    normal = rough = None
    if key == "tiles":
        # Seamless hygienic wall panels (HPL / glass-coated steel, typical
        # of modern ORs): 1.2 m wide, full height, 4 mm shadow joints.
        # The texture spans 2.4 m x 2.4 m (two panels across).
        img = np.full((size, size, 3), (224, 229, 227), np.float32)
        r = np.full((size, size), 0.30, np.float32)
        h = np.ones((size, size), np.float32)
        half = size // 2
        for j in range(2):
            sl = slice(j * half, (j + 1) * half)
            img[:, sl] += rng.normal(0, 1.6)            # panel-to-panel tint
        img += _pnoise(size, 220, rng, 3)[..., None] * 2.2   # soft mottling
        img += _pnoise(size, 3, rng)[..., None] * 0.8        # fine grain
        r += 0.05 * _pnoise(size, 140, rng, 2)                # sheen varies
        g = max(2, size // 600)                     # joint ~4 mm
        for x in (0, half):                          # vertical joints
            for xx in range(x - g, x + g):
                img[:, xx % size] = (150, 158, 158)
                h[:, xx % size] = 0.0
                r[:, xx % size] = 0.7
        for yy in range(-g, g):                      # horizontal joint
            img[yy % size, :] = (158, 165, 165)
            h[yy % size, :] = 0.0
        h = cv2_blur(h, 3)
        normal = _height_to_normal(h, 3.0)
        rough = _rough_map(r)
    elif key == "floor":
        # Seamless sheet vinyl, blue-grey with fine chips (OR flooring),
        # slightly glossy from the floor polish.
        base = np.array((141, 154, 157), np.float32)
        img = np.tile(base, (size, size, 1))
        img += _pnoise(size, 260, rng, 4)[..., None] * 6.0       # mottling
        img += _pnoise(size, 5, rng)[..., None] * 2.5
        chips = rng.random((size, size))
        for thr, c in ((0.9975, (196, 205, 208)), (0.9960, (108, 120, 126)),
                       (0.9990, (172, 160, 140))):
            m = chips > thr if thr > 0.997 else chips < (1 - thr)
            m = m | np.roll(m, 1, 0) | np.roll(m, 1, 1)
            img[m] = c
        h = _pnoise(size, 3, rng) * 0.5 + _pnoise(size, 120, rng) * 0.5
        scuff = _pnoise(size, 160, rng, 3)
        normal = _height_to_normal(h, 0.8)
        rough = _rough_map(0.30 + 0.10 * scuff)
    elif key == "border":
        img = np.full((size // 4, size // 4, 3), (86, 98, 104), np.float32)
        img += _pnoise(size // 4, 40, rng, 3)[..., None] * 4
    elif key == "ceiling":
        # 600 mm acoustic ceiling tiles (texture spans 1.2 m: 2 x 2 tiles)
        img = np.full((size, size, 3), (238, 239, 238), np.float32)
        img += _pnoise(size, 2, rng)[..., None] * 5            # perforation
        h = np.ones((size, size), np.float32)
        half = size // 2
        g = max(3, size // 200)
        for c in (0, half):
            for k in range(c - g, c + g):
                img[:, k % size] = (205, 207, 207)
                img[k % size, :] = (205, 207, 207)
                h[:, k % size] = 0
                h[k % size, :] = 0
        normal = _height_to_normal(cv2_blur(h, 2), 1.5)
    elif key == "fabric":
        # non-woven SMS drape: soft weave + fibre noise
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        weave = np.sin(xx * 0.9) * np.sin(yy * 0.9)
        fib = _pnoise(size, 2, rng)
        img = np.full((size, size, 3), 200, np.float32) + \
            (weave * 6 + fib * 6)[..., None]
        img += _pnoise(size, 80, rng)[..., None] * 5
        img = np.clip(img, 0, 255) / 200.0 * 255.0
        normal = _height_to_normal(weave * 0.5 + fib * 0.6, 0.9)
    else:
        img = np.full((8, 8, 3), 255, np.float32)
    out = {"color": np.clip(img, 0, 255).astype(np.uint8),
           "normal": normal, "rough": rough}
    _TEX_SETS[key] = out
    try:                                  # cache: faster panel start-up
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez_compressed(cache, **{k: v for k, v in out.items()
                                      if v is not None})
    except Exception:                                       # noqa: BLE001
        pass
    return out


def make_texture(key, size=1024):
    """Procedural colour texture (uint8 RGB)."""
    return make_texture_set(key, size)["color"]


