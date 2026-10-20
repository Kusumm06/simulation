"""
Realistic renderer for the OPERATING ROOM panel (adapted from the reference
project's or_render.py).

PyBullet stays the kinematics engine. This module only DRAWS:

  * the two KUKA arms (white covers, aluminium joints, clear sterile
    sleeves), the probe and the biopsy gun + needle are mirrored from
    PyBullet link by link every frame;
  * the primitive room / table / patient / carts are replaced by realistic
    procedural models (props.py, humans.py) built from the same config;
  * the breast is drawn translucent with the PLANNING anatomy inside it
    (lesion ellipsoid, vessels, fascia, pectoralis, ribs, ligaments), all
    registered to the probe exactly as the planner uses them;
  * the live ultrasound slice is drawn as a textured quad in the probe's
    image plane under the transducer;
  * the monitor tower and the scanner show the live raw image and overlay.
If pyrender cannot start, main.py falls back to PyBullet's renderer.
"""
import os
import sys
import time

import numpy as np

if not hasattr(np, "infty"):           # pyrender 0.1.45 vs numpy 2
    np.infty = np.inf

if sys.platform.startswith("linux") and "PYOPENGL_PLATFORM" not in \
        os.environ and not os.environ.get("DISPLAY"):
    os.environ["PYOPENGL_PLATFORM"] = "egl"

import cv2                                                  # noqa: E402
import pybullet as p                                        # noqa: E402
import trimesh                                              # noqa: E402

from ultrabot_sim import config as cfg                      # noqa: E402

_GEOM_SPHERE, _GEOM_BOX, _GEOM_CYLINDER, _GEOM_MESH, _GEOM_PLANE, \
    _GEOM_CAPSULE = 2, 3, 4, 5, 6, 7
_REPLACED = ("room", "table", "patient", "carts", "monitors", "breast")
MAX_RIBS = 10


def _quat_to_mat(pos, quat):
    T = np.eye(4)
    T[:3, :3] = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
    T[:3, 3] = pos
    return T


def _warn(msg):
    print(f"[or-panel] {msg}")


# ---------------------------------------------------------------- tool meshes
def probe_mesh():
    """Linear transducer in the probe BODY frame (box centre, +z toward the
    face): tapered housing, darker lens at the face, cable at the top."""
    bx, by, bz = cfg.PROBE_BODY
    parts = []
    grip = trimesh.creation.box(extents=[bx * 0.62, by * 1.6, bz * 0.62])
    grip.apply_translation([0, 0, -bz * 0.19])
    head = trimesh.creation.box(extents=[bx, by, bz * 0.38])
    head.apply_translation([0, 0, bz * 0.31])
    parts += [grip, head]
    lens = trimesh.creation.box(extents=[bx * 0.96, by * 0.85, 0.004])
    lens.apply_translation([0, 0, bz / 2 - 0.002])
    cable = trimesh.creation.cylinder(radius=0.0045, height=0.09)
    cable.apply_translation([0, 0, -bz / 2 - 0.045])
    return trimesh.util.concatenate(parts), lens, cable


def gun_mesh():
    """Spring-loaded biopsy gun in the GUN BODY frame (+z = needle axis)."""
    L = cfg.GUN_LEN
    r = cfg.GUN_RADIUS
    body = trimesh.creation.box(extents=[2 * r * 0.9, 2 * r, L * 0.92])
    nose = trimesh.creation.cylinder(radius=r * 0.45, height=L * 0.10,
                                     sections=24)
    nose.apply_translation([0, 0, L * 0.46])
    window = trimesh.creation.box(extents=[2 * r * 0.92, r * 0.8, L * 0.35])
    window.apply_translation([0, r * 0.65, L * 0.05])
    trigger = trimesh.creation.box(extents=[r * 0.6, r * 0.9, L * 0.12])
    trigger.apply_translation([0, -r * 1.2, -L * 0.30])
    return trimesh.util.concatenate([body, nose]), window, trigger


class ORRenderer:
    def __init__(self, scene, size=None, offscreen=True):
        import pyrender
        if getattr(cfg, "OR_FAST_GL", True):
            from ultrabot_sim import fastgl
            fastgl.apply()
        self.pr = pyrender
        self.sim = scene
        self.size = size or cfg.OR_SIZE
        self.q = q = cfg.OR_QUALITY_PRESETS[cfg.OR_QUALITY]
        self.ssaa = q["supersample"]
        w, h = self.size
        t0 = time.perf_counter()
        self.renderer = (pyrender.OffscreenRenderer(w * self.ssaa,
                                                    h * self.ssaa)
                         if offscreen else None)
        self.scene = pyrender.Scene(bg_color=[0.05, 0.06, 0.07, 1.0],
                                    ambient_light=q["ambient"])
        self.nodes = []
        self._mesh_cache = {}
        self._tex_cache = {}
        self._screen_nodes = {}
        self._screen_imgs = {}
        replaced = set()
        for g in _REPLACED:
            replaced.update(getattr(scene, "groups", {}).get(g, []))
        self._build_from_pybullet(skip=replaced if cfg.OR_REALISTIC
                                  else set())
        if cfg.OR_REALISTIC:
            self._build_realistic()
        self._build_breast()
        self._add_lights()
        self.cam_node = self.scene.add(
            pyrender.PerspectiveCamera(yfov=np.radians(cfg.OR_FOV),
                                       aspectRatio=w / h, znear=0.02,
                                       zfar=15.0))
        self.view_index = 0
        _warn(f"scene ready in {time.perf_counter() - t0:.1f} s "
              f"({len(self.scene.mesh_nodes)} meshes)")

    # ---------------------------------------------------------- materials
    def _texture(self, key):
        if key not in self._tex_cache:
            from ultrabot_sim import props
            gen = props.make_texture_set(key)
            maps = {"color": gen["color"], "normal": gen["normal"],
                    "rough": gen["rough"]}
            if not self.q.get("normal_maps", True):
                maps["normal"] = None
            sampler = self.pr.Sampler(magFilter=9729, minFilter=9987,
                                      wrapS=10497, wrapT=10497)
            self._tex_cache[key] = {
                k: (None if v is None else
                    self.pr.Texture(source=np.ascontiguousarray(v),
                                    source_channels="RGB", sampler=sampler))
                for k, v in maps.items()}
        return self._tex_cache[key]

    @staticmethod
    def _lin(c):
        c = np.clip(np.asarray(c, float), 0.0, 1.0)
        return np.where(c <= 0.04045, c / 12.92,
                        ((c + 0.055) / 1.055) ** 2.4).tolist()

    def _mat(self, spec):
        rgba = spec.get("color", [0.7, 0.7, 0.7, 1.0])
        rgba = self._lin(rgba[:3]) + [rgba[3]]
        kw = dict(baseColorFactor=rgba,
                  metallicFactor=spec.get("metal", 0.0),
                  roughnessFactor=spec.get("rough", 0.6),
                  alphaMode="BLEND" if rgba[3] < 0.99 else "OPAQUE",
                  doubleSided=rgba[3] < 0.99 or spec.get("double", False))
        g = self.q.get("bounce", 0.0)
        tex = None
        if spec.get("texture") and not spec["texture"].startswith("screen"):
            ts = self._texture(spec["texture"])
            tex = ts["color"]
            kw["baseColorTexture"] = tex
            if ts.get("normal") is not None:
                kw["normalTexture"] = ts["normal"]
            if ts.get("rough") is not None:
                kw["metallicRoughnessTexture"] = ts["rough"]
                kw["roughnessFactor"] = 1.0
        if spec.get("emissive"):
            kw["emissiveFactor"] = self._lin(spec["emissive"])
        elif g > 0:
            kw["emissiveFactor"] = [c * g * spec.get("bounce_k", 1.0)
                                    for c in rgba[:3]]
            if tex is not None:
                kw["emissiveTexture"] = tex
        return self.pr.MetallicRoughnessMaterial(**kw)

    def _material_guess(self, rgba):
        r, g, b, a = rgba
        metalish = abs(r - g) < 0.06 and abs(g - b) < 0.08 and r > 0.55
        return self._mat({"color": list(rgba),
                          "rough": 0.3 if metalish else 0.6,
                          "metal": 0.9 if metalish else 0.0})

    # ---------------------------------------------------------- geometry
    def _trimesh_for(self, vd):
        geom, dims, fname = vd[2], vd[3], vd[4]
        if geom == _GEOM_BOX:
            return trimesh.creation.box(extents=np.asarray(dims[:3]))
        if geom == _GEOM_SPHERE:
            return trimesh.creation.icosphere(subdivisions=3, radius=dims[0])
        if geom == _GEOM_CYLINDER:
            return trimesh.creation.cylinder(radius=dims[1], height=dims[0],
                                             sections=32)
        if geom == _GEOM_CAPSULE:
            return trimesh.creation.capsule(height=dims[0], radius=dims[1],
                                            count=[24, 24])
        if geom == _GEOM_MESH:
            path = fname.decode() if isinstance(fname, bytes) else fname
            if not path or not os.path.isfile(path):
                return None
            key = (path, tuple(np.round(dims, 4)))
            if key not in self._mesh_cache:
                m = trimesh.load(path, force="mesh")
                m.apply_scale(np.asarray(dims[:3]))
                self._mesh_cache[key] = m
            return self._mesh_cache[key].copy()
        return None

    def _build_from_pybullet(self, skip):
        self._sleeve_mat = self.pr.MetallicRoughnessMaterial(
            baseColorFactor=self._lin([0.90, 0.94, 0.97]) + [0.11],
            metallicFactor=0.0, roughnessFactor=0.06, alphaMode="BLEND",
            doubleSided=True)
        cid = self.sim.client
        arm_ids = set(self.sim.arms.values())
        sim = self.sim
        for b in range(p.getNumBodies(physicsClientId=cid)):
            body = p.getBodyUniqueId(b, physicsClientId=cid)
            if body in skip:
                continue
            if body == sim.probe_body:
                housing, lens, cable = probe_mesh()
                for tm, spec in ((housing, {"color": [0.90, 0.91, 0.92, 1],
                                            "rough": 0.35}),
                                 (lens, {"color": [0.15, 0.17, 0.20, 1],
                                         "rough": 0.5}),
                                 (cable, {"color": [0.25, 0.27, 0.30, 1],
                                          "rough": 0.5})):
                    node = self.scene.add(self.pr.Mesh.from_trimesh(
                        tm, material=self._mat(spec)))
                    self.nodes.append((node, body, -1, np.eye(4)))
                continue
            if body == sim.gun_body:
                gb, win, trig = gun_mesh()
                for tm, spec in ((gb, {"color": [0.20, 0.42, 0.72, 1],
                                       "rough": 0.35}),
                                 (win, {"color": [0.85, 0.88, 0.92, 0.6],
                                        "rough": 0.1}),
                                 (trig, {"color": [0.12, 0.13, 0.15, 1],
                                         "rough": 0.5})):
                    node = self.scene.add(self.pr.Mesh.from_trimesh(
                        tm, material=self._mat(spec), smooth=False))
                    self.nodes.append((node, body, -1, np.eye(4)))
                continue
            for vd in p.getVisualShapeData(body, physicsClientId=cid):
                rgba = vd[7]
                if rgba[3] <= 0.01:
                    continue
                tm = self._trimesh_for(vd)
                if tm is None:
                    continue
                link = vd[1]
                local = _quat_to_mat(vd[5], vd[6])
                if body in arm_ids:
                    mats = [self._robot_material(link)]
                    if cfg.OR_REALISTIC and cfg.OR_STERILE_SLEEVES and \
                            link >= 1:
                        mats.append("sleeve")
                else:
                    mats = [self._material_guess(rgba)]
                for mat in mats:
                    mesh_tm = tm
                    if mat == "sleeve":
                        mesh_tm = tm.copy()
                        c = mesh_tm.bounds.mean(axis=0)
                        mesh_tm.apply_translation(-c)
                        mesh_tm.apply_scale(1.05)
                        mesh_tm.apply_translation(c)
                        mat = self._sleeve_mat
                    mesh = self.pr.Mesh.from_trimesh(
                        mesh_tm, material=mat, smooth=vd[2] != _GEOM_BOX)
                    node = self.scene.add(mesh)
                    self.nodes.append((node, body, link, local))

    def _robot_material(self, link):
        if link in (1, 3, 5):
            return self._mat({"color": [0.85, 0.86, 0.88, 1.0], "metal": 0.9,
                              "rough": 0.28})
        return self._mat({"color": [0.96, 0.96, 0.95, 1.0], "metal": 0.0,
                          "rough": 0.30})

    _SMOOTH = ("surgeon", "radiologist", "cove", "clock", "patient", "drape",
               "iv-line", "iv-bag", "light-head", "light-arm", "caster",
               "pot")

    def _add_static(self, name, tm, spec):
        smooth = name.startswith(self._SMOOTH)
        try:
            mesh = self.pr.Mesh.from_trimesh(tm, material=self._mat(spec),
                                             smooth=smooth)
            return self.scene.add(mesh, name=name)
        except Exception as e:                              # noqa: BLE001
            _warn(f"could not add {name}: {e}")
            return None

    def _add_merged(self, items):
        import json
        groups = {}
        for name, tm, spec in items:
            if spec.get("texture") or not isinstance(tm, trimesh.Trimesh):
                self._add_static(name, tm, spec)
                continue
            smooth = name.startswith(self._SMOOTH)
            key = (json.dumps(spec, sort_keys=True, default=str), smooth)
            groups.setdefault(key, (name, spec, []))[2].append(tm)
        for (_, smooth), (name, spec, tms) in groups.items():
            tm = tms[0] if len(tms) == 1 else trimesh.util.concatenate(
                [trimesh.Trimesh(t.vertices, t.faces, process=False)
                 for t in tms])
            self._add_static(name, tm, spec)

    def _build_realistic(self):
        from ultrabot_sim import humans, props
        items = []
        patient_parts = None
        for who, fn in (("patient", humans.build_patient),
                        ("radiologist", humans.build_radiologist)):
            try:
                parts = fn()
                for mat, m in parts:
                    rgba, rough, metal = humans.MATERIALS[mat]
                    items.append((f"{who}-{mat}", m, {"color": rgba,
                                                      "rough": rough,
                                                      "metal": metal}))
                if who == "patient":
                    patient_parts = parts
            except Exception as e:                          # noqa: BLE001
                _warn(f"{who} model failed ({e}); primitive figure used")
                if who == "patient":
                    self._build_from_pybullet_group("patient")
        builders = [("room", props.room), ("table", props.table),
                    ("trolley", props.trolley), ("kit", props.specimen_kit),
                    ("iv", props.iv_stand), ("us machine", props.us_machine)]
        for name, fn in builders:
            try:
                items += fn()
            except Exception as e:                          # noqa: BLE001
                _warn(f"{name} failed ({e})")
        try:
            lights, self._lamp_heads = props.surgical_lights()
            items += lights
        except Exception as e:                              # noqa: BLE001
            _warn(f"lights failed ({e})")
            self._lamp_heads = []
        for base in (cfg.CART_PROBE, cfg.CART_NEEDLE):
            items += props.robot_cart(base)
        if patient_parts is not None:
            try:
                items += props.drapes(patient_parts)
            except Exception as e:                          # noqa: BLE001
                _warn(f"drapes failed ({e})")
        self._build_monitor_tower(items)
        self._add_merged(items)
        for name, tm, spec in props.monitor_screens():
            self._screen_nodes[name] = (tm, None)

    def _build_monitor_tower(self, items):
        from ultrabot_sim import props
        x, y, z = cfg.MONITOR_POS
        yaw = cfg.MONITOR_YAW
        c, s = np.cos(yaw), np.sin(yaw)
        items.append(("tower-pole", props._at(trimesh.creation.cylinder(
            radius=0.045, height=z), [x, y, z / 2]), props.M_GREY_PLASTIC))
        items.append(("tower-base", props._at(trimesh.creation.cylinder(
            radius=0.25, height=0.04), [x, y, 0.02]), props.M_WHITE_PLASTIC))
        Rz = trimesh.transformations.rotation_matrix(yaw, [0, 0, 1])[:3, :3]
        for k in (-1, 1):
            ox, oy = -k * 0.27 * s, k * 0.27 * c
            items.append(("tower-screen", props._rbox(
                [0.04, 0.50, 0.32], [x + ox, y + oy, z], 0.01, Rz),
                props.M_BLACK))

    def _build_from_pybullet_group(self, group):
        ids = set(getattr(self.sim, "groups", {}).get(group, []))
        allb = {p.getBodyUniqueId(i, physicsClientId=self.sim.client)
                for i in range(p.getNumBodies(physicsClientId=self.sim.client))}
        self._build_from_pybullet(skip=allb - ids)

    # ---------------------------------------------------------- breast
    def _build_breast(self):
        """Translucent breast + internal anatomy + the ultrasound plane."""
        from ultrabot_sim.us_geometry import ProbePose
        pr = self.pr
        tm = self.sim._breast_tm
        self._anat_pose = ProbePose.nominal()
        self.scene.add(pr.Mesh.from_trimesh(tm, material=self._mat(
            {"color": [0.92, 0.74, 0.66, 0.42], "rough": 0.5}), smooth=True),
            name="breast")
        # nipple-areola (model)
        pose = self._anat_pose
        nip = pose.probe_to_world([cfg.NIPPLE_X_MM, 0.0, 0.0])
        from ultrabot_sim.anatomy import _skin_profile
        dz = float(_skin_profile(abs(cfg.NIPPLE_X_MM) -
                                 cfg.US_WIDTH_MM / 2.0)) / 1000.0
        disc = trimesh.creation.cylinder(radius=cfg.NIPPLE_R_MM / 1000.0,
                                         height=0.0015, sections=40)
        disc.apply_translation(nip + np.array([0, 0, -dz + 0.0006]))
        self.scene.add(pr.Mesh.from_trimesh(disc, material=self._mat(
            {"color": [0.62, 0.40, 0.36, 1.0], "rough": 0.6})), name="nipple")

        def unit(tm, spec):
            return self.scene.add(pr.Mesh.from_trimesh(
                tm, material=self._mat(spec), smooth=True))
        self._an_nodes = {
            "lesion": unit(trimesh.creation.icosphere(subdivisions=3,
                                                      radius=1.0),
                           {"color": [0.45, 0.18, 0.16, 1.0], "rough": 0.6}),
            "fascia": unit(trimesh.creation.box(extents=[1, 1, 1]),
                           {"color": [0.85, 0.80, 0.70, 0.9], "rough": 0.6}),
            "muscle": unit(trimesh.creation.box(extents=[1, 1, 1]),
                           {"color": [0.62, 0.20, 0.20, 0.85]}),
            "vessels": [unit(trimesh.creation.cylinder(radius=1.0, height=1.0,
                                                       sections=20),
                             {"color": [0.75, 0.10, 0.12, 1.0],
                              "rough": 0.4})
                        for _ in cfg.VESSELS],
            "ribs": [unit(trimesh.creation.cylinder(radius=1.0, height=1.0,
                                                    sections=20),
                          {"color": [0.93, 0.90, 0.82, 1.0], "rough": 0.5})
                     for _ in range(MAX_RIBS)],
        }
        # ultrasound image plane (textured, updated with the live image)
        hw = cfg.US_WIDTH_MM / 2.0
        D = cfg.US_DEPTH_MM
        V = pose.probe_to_world(np.array([[-hw, 0, D], [hw, 0, D],
                                          [hw, 0, 0], [-hw, 0, 0]]))
        q = trimesh.Trimesh(V, np.array([[0, 1, 2], [0, 2, 3]]),
                            process=False)
        q.visual = trimesh.visual.TextureVisuals(
            uv=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float))
        self._screen_nodes["usplane"] = (q, None)
        self.set_anatomy(None)

    def set_anatomy(self, an):
        """Pose the anatomy nodes from Anatomy.to_dict() (probe frame mm)."""
        pose = self._anat_pose
        R = pose.R

        def T_of(center_mm, scale_mm, rot=np.eye(3)):
            T = np.eye(4)
            T[:3, :3] = R @ rot @ np.diag(np.asarray(scale_mm) / 1000.0)
            T[:3, 3] = pose.probe_to_world(center_mm)
            return T
        hide = np.diag([1e-6, 1e-6, 1e-6, 1.0])
        n = self._an_nodes
        if an is None:
            for k in ("lesion", "fascia", "muscle"):
                self.scene.set_pose(n[k], hide)
            for nd in n["vessels"] + n["ribs"]:
                self.scene.set_pose(nd, hide)
            return
        if an.get("lesion"):
            x, z, a, b = an["lesion"]
            self.scene.set_pose(n["lesion"], T_of([x, 0, z], [a, a, b]))
        else:
            self.scene.set_pose(n["lesion"], hide)
        k = an["fascia_slope"]
        ang = np.arctan(k)
        Ry = trimesh.transformations.rotation_matrix(-ang, [0, 1, 0])[:3, :3]
        L = 110.0
        self.scene.set_pose(n["fascia"], T_of([0, 0, an["fascia_z0"] + 0.6],
                                              [L, 40, 1.2], Ry))
        self.scene.set_pose(n["muscle"], T_of(
            [0, 0, an["fascia_z0"] + 1.2 + (cfg.MUSCLE_THICK_MM - 1.2) / 2],
            [L, 40, cfg.MUSCLE_THICK_MM - 1.2], Ry))
        Rx = trimesh.transformations.rotation_matrix(np.pi / 2,
                                                     [1, 0, 0])[:3, :3]
        for i, nd in enumerate(n["vessels"]):
            if i < len(an["vessels"]):
                x, z, r = an["vessels"][i]
                self.scene.set_pose(nd, T_of([x, 0, z], [r, r, 45], Rx))
            else:
                self.scene.set_pose(nd, hide)
        ribs = [rb for rb in an["ribs"] if abs(rb[0]) < 60][:MAX_RIBS]
        for i, nd in enumerate(n["ribs"]):
            if i < len(ribs):
                x, z, r = ribs[i]
                self.scene.set_pose(nd, T_of([x, 0, z], [r, r, 45], Rx))
            else:
                self.scene.set_pose(nd, hide)

    # ---------------------------------------------------------- lights
    def _add_lights(self):
        pr, q = self.pr, self.q
        target = np.asarray(cfg.PROBE_SITE, float)
        heads = getattr(self, "_lamp_heads", None) or \
            [np.array([0.3, 0.0, 2.3])]
        for h in heads:
            pose = np.eye(4)
            pose[:3, :3] = trimesh.geometry.align_vectors(
                [0, 0, -1], target - h)[:3, :3]
            pose[:3, 3] = h
            self.scene.add(pr.SpotLight(color=[1.0, 0.98, 0.94],
                                        intensity=q["lamp"] / len(heads),
                                        innerConeAngle=np.radians(10),
                                        outerConeAngle=np.radians(28)),
                           pose=pose)
        for az in (45, 135, 225, 315):
            d = np.array([np.cos(np.radians(az)) * 0.55,
                          np.sin(np.radians(az)) * 0.55, -1.0])
            pose = np.eye(4)
            pose[:3, :3] = trimesh.geometry.align_vectors([0, 0, -1], d)[:3, :3]
            self.scene.add(pr.DirectionalLight(color=[1.0, 0.99, 0.97],
                                               intensity=q["room"]), pose=pose)

    # ---------------------------------------------------------- camera
    def _camera_pose(self, dist, yaw, pitch, target):
        V = np.array(p.computeViewMatrixFromYawPitchRoll(
            target, dist, yaw, pitch, 0, 2)).reshape(4, 4).T
        return np.linalg.inv(V)

    def set_view(self, i):
        self.view_index = i

    def resize(self, size):
        w, h = int(size[0]), int(size[1])
        self.size = (w, h)
        self.renderer.viewport_width = w * self.ssaa
        self.renderer.viewport_height = h * self.ssaa
        self.cam_node.camera.aspectRatio = w / h
        self._post_lut = None

    def _gl_info(self):
        try:
            from OpenGL import GL
            ven = GL.glGetString(GL.GL_VENDOR)
            ren = GL.glGetString(GL.GL_RENDERER)
            return f"{ven.decode()} / {ren.decode()}"
        except Exception as e:                              # noqa: BLE001
            return f"unknown ({e})"

    def set_screen_images(self, imgs):
        """imgs: dict name -> BGR image (screen0 raw, screen1 overlay,
        screen2 scanner, usplane slice)."""
        if not imgs:
            return
        self._screen_imgs.update({k: v for k, v in imgs.items()
                                  if v is not None})
        self._screen_dirty = True

    def _update_screens(self):
        if not getattr(self, "_screen_dirty", True):
            return
        self._screen_dirty = False
        for name, (tm, node) in list(self._screen_nodes.items()):
            img = self._screen_imgs.get(name)
            if img is None:
                if node is not None:
                    continue
                img = np.full((18, 32, 3), (20, 30, 34), np.uint8)
            if node is not None:
                self.scene.remove_node(node)
            tex = self.pr.Texture(source=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                                  source_channels="RGB")
            mat = self.pr.MetallicRoughnessMaterial(
                baseColorFactor=[0.0, 0.0, 0.0, 1.0], metallicFactor=0.0,
                roughnessFactor=0.15, emissiveTexture=tex,
                emissiveFactor=[1.0, 1.0, 1.0],
                doubleSided=(name == "usplane"))
            node = self.scene.add(self.pr.Mesh.from_trimesh(tm, material=mat))
            self._screen_nodes[name] = (tm, node)

    # ---------------------------------------------------------- render
    def snapshot(self):
        cid = self.sim.client
        poses = []
        base_cache = {}
        for node, body, link, local in self.nodes:
            if link == -1:
                if body not in base_cache:
                    pos, orn = p.getBasePositionAndOrientation(
                        body, physicsClientId=cid)
                    base_cache[body] = _quat_to_mat(pos, orn)
                W = base_cache[body]
            else:
                ls = p.getLinkState(body, link, computeForwardKinematics=True,
                                    physicsClientId=cid)
                W = _quat_to_mat(ls[4], ls[5])
            poses.append((node, W @ local))
        return {"poses": poses, "tops": arm_tops(self.sim),
                "acts": dict(getattr(self.sim, "_label_text", {}) or {}),
                "view": self.view_index}

    def render(self):
        return self.draw(self.snapshot())

    def draw(self, snap):
        for node, T in snap["poses"]:
            self.scene.set_pose(node, T)
        if snap.get("anat", "keep") != "keep":
            self.set_anatomy(snap["anat"])
        if self._screen_nodes:
            self._update_screens()
        if snap.get("size") and tuple(snap["size"]) != tuple(self.size):
            self.resize(snap["size"])
        if snap.get("cam") is not None:
            dist, yaw, pitch, tgt = snap["cam"]
        else:
            name, dist, yaw, pitch, tgt = cfg.OR_VIEWS[
                snap["view"] % len(cfg.OR_VIEWS)]
        self.scene.set_pose(self.cam_node,
                            self._camera_pose(dist, yaw, pitch, tgt))
        flags = self.pr.RenderFlags.NONE
        if self.q["shadows"]:
            flags |= self.pr.RenderFlags.SHADOWS_SPOT
        color, _ = self.renderer.render(self.scene, flags=flags)
        if self.ssaa > 1:
            color = cv2.resize(color, self.size, interpolation=cv2.INTER_AREA)
        if self.q.get("post", False):
            color = self._post(color)
        if getattr(cfg, "OR_CAPTIONS", True):
            color = self._captions(color, dist, yaw, pitch, tgt, snap)
        return color

    def _post(self, rgb):
        if getattr(self, "_post_lut", None) is None:
            x = np.arange(256) / 255.0
            s = np.clip(x - 0.20 * np.sin(2 * np.pi * x) / (2 * np.pi), 0, 1)
            self._post_lut = (s * 255).astype(np.uint8)
            h, w = rgb.shape[:2]
            yy, xx = np.mgrid[0:h, 0:w]
            r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
            self._post_vig = (1.0 - 0.16 * np.clip(r2 / 2, 0, 1) ** 1.3
                              )[..., None].astype(np.float32)
        out = cv2.LUT(rgb, self._post_lut)
        if self._post_vig.shape[:2] == out.shape[:2]:
            out = (out * self._post_vig).astype(np.uint8)
        return out

    def _project(self, P, dist, yaw, pitch, tgt):
        pose = self._camera_pose(dist, yaw, pitch, tgt)
        v = np.linalg.inv(pose) @ np.r_[np.asarray(P, float), 1.0]
        if v[2] >= -1e-3:
            return None
        w, h = self.size
        f = (h / 2) / np.tan(np.radians(cfg.OR_FOV) / 2)
        return (w / 2 + f * v[0] / -v[2], h / 2 - f * v[1] / -v[2])

    def _captions(self, rgb, dist, yaw, pitch, tgt, snap):
        out = rgb.copy()
        H, W = out.shape[:2]
        acts = snap.get("acts", {})
        placed = []
        for name, title, sub, col in (
                ("probe", "PROBE ARM", "linear US probe, fixed on the skin",
                 (60, 130, 230)),
                ("needle", "NEEDLE ARM", "14G core biopsy gun",
                 (240, 140, 40))):
            top = snap.get("tops", {}).get(name)
            if top is None:
                continue
            uv = self._project(top, dist, yaw, pitch, tgt)
            if uv is None:
                continue
            x, y = int(uv[0]), int(uv[1])
            if not (-40 < x < W + 40 and -20 < y < H + 20):
                continue
            line2 = acts.get(name) or sub
            fs = 0.36 if W < 600 else 0.42 * min(1.8, (W / 640) ** 0.6)
            (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX,
                                          fs, 1)
            (sw, sh), _ = cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX,
                                          fs * 0.85, 1)
            bw, bh = max(tw, sw) + 16, th + sh + 16
            bx = int(np.clip(x - bw // 2, 4, W - bw - 4))
            by = int(np.clip(y - bh - 14, 4, H - bh - 4))
            for (px, py, pw, ph) in placed:
                if bx < px + pw and px < bx + bw and by < py + ph and \
                        py < by + bh:
                    by = py + ph + 4 if py + ph + 4 + bh < H else py - bh - 4
            placed.append((bx, by, bw, bh))
            cv2.line(out, (bx + bw // 2, by + bh), (x, y), (250, 250, 250), 1,
                     cv2.LINE_AA)
            cv2.circle(out, (x, y), 3, col, -1, cv2.LINE_AA)
            box = out[by:by + bh, bx:bx + bw].astype(np.float32)
            out[by:by + bh, bx:bx + bw] = (box * 0.35 +
                                           np.array([18, 22, 26]) * 0.65
                                           ).astype(np.uint8)
            cv2.rectangle(out, (bx, by), (bx + 3, by + bh - 1), col, -1)
            cv2.putText(out, title, (bx + 9, by + th + 5),
                        cv2.FONT_HERSHEY_DUPLEX, fs, (245, 245, 245), 1,
                        cv2.LINE_AA)
            cv2.putText(out, line2, (bx + 9, by + th + sh + 11),
                        cv2.FONT_HERSHEY_SIMPLEX, fs * 0.85, (190, 200, 205),
                        1, cv2.LINE_AA)
        cv2.putText(out, "Kinematic research simulation - not for clinical "
                    "use", (8, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (200, 210, 220), 1, cv2.LINE_AA)
        return out

    def close(self):
        try:
            self.renderer.delete()
        except Exception:                                   # noqa: BLE001
            pass


def arm_tops(sim):
    tops = {}
    cid = sim.client
    for name in ("probe", "needle"):
        body = sim.arms.get(name)
        if body is None:
            continue
        try:
            nj = p.getNumJoints(body, physicsClientId=cid)
            pts = [p.getLinkState(body, j, physicsClientId=cid)[4]
                   for j in range(min(nj, 5))]
            tops[name] = np.array(max(pts, key=lambda q: q[2])) + [0, 0, 0.10]
        except Exception:                                   # noqa: BLE001
            pass
    return tops


MONITOR_RES = (640, 360)


def screen_images(raw, overlay):
    """Images for the screens: tower (raw, overlay), scanner (raw), and the
    3D ultrasound slice (overlay, full pane aspect)."""
    if raw is None:
        return None
    W, H = MONITOR_RES
    r = cv2.resize(raw, (W, H), interpolation=cv2.INTER_AREA)
    o = cv2.resize(overlay if overlay is not None else raw, (W, H),
                   interpolation=cv2.INTER_AREA)
    sl = cv2.resize(overlay if overlay is not None else raw, (495, 360),
                    interpolation=cv2.INTER_AREA)
    return {"screen0": r, "screen1": o, "screen2": r, "usplane": sl}


def world_poses(sim, keys):
    cid = sim.client
    Ws = np.empty((len(keys), 4, 4))
    base = {}
    for i, (b, l) in enumerate(keys):
        if l == -1:
            if b not in base:
                pos, orn = p.getBasePositionAndOrientation(
                    b, physicsClientId=cid)
                base[b] = _quat_to_mat(pos, orn)
            Ws[i] = base[b]
        else:
            ls = p.getLinkState(b, l, computeForwardKinematics=True,
                                physicsClientId=cid)
            Ws[i] = _quat_to_mat(ls[4], ls[5])
    return Ws


# --------------------------------------------------------------- process
def _panel_main(conn, quality):
    """Child process: own PyBullet copy (geometry only) + renderer."""
    try:
        cfg.OR_QUALITY = quality
        from ultrabot_sim.scene import Scene
        sc = Scene(gui=False)
        r = ORRenderer(sc)
        keys = sorted({(b, l) for _, b, l, _ in r.nodes})
        idx = {k: i for i, k in enumerate(keys)}
        conn.send(("ready", keys, r._gl_info()))
    except Exception as e:                                  # noqa: BLE001
        import traceback
        conn.send(("error", f"{type(e).__name__}: {e}",
                   traceback.format_exc()))
        return
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            break
        if msg is None:
            break
        Ws, tops, acts, view, imgs, cam, size, anat = msg
        try:
            t = time.perf_counter()
            poses = [(node, Ws[idx[(b, l)]] @ local)
                     for node, b, l, local in r.nodes]
            if imgs:
                r.set_screen_images(imgs)
            img = r.draw({"poses": poses, "tops": tops, "acts": acts,
                          "view": view, "cam": cam, "size": size,
                          "anat": anat})
            conn.send(("img", img, (time.perf_counter() - t) * 1000))
        except Exception as e:                              # noqa: BLE001
            import traceback
            conn.send(("error", f"{type(e).__name__}: {e}",
                       traceback.format_exc()))
            break
    r.close()


class PanelProcess:
    """Draws the OR panel in another process (own Python, own GL context),
    so the console loop is never slowed by pyrender."""

    def __init__(self, scene, timeout=180.0):
        import multiprocessing as mp
        self.sim = scene
        self.error = None
        self.render_ms = 0.0
        self.gl_info = ""
        self._image = None
        self._frames = 0
        self._busy = False
        self._view = 0
        self._ft = 0.0
        self._cam = None
        self._size = None
        self._anat = "keep"
        ctx = mp.get_context("spawn")
        self._conn, child = ctx.Pipe()
        self._proc = ctx.Process(target=_panel_main,
                                 args=(child, cfg.OR_QUALITY),
                                 name="or-panel", daemon=True)
        self._proc.start()
        t0 = time.perf_counter()
        while not self._conn.poll(0.2):
            if not self._proc.is_alive():
                raise RuntimeError("panel process exited during start-up")
            if time.perf_counter() - t0 > timeout:
                self._proc.terminate()
                raise RuntimeError("panel process start-up timed out")
        msg = self._conn.recv()
        if msg[0] == "error":
            raise RuntimeError(f"{msg[1]}\n--- panel process traceback ---\n"
                               f"{msg[2]}")
        _, self._keys, self.gl_info = msg
        _warn(f"panel runs in its own process; OpenGL: {self.gl_info}")
        self.submit(None)
        t0 = time.perf_counter()
        while self._image is None and self.error is None and \
                time.perf_counter() - t0 < 60:
            self._poll(0.05)

    def _poll(self, wait=0.0):
        while self._conn.poll(wait):
            msg = self._conn.recv()
            if msg[0] == "img":
                self._image, self.render_ms = msg[1], msg[2]
                self._frames += 1
                self._busy = False
            elif msg[0] == "error":
                self.error = RuntimeError(msg[1])
                self._busy = False
            wait = 0.0

    @property
    def image(self):
        self._poll()
        return self._image

    @property
    def frames(self):
        self._poll()
        return self._frames

    @property
    def busy(self):
        self._poll()
        return self._busy

    def set_anatomy(self, an_dict):
        self._anat = an_dict

    def submit(self, frames, wait=False):
        """Send the current poses (never waits unless wait=True)."""
        self._poll()
        if self._busy or self.error is not None:
            if not wait:
                return
            t0 = time.perf_counter()
            while self._busy and time.perf_counter() - t0 < 30:
                self._poll(0.02)
        Ws = world_poses(self.sim, self._keys)
        tops = arm_tops(self.sim)
        now = time.perf_counter()
        imgs = None
        if frames is not None and (wait or now - self._ft >= 0.33):
            self._ft = now
            imgs = screen_images(*frames)
        acts = dict(getattr(self.sim, "_label_text", {}) or {})
        anat, self._anat = self._anat, "keep"
        try:
            self._conn.send((Ws, tops, acts, self._view, imgs, self._cam,
                             self._size, anat))
            self._busy = True
        except (OSError, EOFError) as e:
            self.error = e
        if wait:
            t0 = time.perf_counter()
            while self._busy and time.perf_counter() - t0 < 60:
                self._poll(0.02)

    def set_view(self, i):
        self._view = i

    def set_camera(self, cam):
        self._cam = cam

    def set_size(self, size):
        self._size = size

    def close(self):
        try:
            self._conn.send(None)
        except Exception:                                   # noqa: BLE001
            pass
        self._proc.join(timeout=3.0)
        if self._proc.is_alive():
            self._proc.terminate()


class InlinePanel:
    """Same interface, drawn in this process (fallback / tests)."""

    def __init__(self, scene):
        self.sim = scene
        self.r = ORRenderer(scene)
        self.error = None
        self.render_ms = 0.0
        self.gl_info = self.r._gl_info()
        self._image = None
        self._frames = 0
        self._view = 0
        self._cam = None
        self._size = None
        self._anat = "keep"

    image = property(lambda self: self._image)
    frames = property(lambda self: self._frames)
    busy = property(lambda self: False)

    def set_anatomy(self, an_dict):
        self._anat = an_dict

    def submit(self, frames, wait=False):
        t = time.perf_counter()
        if frames is not None:
            self.r.set_screen_images(screen_images(*frames))
        snap = self.r.snapshot()
        snap.update(view=self._view, cam=self._cam, size=self._size,
                    anat=self._anat)
        self._anat = "keep"
        self._image = self.r.draw(snap)
        self._frames += 1
        self.render_ms = (time.perf_counter() - t) * 1000

    def set_view(self, i):
        self._view = i

    def set_camera(self, cam):
        self._cam = cam

    def set_size(self, size):
        self._size = size

    def close(self):
        self.r.close()


class BulletPanel:
    """PyBullet's own (flat) renderer: last-resort fallback."""

    def __init__(self, scene):
        self.sim = scene
        self.error = None
        self.render_ms = 0.0
        self.gl_info = "PyBullet TinyRenderer (fallback)"
        self._image = None
        self._frames = 0
        self._view = 0

    image = property(lambda self: self._image)
    frames = property(lambda self: self._frames)
    busy = property(lambda self: False)

    def set_anatomy(self, an_dict):
        return None

    def submit(self, frames, wait=False):
        t = time.perf_counter()
        self._image = self.sim.render(cfg.OR_SIZE,
                                      cfg.OR_VIEWS[self._view %
                                                   len(cfg.OR_VIEWS)])
        self._frames += 1
        self.render_ms = (time.perf_counter() - t) * 1000

    def set_view(self, i):
        self._view = i

    def set_camera(self, cam):
        return None

    def set_size(self, size):
        return None

    def close(self):
        return None


def make_panel(scene, prefer=None):
    """Try the realistic panel in its own process, then inline, then the
    PyBullet renderer. Returns (panel, description)."""
    modes = prefer or ((cfg.OR_PANEL_MODES + ("bullet",))
                       if cfg.OR_RENDERER == "pyrender" else ("bullet",))
    errors = []
    for m in modes:
        try:
            if m == "process":
                return PanelProcess(scene), "pyrender (own process)"
            if m == "inline":
                return InlinePanel(scene), "pyrender (inline)"
            if m == "bullet":
                return BulletPanel(scene), "PyBullet renderer (fallback)"
        except Exception as e:                              # noqa: BLE001
            errors.append(f"{m}: {e}")
            _warn(f"{m} panel failed: {str(e).splitlines()[0]}")
    raise RuntimeError("no OR panel could start: " + "; ".join(errors))
