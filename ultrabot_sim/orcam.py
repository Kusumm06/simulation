"""
Camera presets for the 3D operating-room views (cfg.OR_VIEWS):
(distance, yaw, pitch, target), like PyBullet's debug camera. v / b in the
console cycle the presets; the big 3D window (or_viewer.py) has its own
mouse orbit / pan / zoom.
"""
from ultrabot_sim import config as cfg


class OrbitCamera:
    def __init__(self, preset=0):
        self.apply_preset(preset)

    @property
    def name(self):
        return cfg.OR_VIEWS[self.preset % len(cfg.OR_VIEWS)][0]

    def apply_preset(self, i):
        self.preset = i % len(cfg.OR_VIEWS)
        _, d, y, pch, t = cfg.OR_VIEWS[self.preset]
        self.dist, self.yaw, self.pitch = float(d), float(y), float(pch)
        self.target = list(map(float, t))

    def next(self, step=1):
        self.apply_preset(self.preset + step)
        return self.name

    def spec(self):
        return (self.dist, self.yaw, self.pitch, list(self.target))
