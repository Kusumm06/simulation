"""Shared fixtures. Heavy objects (scene, models) are built once."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))



@pytest.fixture(scope="session")
def calib():
    from ultrabot_sim.us_geometry import USCalib
    return USCalib.from_config()


@pytest.fixture(scope="session")
def scene():
    from ultrabot_sim.scene import Scene
    sc = Scene()
    sc.set_q("probe", sc.ready_q["probe"])
    yield sc
    sc.disconnect()


@pytest.fixture(scope="session")
def models():
    from ultrabot.pipeline import load_models
    return load_models()


@pytest.fixture
def lesion_state(calib):
    """A lesion state like the tracker produces (9 x 5 mm half-axes)."""
    return make_state(calib, 0.0, 12.0)


def make_state(calib, x, z, a=9.0, b=5.0, track=1):
    from ultrabot_sim.anatomy import LesionGeom
    from ultrabot_sim.lesion_model import LesionState
    g = LesionGeom(x, z, a, b)
    u, v = calib.mm_to_px(x, z)
    return LesionState(frame=1, track_id=track, center_px=(float(u), float(v)),
                       center_mm=(x, z), geom=g, class_name="benign", conf=0.9,
                       lstm_ready=False)
