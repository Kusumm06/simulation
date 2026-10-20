"""Procedure state machine and safety layer (synthetic, stable lesion)."""
import numpy as np
import pytest

from ultrabot_sim import config as cfg
from ultrabot_sim.anatomy import Vessel
from ultrabot_sim.control import Procedure
from ultrabot_sim.inputs import demo_source
from ultrabot_sim.lesion_model import LesionTracker
from ultrabot_sim.perception import FrameObs
from ultrabot_sim.planner import Planner

DT = 1.0 / cfg.CONTROL_HZ


@pytest.fixture(scope="module")
def demo():
    return demo_source()


class Driver:
    def __init__(self, scene, calib, demo, kind="recommended"):
        from ultrabot_sim.main import load_lstm_only
        scene.set_q("needle", scene.home_q["needle"])
        scene.set_q("probe", scene.ready_q["probe"])
        scene.needle.reset()
        demo.rewind(1)
        self.src = demo
        self.per = demo.make_perception()
        self.lt = LesionTracker(calib, load_lstm_only())
        self.proc = Procedure(scene, calib, self.lt, Planner(calib),
                              kind=kind, auto=True, log=lambda *a: None,
                              frame_dt=demo.frame_dt)
        self.t = 0.0
        self.next_t = 0.0
        self.obs = None
        self.blind = False

    def step(self):
        new = False
        if self.proc.state == "IDLE":
            self.next_t = self.t
        elif self.t >= self.next_t - 1e-9:
            self.next_t += self.src.frame_dt
            it = self.src.next()
            if it is not None:
                o = self.per.step(it[1], it[0], self.t)
                if self.blind:            # lesion disappears from the image
                    o = FrameObs(index=o.index, t=self.t,
                                 image_bgr=o.image_bgr)
                self.obs = o
                self.lt.update(o)
                new = True
        self.proc.tick(DT, new, self.obs)
        self.t += DT

    def run_until(self, states, max_t=120.0):
        states = (states,) if isinstance(states, str) else states
        while self.t < max_t:
            self.step()
            if self.proc.state in states:
                return True
        return False


def test_full_synthetic_procedure_succeeds(scene, calib, demo):
    d = Driver(scene, calib, demo)
    assert d.run_until("DONE", 150)
    r = d.proc.results[-1]
    assert r.success, r.to_dict()["criteria"]
    assert r.max_ik_err_mm < 0.1 and r.max_rcm_dev_mm < 0.5


def test_abort_when_lesion_lost(scene, calib, demo):
    d = Driver(scene, calib, demo)
    assert d.run_until("INSERT", 120)
    d.blind = True
    assert d.run_until("ABORT_RETRACT", d.t + 5.0)
    frames = d.lt.frames_without_lesion
    assert frames == cfg.LOST_ABORT_FRAMES + 1
    assert d.run_until("DONE", d.t + 60)
    assert d.proc.results[-1].outcome == "ABORTED"


def test_abort_on_predicted_miss(scene, calib, demo):
    d = Driver(scene, calib, demo)
    assert d.run_until("INSERT", 120)
    p = d.proc.locked_plan
    n = np.array([-p.direction[1], p.direction[0]])
    orig = d.lt.update

    def moved(obs):                     # the lesion jumps 12 mm sideways
        st = orig(obs)
        if st is not None:
            g = st.geom
            g.x += n[0] * 12
            g.z += n[1] * 12
            st.center_mm = (g.x, g.z)
            st.window = None            # no LSTM extrapolation of the jump
        return st
    d.lt.update = moved
    assert d.run_until("ABORT_RETRACT", d.t + 4.0)
    assert "predicted throw miss" in d.proc.result.abort_reason


def test_abort_on_clearance(scene, calib, demo):
    d = Driver(scene, calib, demo)
    assert d.run_until("INSERT", 120)
    d.run_until("PRE_FIRE_CHECK", d.t + 0.5)
    p = d.proc.locked_plan
    pt = np.array(p.entry_mm) + np.array(p.direction) * 2.0
    d.proc.anatomy.vessels.append(Vessel(float(pt[0]), float(pt[1]), 0.8))
    d.step()
    d.step()
    assert d.proc.state == "ABORT_RETRACT"


def test_estop_freezes_within_one_tick(scene, calib, demo):
    d = Driver(scene, calib, demo)
    assert d.run_until("INSERT", 120)
    d.step()
    d.proc.e_stop()
    q0 = list(scene.q["needle"])
    s0 = d.proc.s
    for _ in range(20):
        d.step()
    assert d.proc.state == "E_STOP"
    assert np.allclose(scene.q["needle"], q0) and d.proc.s == s0
    d.proc.resume()                     # needle in tissue -> retract only
    assert d.proc.state == "ABORT_RETRACT"
    s1 = d.proc.s
    for _ in range(30):
        d.step()
    assert d.proc.s < s1


def test_safety_never_pushes_deeper(scene, calib, demo):
    d = Driver(scene, calib, demo)
    assert d.run_until("INSERT", 120)
    for _ in range(10):
        d.step()
    from ultrabot_sim.safety import SafetyReport
    d.proc.report = SafetyReport(level="WARN", messages=["test"])
    before = max(d.proc.s, d.proc.s_target)
    st = d.lt.state
    p = d.proc.locked_plan
    st.geom.x += p.direction[0] * 8                # lesion moves deeper
    st.geom.z += p.direction[1] * 8
    st.center_mm = (st.geom.x, st.geom.z)
    d.proc._update_depth_target()
    assert d.proc.s_target <= before + 1e-9
