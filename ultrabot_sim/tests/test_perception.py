"""Perception wraps ultrabot.pipeline faithfully; the LSTM is never faked."""
import cv2
import numpy as np
import pytest

from ultrabot_sim import config as cfg
from ultrabot_sim.inputs import ImageSource, VideoSource
from ultrabot_sim.perception import Perception

N = 80


@pytest.fixture(scope="module")
def video_obs(models):
    src = VideoSource(cfg.DEFAULT_VIDEO, max_frames=N)
    per = Perception(models=models, mode="video")
    out = []
    while True:
        it = src.next()
        if it is None:
            break
        out.append(per.step(it[1], it[0]))
    return out


def test_matches_run_pipeline(models, video_obs, tmp_path):
    """Frame-by-frame perception == the dashboard's run_pipeline()."""
    from ultrabot.pipeline import run_pipeline
    res = run_pipeline(cfg.DEFAULT_VIDEO, output_path=tmp_path / "o.mp4",
                       max_frames=N, models=models)
    assert len(res.records) == len(video_obs)
    for rec, o in zip(res.records, video_obs):
        assert rec["detected"] == o.detected
        assert rec["lstm_ready"] == o.lstm_ready
        if rec["detected"]:
            assert np.allclose(rec["raw_center"], o.center_px, atol=1e-3)
        if rec["pred_center"] is not None:
            assert np.allclose(rec["pred_center"], o.pred_center_px)


def test_lstm_needs_20_consecutive_hits(video_obs):
    assert any(o.detected for o in video_obs)
    for o in video_obs:
        if o.lstm_ready:
            assert o.consecutive_hits >= 20
            assert o.lstm_window is not None and len(o.lstm_window) == 20
        else:
            assert o.pred_center_px is None


def test_single_image_never_lstm(models, tmp_path):
    cap = cv2.VideoCapture(str(cfg.DEFAULT_VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 800)
    ok, f = cap.read()
    p = tmp_path / "one.png"
    cv2.imwrite(str(p), f)
    src = ImageSource(p)
    per = Perception(models=models, mode="image")
    idx, pane = src.next()
    assert src.next() is None                  # one image, no fake motion
    # even if the same image were fed again and again, never LSTM-ready
    for k in range(30):
        o = per.step(pane, k + 1)
        assert not o.lstm_ready and o.pred_center_px is None
        assert "not available" in o.lstm_status


def test_record_replay_and_rollout(calib, models, video_obs):
    from ultrabot_sim.lesion_model import LesionTracker
    lt = LesionTracker(calib, models[1])
    for o in video_obs:
        lt.update(o)
    if not any(o.lstm_ready for o in video_obs):
        pytest.skip("no LSTM-ready frame in the first frames")
    ro = lt.rollout(cfg.LSTM_MAX_ROLLOUT + 5)
    if lt.state.lstm_ready:
        assert ro.k == cfg.LSTM_MAX_ROLLOUT
        assert np.all(np.diff(ro.radii_mm) > 0)       # uncertainty grows
