"""End-to-end: the real command-line program, headless."""
import json

from ultrabot_sim import config as cfg


def test_headless_demo_run(tmp_path):
    from ultrabot_sim.main import main
    rc = main(["--demo", "--auto", "--headless", "--once", "--tag",
               "pytest_demo"])
    s = json.loads((cfg.RUNS_DIR / "summary_pytest_demo.json").read_text())
    assert s["procedures"], "no procedure ran"
    p = s["procedures"][0]
    assert p["success"], p["criteria"]
    assert rc == 0
    assert (cfg.RUNS_DIR / "metrics_pytest_demo.csv").stat().st_size > 1000
    assert s["input"]["synthetic"] is True
