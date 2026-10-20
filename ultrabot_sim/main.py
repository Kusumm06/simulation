"""
UltraBot 3D robotic ultrasound-guided breast biopsy simulation.

    input -> perception (YOLO + tracker + LSTM, from ultrabot/) -> lesion in
    mm -> anatomy-aware planner (shortest / safest / recommended) -> two
    KUKA iiwa arms (probe + biopsy gun) -> surgeon console + 3D room

Run from the repository root:

    python -m ultrabot_sim.main --auto                    # demo on outputs/input.mp4
    python -m ultrabot_sim.main --compare                 # shortest then safest
    python -m ultrabot_sim.main --image my_scan.png --auto
    python -m ultrabot_sim.main --bundle outputs/sim_bundle.json --auto
    python -m ultrabot_sim.main --headless --once --compare --tag check

Kinematic research simulation, not a medical device. Not for clinical use.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2                                                  # noqa: E402
import numpy as np                                          # noqa: E402

from ultrabot_sim import config as cfg                      # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m ultrabot_sim.main",
        description="UltraBot 3D robotic breast biopsy simulation "
                    "(research simulation, not for clinical use)")
    src = ap.add_argument_group("input")
    src.add_argument("--video", help="ultrasound video (default "
                     "outputs/input.mp4)")
    src.add_argument("--image", help="a single ultrasound image (LSTM not "
                     "available)")
    src.add_argument("--frames", help="folder of ordered frames")
    src.add_argument("--fps", type=float, help="frame rate for --frames")
    src.add_argument("--bundle", help="plan bundle JSON from the dashboard")
    src.add_argument("--demo", action="store_true",
                     help="synthetic data (SYNTHETIC badge)")
    src.add_argument("--weights", help="YOLO weights (default models/best.pt)")
    src.add_argument("--lstm", help="LSTM checkpoint (default "
                     "models/lstm_checkpoint.pth)")
    src.add_argument("--conf", type=float, default=cfg.CONF,
                     help="YOLO confidence (default 0.10)")
    src.add_argument("--start-frame", type=int, default=None)
    src.add_argument("--max-frames", type=int, default=None)
    src.add_argument("--no-auto-window", action="store_true",
                     help="do not jump to the longest lesion track")
    run = ap.add_argument_group("procedure")
    run.add_argument("--auto", action="store_true",
                     help="plan and execute without key presses")
    run.add_argument("--compare", action="store_true",
                     help="shortest, then safest, on the same video window")
    run.add_argument("--path", choices=["shortest", "safest", "recommended"],
                     default="recommended")
    run.add_argument("--planner", choices=["anatomy", "legacy"],
                     default=cfg.DRIVE_PLANNER)
    run.add_argument("--speed", type=float, default=cfg.INSERT_SPEED_MM_S,
                     help="insertion speed mm/s (1..10)")
    run.add_argument("--weights-preset", choices=list(cfg.WEIGHT_PRESETS),
                     default=cfg.WEIGHT_PRESET)
    run.add_argument("--depth-mm", type=float, default=cfg.US_DEPTH_MM,
                     help="scanner imaging depth (scale of the image)")
    run.add_argument("--width-mm", type=float, default=cfg.US_WIDTH_MM,
                     help="probe footprint width")
    run.add_argument("--auto-chestwall", action="store_true",
                     help="estimate the pectoral fascia from the image")
    run.add_argument("--seed", type=int, default=cfg.SEED)
    run.add_argument("--drop", type=float, default=0.0,
                     help="drop detections with this probability")
    run.add_argument("--jitter", type=float, default=0.0,
                     help="add Gaussian noise (px) to detections")
    out = ap.add_argument_group("output")
    out.add_argument("--headless", action="store_true", help="no windows")
    out.add_argument("--record", help="save the console as a video")
    out.add_argument("--seconds", type=float, default=None)
    out.add_argument("--once", action="store_true",
                     help="stop when the procedure(s) finish or the video ends")
    out.add_argument("--tag", default=None)
    out.add_argument("--or-window", action="store_true")
    out.add_argument("--quality", choices=["low", "medium", "high"],
                     default=cfg.OR_QUALITY)
    out.add_argument("--or-every", type=int, default=cfg.OR_RENDER_EVERY,
                     help="control ticks between 3D panel updates")
    out.add_argument("--no-or", action="store_true",
                     help="no 3D operating-room panel")
    out.add_argument("--bullet-window", action="store_true",
                     help="also open PyBullet's own window")
    out.add_argument("--voice", action="store_true")
    out.add_argument("--fast", action="store_true",
                     help="do not wait for real time (headless default)")
    return ap.parse_args(argv)


def apply_config(args):
    cfg.US_DEPTH_MM = float(args.depth_mm)
    cfg.US_WIDTH_MM = float(args.width_mm)
    cfg.CONF = float(args.conf)
    cfg.OR_QUALITY = args.quality
    cfg.DRIVE_PLANNER = args.planner
    cfg.INSERT_SPEED_MM_S = float(np.clip(args.speed, 1.0, 10.0))
    cfg.WEIGHT_PRESET = args.weights_preset
    cfg.AUTO_CHESTWALL = bool(args.auto_chestwall)
    cfg.SEED = int(args.seed)
    cfg.OR_RENDER_EVERY = max(1, int(args.or_every))
    np.random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
    except Exception:                                       # noqa: BLE001
        pass


def load_lstm_only(path=None):
    """LSTM checkpoint without YOLO (bundle / demo replays)."""
    import torch
    from ultrabot.pipeline import LSTMModel, get_lstm_checkpoint_path
    p = get_lstm_checkpoint_path(path)
    if not p.exists():
        return None
    ck = torch.load(p, map_location="cpu", weights_only=False)
    m = LSTMModel()
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    return m


class Runner:
    def __init__(self, args):
        from ultrabot_sim.inputs import make_source
        from ultrabot_sim.us_geometry import USCalib
        self.args = args
        self.headless = args.headless
        self.calib = USCalib.from_config()
        self.tag = args.tag or time.strftime("run_%Y%m%d_%H%M%S")
        print(f"[ultrabot-sim] {cfg.DISCLAIMER}")
        self.source = make_source(args)
        self.models = None
        self.model_lstm = None
        if self.source.mode in ("video", "frames", "image"):
            from ultrabot.pipeline import load_models
            print("[ultrabot-sim] loading YOLO + LSTM from models/ ...")
            self.models = load_models(args.weights, args.lstm)
            self.model_lstm = self.models[1]
        else:
            self.model_lstm = load_lstm_only(args.lstm)
        print(f"[ultrabot-sim] input: {self.source.mode} - "
              f"{self.source.label} @ {self.source.fps:.1f} fps")
        self.window = None
        self._choose_window()
        from ultrabot_sim.scene import Scene
        self.scene = Scene(gui=args.bullet_window)
        if not self.scene.home_ok:
            print("[ultrabot-sim] WARNING: a home pose was not reached "
                  "exactly; check the cart positions in config.py")
        from ultrabot_sim.metrics import MetricsLogger
        self.logger = MetricsLogger(self.tag, self.calib)
        self.perf = None
        if cfg.PERF_LOG:
            from ultrabot_sim.perfmon import PerfMonitor
            self.perf = PerfMonitor(self.tag)
        self.panel = None
        self.panel_desc = "off"
        self.or_cam = None
        self.or_window = None
        self.console = None
        self.writer = None
        self.show_anatomy = True
        self.show_legacy = False
        self.last_cmd = ""
        self.compare_finals = []
        self.quit = False
        self.awaiting_confirm = False
        self.voice = None
        from ultrabot_sim.commands import CommandBus
        self.bus = CommandBus()

    # ------------------------------------------------------------ window
    def _choose_window(self):
        a = self.args
        want = (a.auto or a.compare) and a.start_frame is None and \
            not a.no_auto_window
        if not want or self.source.mode in ("image", "frames"):
            return
        from ultrabot_sim.inputs import best_window, scan_video
        if self.source.mode == "video":
            data = scan_video(self.source.path, self.models, cfg.CONF)
            rows = data["rows"]
        else:
            rows = [{"index": int(r["index"]), "detected": bool(r["detected"]),
                     "track_id": r.get("track_id"),
                     "lstm_ready": bool(r.get("lstm_ready"))}
                    for r in self.source.records]
        w = best_window(rows)
        if w is None:
            print("[ultrabot-sim] no lesion track found; playing from the "
                  "start")
            return
        start, end, tid, hits = w
        first = rows[0]["index"] if rows else 1
        s0 = max(first, start - 3)
        self.window = (s0, end, tid, hits)
        print(f"[ultrabot-sim] longest lesion track: frames {start}-{end} "
              f"({hits} detections, track {tid}); starting at frame {s0}")
        self.source.rewind(s0)

    # ------------------------------------------------------------ setup UI
    def setup_ui(self):
        from ultrabot_sim.display import Console
        self.console = Console(self.calib)
        need_panel = (not self.args.no_or) and (not self.headless or
                                                self.args.record)
        if need_panel:
            from ultrabot_sim.or_render import make_panel
            from ultrabot_sim.orcam import OrbitCamera
            print("[ultrabot-sim] starting the 3D operating-room panel ...")
            try:
                self.panel, self.panel_desc = make_panel(self.scene)
            except Exception as e:                          # noqa: BLE001
                print(f"[ultrabot-sim] no 3D panel: {e}")
                self.panel = None
            self.or_cam = OrbitCamera(0)
        if not self.headless:
            cv2.namedWindow(cfg.WINDOW, cv2.WINDOW_NORMAL)
            W, H = cfg.CONSOLE_SIZE
            k = min(cfg.MAX_DISPLAY_WIDTH / W, cfg.MAX_DISPLAY_HEIGHT / H, 1.0)
            cv2.resizeWindow(cfg.WINDOW, int(W * k), int(H * k))
            if self.args.or_window:
                self._toggle_or_window()
            if self.args.voice:
                from ultrabot_sim.voice import VoiceListener
                self.voice = VoiceListener(self.bus)
                self.voice.start()
        if self.args.record:
            Path(self.args.record).parent.mkdir(parents=True, exist_ok=True)
            self.writer = cv2.VideoWriter(
                self.args.record, cv2.VideoWriter_fourcc(*"mp4v"),
                cfg.CONTROL_HZ, cfg.CONSOLE_SIZE)

    def _toggle_or_window(self):
        from ultrabot_sim.or_viewer import ORWindow3D
        if self.or_window is None:
            self.or_window = ORWindow3D()
        self.or_window.toggle(self.or_cam.preset if self.or_cam else 0)

    # ------------------------------------------------------------ run
    def run(self):
        self.setup_ui()
        a = self.args
        queue = ["shortest", "safest"] if a.compare else [a.path]
        auto = a.auto or a.compare
        compare = a.compare
        results = []
        ep_start = getattr(self.source, "start_frame", 1)
        plan_summaries = []
        self.pending_compare = False
        first = True
        try:
            while queue and not self.quit:
                kind = queue.pop(0)
                if not first:
                    print(f"[ultrabot-sim] replaying the same window "
                          f"(frame {ep_start}) for the {kind} path")
                    self.source.rewind(ep_start)
                first = False
                hint = None
                if plan_summaries:
                    nxt = plan_summaries[-1].get(kind)
                    hint = nxt.get("side") if nxt else None
                proc = self.episode(kind, auto, compare, hint)
                results += proc.results
                if proc.plan_set is not None:
                    plan_summaries.append(proc.plan_set.summary())
                for r in proc.results:
                    if r.fire:
                        self.compare_finals.append(r)
                if self.pending_compare:
                    self.pending_compare = False
                    queue = ["shortest", "safest"]
                    auto = compare = True
        finally:
            self.finish(results, plan_summaries)
        return results

    def episode(self, kind, auto, compare, side_hint=None):
        from ultrabot_sim.control import Procedure
        from ultrabot_sim.lesion_model import LesionTracker
        from ultrabot_sim.planner import Planner
        a = self.args
        per = self.source.make_perception(
            models=self.models, conf=cfg.CONF, drop=a.drop, jitter=a.jitter,
            seed=a.seed) if self.source.mode in ("video", "frames", "image") \
            else self.source.make_perception()
        lt = LesionTracker(self.calib, self.model_lstm)
        log = self._log_fn()
        proc = Procedure(self.scene, self.calib, lt, Planner(self.calib),
                         kind=kind, auto=auto,
                         image_mode=self.source.mode == "image", log=log,
                         frame_dt=self.source.frame_dt,
                         preset=cfg.WEIGHT_PRESET,
                         auto_chestwall=cfg.AUTO_CHESTWALL, seed=a.seed,
                         side_hint=side_hint)
        proc.speed = cfg.INSERT_SPEED_MM_S
        self.proc, self.per, self.lt = proc, per, lt
        dt = 1.0 / cfg.CONTROL_HZ
        t = 0.0
        next_frame_t = 0.0
        obs = None
        video_ended = False
        tick = 0
        last_anat_id = None
        wall0 = time.perf_counter()
        loop_t = time.perf_counter()
        loop_fps = 0.0
        realtime = not (self.headless or a.fast) and cfg.VIDEO_REALTIME
        while not self.quit and not self.pending_compare:
            new_frame = False
            if proc.state == "IDLE":
                next_frame_t = t          # imaging starts at skin contact
            elif t >= next_frame_t - 1e-9:
                next_frame_t += self.source.frame_dt
                item = None if video_ended else self.source.next()
                if item is not None:
                    idx, pane = item
                    obs = per.step(pane, idx, t)
                    lt.update(obs)
                    self.logger.count_frame(obs)
                    new_frame = True
                elif self.source.mode != "image" and obs is not None:
                    if not video_ended:
                        log(t, "input", "end of the video")
                    video_ended = True
                    # no more images: the lesion is no longer observed
                    from ultrabot_sim.perception import FrameObs
                    obs = FrameObs(index=obs.index + 1, t=t,
                                   image_bgr=obs.image_bgr, source="ended",
                                   lstm_status="video ended")
                    lt.update(obs)
                    new_frame = True
            for cmd, srcn in self.bus.drain():
                self.handle(cmd, proc, srcn)
            proc.tick(dt, new_frame, obs)
            self.logger.tick(proc, obs)
            # end conditions
            if proc.state == "DONE" and proc.results and (auto or a.once):
                if len(proc.results) >= 1 and (compare or auto):
                    self._render(proc, obs, loop_fps, final=True)
                    break
            if a.seconds and t >= a.seconds:
                log(t, "input", f"--seconds {a.seconds} reached")
                self.quit = True
            if video_ended and proc.state in ("SCAN", "DONE", "PLAN") and \
                    (a.once or auto):
                break
            if self.source.mode == "image" and a.once and \
                    proc.state in ("DONE",):
                break
            # rendering / UI
            if self.console is not None and (not self.headless or
                                             self.writer is not None):
                ps = proc.plan_set
                an_id = id(ps.anatomy) if ps is not None else None
                if self.panel is not None and (tick % cfg.OR_RENDER_EVERY
                                               == 0):
                    if an_id != last_anat_id and proc.anatomy is not None:
                        self.panel.set_anatomy(proc.anatomy.to_dict())
                        last_anat_id = an_id
                    self.panel.set_view(self.or_cam.preset)
                    frames = (getattr(self, "_raw_last", None),
                              getattr(self, "_ov_last", None))
                    self.panel.submit(frames if frames[0] is not None else
                                      None, wait=bool(self.writer) and
                                      self.headless)
                if self.or_window is not None and self.or_window.open:
                    self.or_window.push(self.scene,
                                        (self._raw_last, self._ov_last)
                                        if getattr(self, "_raw_last", None)
                                        is not None else None)
                if tick % cfg.CONSOLE_EVERY == 0 or \
                        getattr(self, "_canvas", None) is None:
                    self._render(proc, obs, loop_fps)
                elif self.writer is not None:
                    self.writer.write(self._canvas)      # keep video timing
                elif not self.headless:
                    self._poll_keys()
            if self.perf is not None:
                self.perf.tick(obs.index if obs else 0,
                               obs.infer_ms if obs and new_frame else 0.0,
                               self.panel.frames if self.panel else 0,
                               self.panel.render_ms if self.panel else 0.0,
                               proc.plan_set.plan_ms if proc.plan_set and
                               new_frame else 0.0)
            tick += 1
            t += dt
            now = time.perf_counter()
            loop_fps = 0.9 * loop_fps + 0.1 / max(now - loop_t, 1e-6)
            loop_t = now
            if realtime:
                lag = (wall0 + t) - time.perf_counter()
                if lag > 0:
                    time.sleep(lag)
        return proc

    def _log_fn(self):
        def log(t, kind, msg):
            print(f"[{t:7.2f}] {kind:6s} {msg}")
            self.logger.event(t, kind, msg)
        return log

    # ------------------------------------------------------------ render
    def _render(self, proc, obs, loop_fps, final=False):
        from ultrabot_sim.us_render import render_overlay, render_raw
        if obs is None or obs.image_bgr is None:
            pane = np.zeros((cfg.PANE_H_PX, cfg.PANE_W_PX, 3), np.uint8)
        else:
            pane = obs.image_bgr
        raw = render_raw(pane, self.calib)
        ov = render_overlay(pane, obs, proc.lt, proc, self.calib,
                            self.show_anatomy, self.show_legacy)
        if self.awaiting_confirm:
            cv2.rectangle(ov, (200, 300), (790, 400), (20, 20, 30), -1)
            cv2.putText(ov, f"Start the {proc.kind} path?  y / n", (240, 362),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (60, 220, 255), 2,
                        cv2.LINE_AA)
        self._raw_last, self._ov_last = raw, ov
        badges = [(self.source.mode.upper(), (230, 190, 60))]
        if self.source.synthetic:
            badges.append(("SYNTHETIC", (60, 140, 255)))
        if cfg.DRIVE_PLANNER == "legacy":
            badges.append(("LEGACY PLANNER", (160, 160, 160)))
        if proc.frozen:
            badges.append(("TARGET FROZEN", (200, 120, 240)))
        if self.source.mode == "image":
            badges.append(("SINGLE IMAGE: NO LSTM", (60, 200, 255)))
        info = {
            "badges": badges,
            "source": f"input: {self.source.label}   frame "
                      f"{obs.index if obs else '-'}   window "
                      f"{self.window[0]}-{self.window[1] if self.window else ''}"
                      if self.window else
                      f"input: {self.source.label}   frame "
                      f"{obs.index if obs else '-'}",
            "loop_fps": loop_fps,
            "yolo_ms": obs.infer_ms if obs else 0.0,
            "plan_ms": proc.plan_set.plan_ms if proc.plan_set else 0.0,
            "or_view": (f"{self.or_cam.name}  |  {self.panel_desc}"
                        if self.or_cam else ""),
            "or_status": "3D panel off (--no-or)" if self.args.no_or else
            "3D panel starting...",
            "last_cmd": self.last_cmd, "lt": proc.lt,
            "finals": self.compare_finals,
        }
        img = self.panel.image if self.panel is not None else None
        canvas = self.console.compose(raw, ov, img, proc, obs, info)
        if self.writer is not None:
            self.writer.write(canvas)
        self._canvas = canvas
        if not self.headless:
            cv2.imshow(cfg.WINDOW, canvas)
            self._poll_keys()

    def _poll_keys(self):
        key = cv2.waitKeyEx(1)
        from ultrabot_sim.commands import key_command
        cmd = key_command(key)
        if cmd:
            self.bus.push(cmd, "key")
        try:
            if cv2.getWindowProperty(cfg.WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                self.quit = True
        except cv2.error:
            self.quit = True

    # ------------------------------------------------------------ commands
    def handle(self, cmd, proc, source="key"):
        self.last_cmd = f"{cmd.lower()} ({source})"
        if self.awaiting_confirm:
            if cmd == "CONFIRM_YES":
                self.awaiting_confirm = False
                if proc.state == "DONE":
                    proc.next_run(proc.kind)
                proc.request_start()
                return
            if cmd in ("CONFIRM_NO", "QUIT"):
                self.awaiting_confirm = False
                proc.log("cmd", "start cancelled")
                if cmd == "CONFIRM_NO":
                    return
        if cmd == "QUIT":
            self.quit = True
        elif cmd == "PATH_SHORTEST":
            proc.select("shortest")
        elif cmd == "PATH_SAFEST":
            proc.select("safest")
        elif cmd == "PATH_RECOMMENDED":
            proc.select("recommended")
        elif cmd == "PLAN_FREEZE":
            proc.toggle_freeze()
        elif cmd == "START":
            if proc.state in ("SCAN", "PLAN", "DONE"):
                self.awaiting_confirm = True
            else:
                proc.log("cmd", f"start refused in {proc.state}")
        elif cmd == "COMPARE":
            if proc.state in ("IDLE", "SCAN", "PLAN", "DONE"):
                proc.log("cmd", "compare mode: shortest, reset, safest on the "
                         "same video window")
                self.pending_compare = True
            else:
                proc.log("cmd", f"compare refused in {proc.state}")
        elif cmd == "PAUSE":
            proc.toggle_pause()
        elif cmd == "E_STOP":
            proc.e_stop()
        elif cmd == "RESUME":
            proc.resume()
        elif cmd == "SPEED_UP":
            proc.change_speed(+1)
        elif cmd == "SPEED_DOWN":
            proc.change_speed(-1)
        elif cmd == "WEIGHTS_NEXT":
            proc.cycle_preset(+1)
        elif cmd == "WEIGHTS_PREV":
            proc.cycle_preset(-1)
        elif cmd == "LEGACY":
            self.show_legacy = not self.show_legacy
        elif cmd == "ANATOMY":
            self.show_anatomy = not self.show_anatomy
        elif cmd == "SLOWMO":
            proc.slowmo = not proc.slowmo
            proc.log("cmd", f"slow-motion throw {'on' if proc.slowmo else 'off'}")
        elif cmd in ("VIEW_NEXT", "VIEW_PREV") and self.or_cam is not None:
            name = self.or_cam.next(1 if cmd == "VIEW_NEXT" else -1)
            proc.log("cmd", f"OR view: {name}")
        elif cmd == "OR_WINDOW":
            if self.or_cam is not None:
                self._toggle_or_window()
        elif cmd == "SCREENSHOT":
            cfg.ensure_runs_dir()
            p = cfg.RUNS_DIR / f"shot_{time.strftime('%H%M%S')}_{int(proc.t*10)}.png"
            if getattr(self, "_canvas", None) is not None:
                cv2.imwrite(str(p), self._canvas)
                proc.log("cmd", f"screenshot {p.name}")
        elif cmd == "MARK":
            self.logger.mark(proc.t)
        elif cmd in ("PROBE_LEFT", "PROBE_RIGHT"):
            self._nudge_probe(proc, -1.0 if cmd == "PROBE_LEFT" else 1.0)

    def _nudge_probe(self, proc, mm):
        if proc.state not in ("IDLE", "SCAN", "PLAN") or proc.locked:
            proc.log("cmd", "probe locked: no probe moves after the approach "
                     "has started")
            return
        pose = self.scene.probe_pose()
        pose.t = pose.t + pose.R[:, 0] * (mm / 1000.0)
        q, e, a, m, ok = self.scene.ik_probe_robust(pose, self.scene.q["probe"])
        if ok and self.scene.collision_free("probe", q):
            self.scene.set_q("probe", q)
            proc.log("cmd", f"probe slid {mm:+.0f} mm along x (the recorded "
                     "image does not change; only its registration)")
        else:
            proc.log("cmd", "probe move refused (IK / collision)")

    # ------------------------------------------------------------ finish
    def finish(self, results, plan_summaries):
        from ultrabot_sim.metrics import success_table
        lt = getattr(self, "lt", None)
        extra = {"input": {"mode": self.source.mode,
                           "label": self.source.label,
                           "synthetic": self.source.synthetic,
                           "fps": self.source.fps, "window": self.window},
                 "planner": cfg.DRIVE_PLANNER,
                 "lstm_one_step_error": lt.error_stats() if lt else {},
                 "lstm_errors_mm": [round(e, 4) for e in lt.errors_mm]
                 if lt else [],
                 "plan_sets_last": plan_summaries,
                 "or_panel": self.panel_desc,
                 "robustness": {"drop": self.args.drop,
                                "jitter_px": self.args.jitter}}
        try:
            if self.writer is not None:
                self.writer.release()
            if self.panel is not None:
                self.panel.close()
            if self.or_window is not None:
                self.or_window.stop()
            if self.voice is not None:
                self.voice.stop()
            if not self.headless:
                cv2.destroyAllWindows()
        except Exception:                                   # noqa: BLE001
            pass
        summary = self.logger.close(results, extra)
        print()
        print("=" * 72)
        print(" UltraBot 3D biopsy simulation - results")
        print("=" * 72)
        if not results:
            print("  no procedure was completed in this run")
        for line in success_table(results):
            print(line)
        if len(results) >= 2:
            print(compare_table(results))
        print(f"\n  logs: {self.logger.path_summary.name}, "
              f"{self.logger.path_ticks.name}, {self.logger.path_events.name}"
              f" in {cfg.RUNS_DIR}")
        if self.perf is not None:
            self.perf.report()
        self.source.close()
        self.scene.disconnect()
        return summary


def compare_table(results):
    rows = [("outcome", lambda r: r.outcome),
            ("insertion (mm)", lambda r: f"{r.plan.get('insertion_mm', 0):.1f}"),
            ("angle to chest wall", lambda r: f"{r.plan.get('angle_chestwall_deg', 0):.1f}"),
            ("planned min clearance", lambda r: f"{r.plan.get('min_clearance_mm', 0):.1f} "
             f"({r.plan.get('min_clearance_struct')})"),
            ("measured min clearance", lambda r: f"{r.min_needle_clearance_mm:.1f}"
             if math.isfinite(r.min_needle_clearance_mm) else "-"),
            ("ligaments crossed", lambda r: str(r.plan.get("ligaments_crossed"))),
            ("notch in lesion", lambda r: f"{100 * r.fire.get('notch_in_lesion_frac', 0):.0f} %"
             if r.fire else "-"),
            ("max IK error (mm)", lambda r: f"{r.max_ik_err_mm:.4f}"),
            ("max skin RCM (mm)", lambda r: f"{r.max_rcm_dev_mm:.4f}"),
            ("procedure time (s)", lambda r: f"{r.t_end - r.t_start:.1f}"),
            ("success", lambda r: "YES" if r.success else "NO")]
    head = "\n  COMPARISON".ljust(26) + "".join(f"{r.kind:>16s}" for r in results)
    out = [head]
    for name, f in rows:
        out.append(f"  {name:22s}" + "".join(f"{f(r):>16s}" for r in results))
    return "\n".join(out)


def main(argv=None):
    args = parse_args(argv)
    if args.headless:
        args.fast = True
    apply_config(args)
    r = Runner(args)
    results = r.run()
    return 0 if results and all(x.success for x in results) else 1


if __name__ == "__main__":
    sys.exit(main())
