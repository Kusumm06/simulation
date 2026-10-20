"""
Command bus: keyboard and (optional) voice both push commands here; the
main loop executes them. One place defines what each command does.

Safety note: like the reference project, a spoken or typed command is an
INPUT to the controller, never a bypass of the safety layer (e.g. "start"
is refused without a feasible plan; nothing can push the needle deeper
than the safety layer allows).
"""
import threading
import time

from ultrabot_sim import config as cfg

# key (cv2.waitKeyEx code or character) -> command
KEYMAP = {
    ord("1"): "PATH_SHORTEST", ord("2"): "PATH_SAFEST",
    ord("3"): "PATH_RECOMMENDED", ord("p"): "PLAN_FREEZE",
    13: "START", 10: "START", ord("c"): "COMPARE", ord(" "): "PAUSE",
    ord("z"): "E_STOP", ord("r"): "RESUME", ord("+"): "SPEED_UP",
    ord("="): "SPEED_UP", ord("-"): "SPEED_DOWN", ord("["): "WEIGHTS_PREV",
    ord("]"): "WEIGHTS_NEXT", ord("L"): "LEGACY", ord("l"): "LEGACY",
    ord("a"): "ANATOMY", ord("t"): "SLOWMO", ord("v"): "VIEW_NEXT",
    ord("b"): "VIEW_PREV", ord("o"): "OR_WINDOW", ord("s"): "SCREENSHOT",
    ord("m"): "MARK", ord("q"): "QUIT", 27: "QUIT", ord("y"): "CONFIRM_YES",
    ord("n"): "CONFIRM_NO", ord("j"): "PROBE_LEFT", ord("k"): "PROBE_RIGHT",
}

DESCRIPTIONS = {
    "PATH_SHORTEST": "select the shortest path",
    "PATH_SAFEST": "select the safest path",
    "PATH_RECOMMENDED": "select the recommended path",
    "PLAN_FREEZE": "freeze / unfreeze the planning target",
    "START": "start the procedure (asks y/n)",
    "COMPARE": "compare: shortest, reset, safest on the same window",
    "PAUSE": "pause / resume",
    "E_STOP": "E-stop: both arms freeze",
    "RESUME": "resume after E-stop (needle retracts if it was inside)",
    "SPEED_UP": "insertion speed +1 mm/s",
    "SPEED_DOWN": "insertion speed -1 mm/s",
    "WEIGHTS_PREV": "previous cost-weight preset",
    "WEIGHTS_NEXT": "next cost-weight preset",
    "LEGACY": "legacy 2D planner layer on/off",
    "ANATOMY": "anatomy overlay on/off",
    "SLOWMO": "slow-motion throw on/off",
    "VIEW_NEXT": "next OR camera angle", "VIEW_PREV": "previous angle",
    "OR_WINDOW": "big zoomable 3D window", "SCREENSHOT": "screenshot",
    "MARK": "time mark to CSV", "QUIT": "quit",
    "PROBE_LEFT": "slide the probe 1 mm along -x (before insertion only)",
    "PROBE_RIGHT": "slide the probe 1 mm along +x (before insertion only)",
}

# voice: phrase after the wake word -> command ("stop" works alone)
GRAMMAR = {
    "shortest": "PATH_SHORTEST", "safest": "PATH_SAFEST",
    "recommended": "PATH_RECOMMENDED", "start": "START", "pause": "PAUSE",
    "resume": "RESUME", "faster": "SPEED_UP", "slower": "SPEED_DOWN",
    "next view": "VIEW_NEXT", "picture": "SCREENSHOT", "mark": "MARK",
    "yes": "CONFIRM_YES", "no": "CONFIRM_NO", "compare": "COMPARE",
}


def key_command(key):
    if key is None or key < 0:
        return None
    k = key & 0xFFFF if key > 0xFFFF else key
    if k in KEYMAP:
        return KEYMAP[k]
    if 0 <= k < 256 and ord(chr(k).lower()) in KEYMAP:
        return KEYMAP[ord(chr(k).lower())]
    return None


def vosk_phrases():
    w = cfg.VOICE_WAKE_WORD
    out = ["stop"]
    for ph in GRAMMAR:
        out.append(ph)
        if w:
            out.append(f"{w} {ph}")
            for a in cfg.VOICE_WAKE_ALIASES:
                out.append(f"{a} {ph}")
    return out


def match_phrase(text):
    t = (text or "").strip().lower()
    if not t:
        return None
    if t == "stop" or t.endswith(" stop"):
        return "E_STOP"
    w = cfg.VOICE_WAKE_WORD
    wakes = [w] + list(cfg.VOICE_WAKE_ALIASES) if w else [""]
    for wk in wakes:
        if wk and not t.startswith(wk + " "):
            continue
        rest = t[len(wk):].strip() if wk else t
        if rest in GRAMMAR:
            return GRAMMAR[rest]
    return None


class CommandBus:
    def __init__(self):
        self._lock = threading.Lock()
        self._q = []
        self.last = ("", 0.0, "")

    def push(self, cmd, source="key"):
        if cmd is None:
            return
        with self._lock:
            self._q.append((cmd, source))
            self.last = (cmd, time.time(), source)

    def drain(self):
        with self._lock:
            q, self._q = self._q, []
        return q
