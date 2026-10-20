"""
Optional offline voice commands (Vosk), same design as the reference.
Off by default (--voice). If vosk / sounddevice / the model are missing,
it says so and the simulation stays keyboard-only.

    pip install vosk sounddevice
    model: vosk-model-small-en-us-0.15 -> models/vosk-small-en/
"""
import json
import os
import queue
import threading

from ultrabot_sim import config as cfg
from ultrabot_sim.commands import match_phrase, vosk_phrases


class VoiceListener(threading.Thread):
    def __init__(self, bus, model_dir=None):
        super().__init__(daemon=True)
        self.bus = bus
        self.model_dir = model_dir or cfg.VOSK_MODEL_DIR
        self._stop_evt = threading.Event()
        self.heard = ""
        self.available = False
        self.status = "not started"
        self._q = queue.Queue()
        self._rec = None
        self._sd = None

    def setup(self):
        try:
            import sounddevice as sd
            from vosk import KaldiRecognizer, Model
        except ImportError:
            self.status = "vosk/sounddevice not installed - keyboard only"
            return False
        if not os.path.isdir(self.model_dir):
            self.status = f"model not found at {self.model_dir} - keyboard only"
            return False
        try:
            model = Model(self.model_dir)
            grammar = json.dumps(vosk_phrases() + ["[unk]"])
            self._rec = KaldiRecognizer(model, cfg.VOICE_SAMPLE_RATE, grammar)
            self._sd = sd
            self.available = True
            self.status = f'listening - say "{cfg.VOICE_WAKE_WORD} ..."'
            return True
        except Exception as e:                              # noqa: BLE001
            self.status = f"voice init failed: {e}"
            return False

    def _callback(self, indata, frames, time_info, status):
        self._q.put(bytes(indata))

    def run(self):
        if not self.setup():
            print(f"[voice] {self.status}")
            return
        print(f"[voice] {self.status}")
        with self._sd.RawInputStream(samplerate=cfg.VOICE_SAMPLE_RATE,
                                     blocksize=4000, dtype="int16",
                                     channels=1, callback=self._callback):
            while not self._stop_evt.is_set():
                try:
                    data = self._q.get(timeout=0.2)
                except queue.Empty:
                    continue
                if self._rec.AcceptWaveform(data):
                    text = json.loads(self._rec.Result()).get("text", "")
                    if text:
                        self.heard = text
                        cmd = match_phrase(text)
                        if cmd:
                            self.bus.push(cmd, "voice")

    def stop(self):
        self._stop_evt.set()
