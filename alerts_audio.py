"""Two distinct, dependency-free alert sounds (generated, no audio files needed).
fatigue / microsleep : low, slow double buzz (330 Hz square wave)
distraction          : high, fast triple beep (1400 Hz square wave)
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time
import wave

import numpy as np

RATE = 22050


def _beeps(freq, dur, gap, n):
    t = np.arange(int(RATE * dur)) / RATE
    beep = np.where(np.sin(2 * np.pi * freq * t) >= 0, 1.0, -1.0)   # square wave = buzzer
    fade = int(RATE * 0.01)                                          # 10 ms fade avoids clicks
    beep[:fade] *= np.linspace(0, 1, fade)
    beep[-fade:] *= np.linspace(1, 0, fade)
    unit = np.concatenate([beep, np.zeros(int(RATE * gap))])
    return (np.tile(unit, n) * 0.5 * 32767).astype(np.int16)


class AlertPlayer:
    COOLDOWN = {"fatigue": 3.0, "distraction": 2.0}   # s between repeats while alert stays active

    def __init__(self):
        samples = {"fatigue": _beeps(330, 0.45, 0.15, 2),
                   "distraction": _beeps(1400, 0.12, 0.08, 3)}
        folder = tempfile.mkdtemp(prefix="driver_alerts_")
        self.paths, self.dur = {}, {}
        for kind, s in samples.items():
            self.paths[kind] = os.path.join(folder, kind + ".wav")
            with wave.open(self.paths[kind], "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(RATE)
                w.writeframes(s.tobytes())
            self.dur[kind] = len(s) / RATE
        if sys.platform == "win32":
            self.cmd = None
        elif sys.platform == "darwin":
            self.cmd = "afplay"
        else:
            self.cmd = next((c for c in ("paplay", "aplay") if shutil.which(c)), None)
        self.last = {k: -99.0 for k in samples}
        self.busy_until = 0.0

    def _play(self, kind):
        path = self.paths[kind]
        if sys.platform == "win32":
            import winsound
            winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
        elif self.cmd:
            subprocess.Popen([self.cmd, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            print("\a", end="", flush=True)   # no player found: terminal bell

    def update(self, kinds):
        """Call every frame with the set of active alert kinds. Plays one sound at a time
        (fatigue first); if both are active they take turns."""
        t = time.monotonic()
        if t < self.busy_until:
            return
        for kind in ("fatigue", "distraction"):
            if kind in kinds and t - self.last[kind] >= self.COOLDOWN[kind]:
                self._play(kind)
                self.last[kind] = t
                self.busy_until = t + self.dur[kind] + 0.2
                return
