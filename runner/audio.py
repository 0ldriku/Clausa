"""A recording's waveform and playing part of it, for the window's Part to analyse.

peaks() reads the whole recording once (PyAV, any format the app takes) into the
loudest sample of each of n equal slices; clip() writes a stretch of it as a WAV
(mono, 16 kHz), and Player plays that through the system: afplay on macOS,
winsound on Windows (the webview cannot read files outside the app's folder).
"""

from __future__ import annotations

import subprocess
import sys
import threading
import wave
from pathlib import Path

import numpy as np

PEAK_RATE = 8000  # Hz, enough to draw speech
CLIP_RATE = 16000


def _frames(path: Path, rate: int, start: float = 0.0):
    """(time of the first sample, mono float32 samples at rate) per decoded frame, from start on."""
    import av
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        if start > 0:
            container.seek(int(start * av.time_base), any_frame=False, backward=True)
        resampler = av.AudioResampler(format="flt", layout="mono", rate=rate)
        t = None
        for frame in container.decode(stream):
            if t is None:
                t = float(frame.pts * frame.time_base) if frame.pts is not None else 0.0
            for out in resampler.resample(frame):
                samples = out.to_ndarray().reshape(-1)
                yield t, samples
                t += len(samples) / rate
        for out in resampler.resample(None):
            samples = out.to_ndarray().reshape(-1)
            yield t, samples
            t += len(samples) / rate


def peaks(path: Path, duration: float, n: int = 600) -> list[float]:
    """The loudest sample of each of n equal slices of the recording, 0 to 1 (scaled to the loudest)."""
    out = np.zeros(n, dtype=np.float32)
    per = max(duration, 1e-6) / n
    for t, samples in _frames(path, PEAK_RATE):
        if not len(samples):
            continue
        idx = np.clip(((t + np.arange(len(samples)) / PEAK_RATE) / per).astype(np.int64), 0, n - 1)
        np.maximum.at(out, idx, np.abs(samples))
    top = float(out.max())
    return [round(float(v) / top, 3) if top > 0 else 0.0 for v in out]


def clip(path: Path, start: float, end: float, out_wav: Path) -> None:
    """Write [start, end] of the recording to out_wav (16-bit mono, CLIP_RATE)."""
    parts = []
    for t, samples in _frames(path, CLIP_RATE, start):
        if t >= end:
            break
        a = max(0, int(round((start - t) * CLIP_RATE)))
        b = min(len(samples), int(round((end - t) * CLIP_RATE)))
        if b > a:
            parts.append(samples[a:b])
    data = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    pcm = (np.clip(data, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(out_wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(CLIP_RATE)
        w.writeframes(pcm.tobytes())


class Player:
    """Plays one WAV at a time through the system; stop() ends it."""

    def __init__(self):
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None

    def play(self, wav: Path) -> None:
        with self._lock:
            self._stop()
            if sys.platform == "win32":
                import winsound
                winsound.PlaySound(str(wav), winsound.SND_FILENAME | winsound.SND_ASYNC)
            else:
                self._proc = subprocess.Popen(["afplay", str(wav)], stdout=subprocess.DEVNULL,
                                              stderr=subprocess.DEVNULL)

    def stop(self) -> None:
        with self._lock:
            self._stop()

    def _stop(self) -> None:
        if sys.platform == "win32":
            import winsound
            winsound.PlaySound(None, 0)
        elif self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None
