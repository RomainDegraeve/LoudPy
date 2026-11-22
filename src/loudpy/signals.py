"""
Drive signals for lp.Transient (a force on the coil, lp.Force(signal, fs=…)): each function returns (signal, fs).

    sig, fs = lp.signals.sine(200.0, periods=5, fs=48000, amplitude=1.0)
    sig, fs = lp.signals.multitone([50, 120, 330], duration=0.1, fs=48000)     equal amplitudes, low crest factor
    sig, fs = lp.signals.wav("burst.wav", duration=0.05, amplitude=2.0)       first channel, scaled to ±amplitude
    lp.Transient(model, drive=lp.Force(sig, fs=fs)).solve(n_steps=len(sig) - 1)

Times start at 0 and include the last sample (len = duration · fs + 1).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from loudpy.errors import ModelError

__all__ = ["sine", "multitone", "wav"]


def _times(duration: float, fs: float) -> np.ndarray:
    if not duration > 0 or not fs > 0:
        raise ModelError(f"signals: duration and fs must be > 0, got {duration!r} s and {fs!r} Hz")
    return np.arange(int(round(duration * fs)) + 1) / float(fs)


def sine(f: float, periods: float = 5.0, fs: float = 48000.0, amplitude: float = 1.0, duration: float | None = None):
    """A sine of frequency f [Hz] over `periods` periods (or `duration` [s]); (signal, fs)."""
    if not f > 0:
        raise ModelError(f"signals.sine: f must be > 0 Hz, got {f!r}")
    if duration is not None and not duration > 0:
        raise ModelError(f"signals.sine: duration must be > 0 s (or None: periods / f), got {duration!r}")
    if not duration and not periods > 0:
        raise ModelError(f"signals.sine: periods must be > 0, got {periods!r}")
    t = _times(duration if duration else periods / f, fs)
    return amplitude * np.sin(2 * np.pi * f * t), float(fs)


def multitone(freqs, duration: float | None = None, fs: float = 48000.0, amplitude: float = 1.0,
              periods: float = 5.0):
    """
    Sines of equal amplitude at `freqs` [Hz] with Schroeder phases φ_k = −π k (k − 1) / N (a low crest factor), the
    sum scaled by 1/N and by `amplitude`; over `duration` [s] (default: `periods` periods of the lowest frequency).
    """
    freqs = [float(f) for f in np.atleast_1d(freqs)]
    if not freqs or min(freqs) <= 0:
        raise ModelError("signals.multitone: give positive frequencies [Hz]")
    if duration is not None and not duration > 0:
        raise ModelError(f"signals.multitone: duration must be > 0 s (or None), got {duration!r}")
    if not duration and not periods > 0:
        raise ModelError(f"signals.multitone: periods must be > 0, got {periods!r}")
    t = _times(duration if duration else periods / min(freqs), fs)
    n = len(freqs)
    x = sum(np.sin(2 * np.pi * f * t - np.pi * k * (k - 1) / n) for k, f in enumerate(freqs, start=1)) / n
    return amplitude * np.asarray(x, float), float(fs)


def wav(path, duration: float | None = None, amplitude: float = 1.0):
    """A WAV file as a signal: its first channel scaled to ±`amplitude` (peak), cut to `duration` [s]; its own
    sampling rate is returned."""
    from scipy.io import wavfile
    path = Path(path)
    if not path.exists():
        raise ModelError(f"signals.wav: {path} not found")
    fs, x = wavfile.read(path)
    x = np.asarray(x, float)
    x = x[:, 0] if x.ndim == 2 else x
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    x = x / peak if peak > 0 else x
    if duration:
        x = x[: int(round(float(duration) * fs)) + 1]
    return amplitude * x, float(fs)
