"""
Progress of long computations, for an interface (loudpy.ui) or your own monitor. Silent unless a callback is set.

    from loudpy import progress
    progress.set_callback(lambda what, i, n: print(f"{what} {i}/{n}"))
    lp.FrequencyResponse(model).solve(freqs)          # "frequency 1/200", "frequency 2/200", …
    progress.set_callback(None)

Reported: frequencies of a frequency response, time steps of a transient, shifts of a modes search (n unknown: 0),
runs of a sweep. The callback runs in the computing thread; keep it short.
"""
from __future__ import annotations

from typing import Callable

_callback: Callable[[str, int, int], None] | None = None

__all__ = ["set_callback", "step"]


def set_callback(fn: Callable[[str, int, int], None] | None):
    """Set (or clear, None) the function called as fn(what, i, n) at every step of a long computation."""
    global _callback
    _callback = fn


def step(what: str, i: int, n: int = 0):
    """Report step i of n (n = 0: unknown) of `what`; does nothing without a callback."""
    if _callback is not None:
        _callback(what, int(i), int(n))
