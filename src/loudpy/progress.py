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

from contextlib import contextmanager
from typing import Callable

_callback: Callable[[str, int, int], None] | None = None

__all__ = ["part", "set_callback", "step"]


def set_callback(fn: Callable[[str, int, int], None] | None):
    """Set (or clear, None) the function called as fn(what, i, n) at every step of a long computation."""
    global _callback
    _callback = fn


def step(what: str, i: int, n: int = 0):
    """Report step i of n (n = 0: unknown) of `what`; does nothing without a callback."""
    if _callback is not None:
        _callback(what, int(i), int(n))


@contextmanager
def part(k: int, n_parts: int, label: str = ""):
    """Steps reported inside count as part k (0-based) of n_parts: step i of n becomes step k·n + i of n_parts·n,
    `label` put before what is reported (the azimuthal orders of a modes search: one count over all of them)."""
    global _callback
    outer = _callback
    if outer is None or n_parts <= 1:
        yield
        return

    def inner(what: str, i: int, n: int):
        outer(f"{label}{what}", k * n + i if n else i, n_parts * n)
    _callback = inner
    try:
        yield
    finally:
        _callback = outer
