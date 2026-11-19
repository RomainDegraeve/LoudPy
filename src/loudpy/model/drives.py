"""
Drives: how a study excites the model.

    lp.Force(1.0)                               harmonic force on the coil [N] (complex: amplitude, phase)
    lp.Force(signal, fs=48_000)                 force time signal [N] for a transient study
    lp.Voltage(2.83, Re=8, Le=4e-3, Bl=7)       voltage on the voice coil, with its circuit
    {"top": lp.Force(1), "bottom": lp.Force(-1)}   one drive per driver (3D scenes)

Phase 1 of the refactor defines them; the studies use them from phase 5 on.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Force:
    amplitude: complex | np.ndarray = 1.0       # [N], or a time signal (with fs)
    fs: float | None = None                     # sampling rate of the signal [Hz]

    @property
    def is_signal(self) -> bool:
        return np.ndim(self.amplitude) > 0


@dataclass(frozen=True)
class Voltage:
    amplitude: complex                          # [V]
    Re: float                                   # coil resistance [Ω]
    Bl: float                                   # force factor [T·m]
    Le: float = 0.0                             # coil inductance [H]
    n: float = 1.0                              # inductance exponent: Ze = Re + (jω)^n Le

    def impedance(self, omega: float) -> complex:
        """Blocked electrical impedance Ze(ω)."""
        return self.Re + (1j * omega) ** self.n * self.Le


Drive = Force | Voltage
