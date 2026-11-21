"""
Drives: how a study excites the model.

    lp.Force(1.0)                               harmonic force on the coil [N] (complex: amplitude, phase)
    lp.Force(signal, fs=48_000)                 force time signal [N] for a transient study
    lp.Voltage(2.83, Re=8, Le=4e-3, Bl=7)       voltage on the voice coil, with its circuit
    {"top": lp.Force(1), "bottom": lp.Force(-1)}   one drive per driver (3D scenes)

The studies check the drive they get (FrequencyResponse: harmonic Force or Voltage; Transient: a
Force signal; a Scene: Force or {driver name: Force}).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from loudpy.errors import ModelError


@dataclass(frozen=True)
class Force:
    """Force on the coil: a complex amplitude [N] (harmonic), or a time signal sampled at fs [Hz] (transient)."""
    amplitude: complex | np.ndarray = 1.0       # [N], or a time signal (with fs)
    fs: float | None = None                     # sampling rate of the signal [Hz]

    def __post_init__(self):
        from loudpy.checks import check
        check(self.fs, "sampling_rate", "Force.fs")
        if self.is_signal and self.fs is None:
            raise ModelError("a Force signal needs its sampling rate: lp.Force(signal, fs=…)")

    @property
    def is_signal(self) -> bool:
        """True when the amplitude is a time signal (an array), for lp.Transient."""
        return np.ndim(self.amplitude) > 0


@dataclass(frozen=True)
class Voltage:
    """Voltage drive of an electro-mechanical coil: amplitude [V], resistance Re [Ω], force factor
    Bl [T·m], inductance Le [H] (semi-inductance exponent n)."""
    amplitude: complex                          # [V]
    Re: float                                   # coil resistance [Ω]
    Bl: float                                   # force factor [T·m]
    Le: float = 0.0                             # coil inductance [H]
    n: float = 1.0                              # inductance exponent: Ze = Re + (jω)^n Le

    def __post_init__(self):
        from loudpy.checks import check
        check(self.Re, "resistance", "Voltage.Re")
        check(self.Bl, "force_factor", "Voltage.Bl")
        check(self.Le, "inductance", "Voltage.Le")
        check(self.n, "inductance_exponent", "Voltage.n")

    def impedance(self, omega: float) -> complex:
        """Blocked electrical impedance Ze(ω)."""
        return self.Re + (1j * omega) ** self.n * self.Le


Drive = Force | Voltage
