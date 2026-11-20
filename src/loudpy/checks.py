"""
SI plausibility checks of user inputs.

loudpy works in SI units everywhere (m, kg, s, Pa, Hz, N, V, Ω, H, K). Every input that has a
physical meaning is checked once, where it enters the model:

  - a value outside its **valid** range raises ModelError (a negative density, ν ≥ 0.5 …);
  - a valid value outside its **plausible** range warns (PlausibilityWarning): it is kept, but it is
    probably not in SI units (a mesh size of 2 for 2 mm, a Young's modulus in MPa …).

    from loudpy.checks import check
    size = check(size, "mesh_size", "Solid('cone').mesh_size")
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np

from loudpy.errors import ModelError, PlausibilityWarning


@dataclass(frozen=True)
class Range:
    """Valid range (lo, hi, inclusive or not) and plausible range of a kind of value, with its SI unit."""
    lo: float
    hi: float
    plausible: tuple[float, float]
    unit: str
    lo_inclusive: bool = False
    hi_inclusive: bool = True
    hint: str = ""

    def valid(self, v: float) -> bool:
        above = v >= self.lo if self.lo_inclusive else v > self.lo
        below = v <= self.hi if self.hi_inclusive else v < self.hi
        return above and below

    def describe(self) -> str:
        lo = "[" if self.lo_inclusive else "("
        hi = "]" if self.hi_inclusive else ")"
        return f"{lo}{self.lo:g}, {self.hi:g}{hi}" + (f" {self.unit}" if self.unit else "")


INF = math.inf
RANGES: dict[str, Range] = {
    "length": Range(0.0, INF, (1e-6, 10.0), "m", hint="lengths are in metres"),
    "mesh_size": Range(0.0, INF, (1e-6, 1.0), "m", hint="element sizes are in metres (0.5 mm = 5e-4)"),
    "thickness": Range(0.0, INF, (1e-5, 10.0), "m", hint="thicknesses are in metres"),
    "frequency": Range(0.0, INF, (1e-3, 1e6), "Hz", lo_inclusive=True),
    "frequency_positive": Range(0.0, INF, (1e-3, 1e6), "Hz"),
    "sampling_rate": Range(0.0, INF, (100.0, 1e7), "Hz"),
    "density": Range(0.0, INF, (1e-2, 3e4), "kg/m³", hint="densities are in kg/m³ (paper ≈ 1e3)"),
    "young_modulus": Range(0.0, INF, (1e3, 1.5e12), "Pa", hint="moduli are in Pa (1 GPa = 1e9)"),
    "poisson_ratio": Range(-1.0, 0.5, (-0.5, 0.4999), "", hi_inclusive=False),
    "speed_of_sound": Range(0.0, INF, (10.0, 1e4), "m/s"),
    "loss_factor": Range(0.0, INF, (0.0, 2.0), "", lo_inclusive=True),
    "rayleigh_alpha": Range(0.0, INF, (0.0, 1e5), "1/s", lo_inclusive=True),
    "rayleigh_beta": Range(0.0, INF, (0.0, 1e-2), "s", lo_inclusive=True),
    "temperature": Range(0.0, INF, (173.15, 473.15), "K", hint="temperatures are in kelvin (20 °C = 293.15)"),
    "pressure": Range(0.0, INF, (1e3, 1e7), "Pa", hint="static pressures are in Pa (101325)"),
    "relative_humidity": Range(0.0, 1.0, (0.0, 1.0), "", lo_inclusive=True, hint="a fraction (50 % = 0.5)"),
    "resistance": Range(0.0, INF, (1e-3, 1e3), "Ω"),
    "inductance": Range(0.0, INF, (0.0, 1.0), "H", lo_inclusive=True, hint="inductances are in H (1 mH = 1e-3)"),
    "inductance_exponent": Range(0.0, 1.0, (0.3, 1.0), "", lo_inclusive=False),
    "force_factor": Range(0.0, INF, (1e-3, 100.0), "T·m"),
    "elements_per_wavelength": Range(0.0, INF, (2.0, 50.0), ""),
    "wavelengths": Range(0.0, INF, (0.01, 10.0), "λ"),
    "positive": Range(0.0, INF, (0.0, INF), ""),
    "non_negative": Range(0.0, INF, (0.0, INF), "", lo_inclusive=True),
}


def check(value, kind: str, name: str):
    """
    Check a scalar or array `value` of a given kind (a key of RANGES) and return it unchanged.
    None passes through (an optional value not given). Complex values (amplitudes) are not checked.
    """
    if value is None:
        return value
    r = RANGES[kind]
    arr = np.asarray(value)
    if np.iscomplexobj(arr) or arr.dtype.kind not in "fiu":
        return value
    if arr.size == 0:
        return value
    if not np.all(np.isfinite(arr)):
        raise ModelError(f"{name} must be finite, got {value!r}")
    lo, hi = float(arr.min()), float(arr.max())
    for v in (lo, hi):
        if not r.valid(v):
            raise ModelError(f"{name} = {v:g} is outside its valid range {r.describe()}")
    p_lo, p_hi = r.plausible
    if lo < p_lo or hi > p_hi:
        v = lo if lo < p_lo else hi
        unit = f" {r.unit}" if r.unit else ""
        hint = f" ({r.hint})" if r.hint else ""
        warnings.warn(f"{name} = {v:g}{unit} is unusual (expected {p_lo:g} to {p_hi:g}{unit}); loudpy uses SI "
                      f"units{hint}", PlausibilityWarning, stacklevel=3)
    return value


__all__ = ["check", "RANGES", "Range"]
