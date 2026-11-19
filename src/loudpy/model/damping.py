"""
Damping models of a solid, given separately from its material.

    lp.Rayleigh()                   C = α M + β K with α, β from the material (alpha_ray, beta_ray)
    lp.Rayleigh(alpha=10, beta=2e-6)
    lp.Hysteretic()                 complex stiffness E (1 + iη), η from the material (eta)
    lp.Hysteretic(eta=0.1)
    lp.Undamped()                   no damping
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rayleigh:
    alpha: float | None = None          # mass-proportional coefficient [1/s] (None: material alpha_ray)
    beta: float | None = None           # stiffness-proportional coefficient [s] (None: material beta_ray)


@dataclass(frozen=True)
class Hysteretic:
    eta: float | None = None            # loss factor (None: material eta)


@dataclass(frozen=True)
class Undamped:
    pass


Damping = Rayleigh | Hysteretic | Undamped
