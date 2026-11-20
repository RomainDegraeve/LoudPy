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
    """Rayleigh damping C = alpha M + beta K (None: the material's alpha_ray / beta_ray)."""
    alpha: float | None = None          # mass-proportional coefficient [1/s] (None: material alpha_ray)
    beta: float | None = None           # stiffness-proportional coefficient [s] (None: material beta_ray)

    def __post_init__(self):
        from loudpy.checks import check
        check(self.alpha, "rayleigh_alpha", "Rayleigh.alpha")
        check(self.beta, "rayleigh_beta", "Rayleigh.beta")


@dataclass(frozen=True)
class Hysteretic:
    """Hysteretic (structural) damping: complex stiffness K (1 + i eta) (None: the material's eta)."""
    eta: float | None = None            # loss factor (None: material eta)

    def __post_init__(self):
        from loudpy.checks import check
        check(self.eta, "loss_factor", "Hysteretic.eta")


@dataclass(frozen=True)
class Undamped:
    """No structural damping."""
    pass


Damping = Rayleigh | Hysteretic | Undamped
