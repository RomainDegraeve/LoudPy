"""
Damping models of a solid, given separately from its material.

    lp.Rayleigh()                   C = α M + β K with α, β from the material's fit (Material.rayleigh)
    lp.Rayleigh(alpha=10, beta=2e-6)
    lp.Hysteretic()                 complex stiffness E (1 + iη), η from the material (Material.loss_factor)
    lp.Hysteretic(eta=0.1)
    lp.Undamped()                   no damping
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rayleigh:
    """Rayleigh damping C = alpha M + beta K (None: from the material's fit Material.rayleigh)."""
    alpha: float | None = None          # mass-proportional coefficient [1/s] (None: Material.rayleigh[0])
    beta: float | None = None           # stiffness-proportional coefficient [s] (None: Material.rayleigh[1])

    def __post_init__(self):
        from loudpy.checks import check
        check(self.alpha, "rayleigh_alpha", "Rayleigh.alpha")
        check(self.beta, "rayleigh_beta", "Rayleigh.beta")


@dataclass(frozen=True)
class Hysteretic:
    """Hysteretic (structural) damping: complex stiffness K (1 + i eta) (None: Material.loss_factor)."""
    eta: float | None = None            # loss factor (None: Material.loss_factor)

    def __post_init__(self):
        from loudpy.checks import check
        check(self.eta, "loss_factor", "Hysteretic.eta")


@dataclass(frozen=True)
class Undamped:
    """No structural damping."""
    pass


Damping = Rayleigh | Hysteretic | Undamped
