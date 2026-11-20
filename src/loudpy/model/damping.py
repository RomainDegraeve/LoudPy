"""
Damping models of a solid, given separately from its material.

    lp.Rayleigh()                   C = α M + β K with α, β from the material's fit (Material.rayleigh)
    lp.Rayleigh(alpha=10, beta=2e-6)
    lp.Rayleigh.from_loss_factor(0.03)  α, β giving η = 0.03 at 100 Hz and 1 kHz
    lp.Hysteretic()                 complex stiffness E (1 + iη), η from the material (Material.loss_factor)
    lp.Hysteretic(eta=0.1)
    lp.Undamped()                   no damping
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Rayleigh:
    """Rayleigh damping C = alpha M + beta K (None: from the material's fit Material.rayleigh)."""
    alpha: float | None = None          # mass-proportional coefficient [1/s] (None: Material.rayleigh[0])
    beta: float | None = None           # stiffness-proportional coefficient [s] (None: Material.rayleigh[1])

    def __post_init__(self):
        from loudpy.checks import check
        check(self.alpha, "rayleigh_alpha", "Rayleigh.alpha")
        check(self.beta, "rayleigh_beta", "Rayleigh.beta")

    @classmethod
    def from_loss_factor(cls, eta: float, band: tuple[float, float] = (100.0, 1000.0)) -> "Rayleigh":
        """
        The Rayleigh damping whose loss factor equals `eta` at the two frequencies of `band` [Hz]
        (η(ω) = α/ω + βω, lower in between, higher outside):

            α = η ω₁ω₂ / (ω₁ + ω₂),   β = η / (ω₁ + ω₂)

        The fits of loudpy's material library use the band 100 Hz – 1 kHz.
        """
        from loudpy.checks import check
        check(eta, "loss_factor", "Rayleigh.from_loss_factor: eta")
        f1, f2 = (check(f, "frequency_positive", "Rayleigh.from_loss_factor: band") for f in band)
        w1, w2 = 2 * math.pi * f1, 2 * math.pi * f2
        return cls(alpha=eta * w1 * w2 / (w1 + w2), beta=eta / (w1 + w2))

    def loss_factor(self, f):
        """Equivalent loss factor η(f) = α/ω + βω of this damping at the frequencies f [Hz]."""
        w = 2 * math.pi * np.asarray(f, float)
        return (self.alpha or 0.0) / w + (self.beta or 0.0) * w


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
