"""
Boundaries: conditions on curve groups of the geometry.

    lp.Clamped("interface_constrained")             blocked DOFs from each group name (_r_, _z_, _rz_)
    lp.Clamped("interface_constrained", dofs="rz")  the same blocked DOFs for all its groups
    lp.Coil("interface_forced")                     where the drive acts
    lp.FluidStructure("interface_acou_meca_front")  structure ↔ air coupling (side from the name)
    lp.Radiation("interface_dtn")                   exact non-reflecting (DtN) arc

`name` selects the groups as for the regions (loudpy.geometry.naming).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Boundary:
    name: str


@dataclass
class Clamped(Boundary):
    """Fixed displacement on the selected curves (dofs "r", "z" or "rz"; None: from the group names)."""
    dofs: str | None = None             # "r", "z" or "rz" (None: from each group name)


@dataclass
class Coil(Boundary):
    """Where the drive acts: lp.Force (axial) or lp.Voltage (electro-mechanical coil)."""
    pass


@dataclass
class FluidStructure(Boundary):
    """Fluid–structure interface between a Solid and an Air (side "front" / "rear"; None: from the names)."""
    side: str | None = None             # "front" / "rear" (None: from each group name)


@dataclass
class Radiation(Boundary):
    """Exact radiation (Dirichlet-to-Neumann) condition on a spherical boundary of the air."""
    side: str | None = None
