"""
Boundaries: conditions on curve groups of the geometry. Names only select the groups; everything
physical is a parameter or comes from the mesh.

    lp.Clamped("interface_constrained_rz")             both displacement components blocked (default)
    lp.Clamped("interface_constrained_r", dofs="r")    the radial component only
    lp.Coil("interface_forced")                        where the drive acts (axial by default)
    lp.FluidStructure("interface_acou_meca_front")     structure ↔ air coupling (normal from the mesh)
    lp.Radiation("dtn_boundary")                       exact non-reflecting (DtN) arc

`name` selects the groups as for the regions (loudpy.geometry.naming).
"""
from __future__ import annotations

from dataclasses import dataclass

from loudpy.errors import ModelError

COMPONENTS = {"r": (0,), "z": (1,), "rz": (0, 1)}       # displacement components (0 = r, 1 = z)


def _check_components(owner, value):
    if value not in COMPONENTS:
        raise ModelError(f"{owner}: components must be one of {sorted(COMPONENTS)}, got {value!r}")


@dataclass
class Boundary:
    name: str


@dataclass
class Clamped(Boundary):
    """Fixed displacement on the selected curves: dofs "rz" (default, fully clamped), "r" or "z"."""
    dofs: str = "rz"

    def __post_init__(self):
        _check_components(f"Clamped({self.name!r}).dofs", self.dofs)


@dataclass
class Coil(Boundary):
    """Where the drive acts (lp.Force or lp.Voltage), along `direction`: "z" (axial, default) or "r"."""
    direction: str = "z"

    def __post_init__(self):
        _check_components(f"Coil({self.name!r}).direction", self.direction)


@dataclass
class FluidStructure(Boundary):
    """Fluid–structure interface between a Solid and an Air; the normal is the outward normal of the air
    (computed from the mesh, on each edge)."""


@dataclass
class Radiation(Boundary):
    """Exact radiation (Dirichlet-to-Neumann) condition on a spherical boundary of the air."""
