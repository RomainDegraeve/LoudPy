"""
Boundaries: conditions on curve groups of the geometry. Names only select the groups; everything
physical is a parameter or comes from the mesh.

    lp.Clamped("rims", on=lp.Group("surround_rim", "spider_rim"))             both displacement components blocked (default)
    lp.Clamped("axis", dofs="r")    the radial component only
    lp.Coil("coil_drive")                        where the drive acts (axial by default)
    lp.FluidStructure("wet_front")     structure ↔ air coupling (normal from the mesh)
    lp.Radiation("dtn_boundary")                       exact non-reflecting (DtN) arc
    lp.FluidStructure("cone_front", on=lp.Interface("cone", "air_front"))

`on` selects the edges (loudpy.geometry.selectors), evaluated on the mesh: lp.Interface, lp.BoundaryOf,
lp.Box, lp.Near, lp.Group and their combinations. By default the name is the label that selects named
curve groups (rules of loudpy.geometry.naming). The name is the label of the boundary in the results.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loudpy.errors import ModelError
from loudpy.geometry.selectors import Combinable, Selector, as_selector
from loudpy.model.facts import Facts, Fit, register

COMPONENTS = {"r": (0,), "z": (1,), "rz": (0, 1)}       # displacement components (0 = r, 1 = z)


def _check_components(owner, value):
    if value not in COMPONENTS:
        raise ModelError(f"{owner}: components must be one of {sorted(COMPONENTS)}, got {value!r}")


@dataclass
class Boundary(Combinable):
    """Base of the boundary items (lp.Clamped, lp.Coil, lp.FluidStructure, lp.Radiation): a name and the
    selector of its edges (on=…, default: the name as a label)."""
    name: str
    on: Selector | str | None = field(default=None, kw_only=True)      # default: Group(name)
    facts = Facts(selects="curves", table=False)                         # (loudpy.model.facts)

    @property
    def selector(self) -> Selector:
        """What the boundary covers: `on`, or its name as a label."""
        return as_selector(self.name if self.on is None else self.on)


@dataclass
class Clamped(Boundary):
    """Fixed displacement on the selected curves: dofs "rz" (default, fully clamped), "r" or "z"."""
    dofs: str = "rz"
    facts = Facts(selects="curves", physics="structure", role="held", key="dofs", fits=(
        Fit("solid|-", "a free edge of the structure: clamped where it is held"),
        Fit("solid|axis", "the symmetry axis: no radial displacement there", params={"dofs": "r"})))

    def __post_init__(self):
        _check_components(f"Clamped({self.name!r}).dofs", self.dofs)


@dataclass
class Coil(Boundary):
    """Where the drive acts (lp.Force, lp.Voltage or an lp.Circuit), along `direction`: "z" (axial, default) or
    "r"; `motor` the driver's electrical motor — its blocked coil and force factor (loudpy.circuit: Semi, Leach, LR2,
    Wright) — that a voltage or a circuit drives (kept in the result files: a scene's driver takes it from its
    modes file)."""
    direction: str = "z"
    motor: Any = None
    facts = Facts(selects="curves", physics="structure", role="drive", key="direction", fits=(
        Fit("solid|solid", "where the drive acts (the coil's edge)", usual=False),
        Fit("solid|-", "where the drive acts (the coil's edge)", usual=False)))

    def __post_init__(self):
        _check_components(f"Coil({self.name!r}).direction", self.direction)
        from loudpy.circuit import COIL_MODELS
        if self.motor is not None and not isinstance(self.motor, COIL_MODELS):
            raise ModelError(f"Coil({self.name!r}).motor: one of {[t.__name__ for t in COIL_MODELS]} "
                             f"(loudpy.circuit), got {self.motor!r}")


@dataclass
class FluidStructure(Boundary):
    """Fluid–structure interface between a Solid and an Air; the normal is the outward normal of the air
    (computed from the mesh, on each edge)."""
    facts = Facts(selects="curves", physics="coupling", role="coupling",
                  fits=(Fit("solid|air", "between the structure and the air: a fluid–structure interface"),))


@dataclass
class Radiation(Boundary):
    """Exact radiation (Dirichlet-to-Neumann) condition on a spherical boundary of the air."""
    facts = Facts(selects="curves", physics="air", role="absorbing",
                  fits=(Fit("air|-", "an outer arc of the air: exact radiation on it", shape="arc"),))


for _cls in (Clamped, Coil, FluidStructure, Radiation):
    register(_cls)
