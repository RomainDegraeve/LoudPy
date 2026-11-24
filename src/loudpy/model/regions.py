"""
Regions: the domains of a model (one or several physical groups of the geometry each).

    lp.Solid("cone", "Paper", damping=lp.Rayleigh(), mesh_size=0.5e-3)
    lp.Solid("plate", lp.Material("Steel", rho=7850, E=2.1e11, nu=0.3), damping=lp.Undamped())
    lp.Air("air", mesh_size=5e-3)
    lp.PML("PML", thickness=0.1, tuned_to=1000.0)
    lp.PML("absorber", on=lp.Box(r=0.3) | lp.Box(z=0.532), thickness=0.1, tuned_to=1000.0)
    lp.Solid("cone", "Paper", on=lp.Group("cone", "dustcap"))      # name = label, on = selection

`on` selects the domain groups (loudpy.geometry.selectors: lp.Group, lp.Box, lp.Near and their
combinations; for a PML the curves it grows from); by default the name is the label that selects them (rules in loudpy.geometry.naming:
"glue" selects glue_1 and glue_2, "coil" does not select coil_former; "glue_*" is a
glob). The name is the label of the region in the results. Material properties given here override
those of the material.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from loudpy.checks import check
from loudpy.geometry.selectors import Combinable, Selector, as_selector
from loudpy.model.damping import Damping, Rayleigh
from loudpy.model.facts import Facts, Fit, register
from loudpy.model.materials import Material


@dataclass
class Region(Combinable):
    """Base of the region items (lp.Solid, lp.Air, lp.PML): a name and the selector of its domains (on=…,
    default: the name as a label)."""
    name: str
    on: Selector | str | None = field(default=None, kw_only=True)      # default: Group(name)
    facts = Facts(selects="domains", table=False)                        # (loudpy.model.facts)

    @property
    def selector(self) -> Selector:
        """What the region covers: `on`, or its name as a label."""
        return as_selector(self.name if self.on is None else self.on)


@dataclass
class Solid(Region):
    """Elastic structure (axisymmetric solid) on the physical groups selected by `name`.

    material: a name of the material library or an lp.Material; damping: Rayleigh (default),
    Hysteretic or Undamped; mesh_size [m]; rho, E, nu…: overrides of the material properties."""
    material: str | Material
    damping: Damping = field(default_factory=Rayleigh)
    mesh_size: float = 0.002            # element size [m]
    rho: float | None = None
    E: float | None = None
    nu: float | None = None
    facts = Facts(selects="domains", medium="solid", physics="structure", key="material")

    def __post_init__(self):
        check(self.mesh_size, "mesh_size", f"Solid({self.name!r}).mesh_size")
        check(self.rho, "density", f"Solid({self.name!r}).rho")
        check(self.E, "young_modulus", f"Solid({self.name!r}).E")
        check(self.nu, "poisson_ratio", f"Solid({self.name!r}).nu")


@dataclass
class Air(Region):
    """Acoustic fluid (linear acoustics) on the groups selected by `name`; rho, c override the material."""
    material: str | Material = "Air"
    mesh_size: float = 0.01
    rho: float | None = None
    c: float | None = None
    facts = Facts(selects="domains", medium="air", physics="air", key="material")

    def __post_init__(self):
        check(self.mesh_size, "mesh_size", f"Air({self.name!r}).mesh_size")
        check(self.rho, "density", f"Air({self.name!r}).rho")
        check(self.c, "speed_of_sound", f"Air({self.name!r}).c")


@dataclass
class PML(Region):
    """Perfectly matched layer grown by the mesher from its base curves, outward from the air they bound.

    on: the base curves (straight lines r = const or z = const of the geometry, each bounding one air
    region), any selector of curves: lp.Box(r=0.3) | lp.Box(z=0.5), lp.BoundaryOf("air") - …; by default
    the name is the label of named curve groups ("PML" selects PML_front_r …). Where two layers meet at
    a right angle a corner layer is added. thickness [m]; tuned_to [Hz]: frequency of the absorption."""
    material: str | Material = "Air"
    thickness: float = 1.0              # layer thickness [m]
    mesh_size: float = 0.01
    tuned_to: float | None = None       # frequency the absorption is tuned to [Hz]
    order: float = 1.0                  # order of the absorption profile (1: linear)
    strength: float = 3.0               # absorption strength (dimensionless)
    rho: float | None = None
    c: float | None = None
    facts = Facts(selects="curves", physics="air", role="absorbing", key="thickness",
                  fits=(Fit("air|-", "an outer edge of the air: a layer grown outward from it", shape="straight"),),
                  starting={"thickness": 0.1, "tuned_to": 1000.0})

    def __post_init__(self):
        check(self.thickness, "thickness", f"PML({self.name!r}).thickness")
        check(self.mesh_size, "mesh_size", f"PML({self.name!r}).mesh_size")
        check(self.tuned_to, "frequency_positive", f"PML({self.name!r}).tuned_to")
        check(self.order, "positive", f"PML({self.name!r}).order")
        check(self.strength, "positive", f"PML({self.name!r}).strength")
        check(self.rho, "density", f"PML({self.name!r}).rho")
        check(self.c, "speed_of_sound", f"PML({self.name!r}).c")


for _cls in (Solid, Air, PML):
    register(_cls)
