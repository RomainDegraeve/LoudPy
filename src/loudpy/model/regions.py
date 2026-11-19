"""
Regions: the domains of a model (one or several physical groups of the geometry each).

    lp.Solid("membranne", "Paper", damping=lp.Rayleigh(), mesh_size=0.5e-3)
    lp.Solid("plate", lp.Material("Steel", rho=7850, E=2.1e11, nu=0.3), damping=lp.Undamped())
    lp.Air("subacou", mesh_size=5e-3)
    lp.PML("PML", thickness=0.1, tuned_to=1000.0)

`name` selects the physical groups (rules in loudpy.geometry.naming: "glue" selects
submeca_glue_1 and submeca_glue_2, "coil" does not select coil_former; "glue_*" is a glob).
Material properties given here override those of the material.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from loudpy.model.damping import Damping, Rayleigh
from loudpy.model.materials import Material


@dataclass
class Region:
    name: str


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


@dataclass
class Air(Region):
    """Acoustic fluid (linear acoustics) on the groups selected by `name`; rho, c override the material."""
    material: str | Material = "Air"
    mesh_size: float = 0.01
    rho: float | None = None
    c: float | None = None


@dataclass
class PML(Region):
    """Perfectly matched layer built by the mesher outward from the PML curves of the geometry."""
    material: str | Material = "Air"
    thickness: float = 1.0              # layer thickness [m]
    mesh_size: float = 0.01
    tuned_to: float | None = None       # frequency the absorption is tuned to [Hz]
    order: float | None = None          # profile order (None: material n)
    strength: float | None = None       # absorption strength (None: material alpha)
    rho: float | None = None
    c: float | None = None
