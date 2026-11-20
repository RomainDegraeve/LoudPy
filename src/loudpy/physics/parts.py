"""
Physics objects: one mesh group (composition, not inheritance) with its physical parameters
and its role (registry.py).

    regions     SolidPart (solid), AirPart (air), PMLPart (pml)
    boundaries  ClampedEdge (clamped), CoilEdge (coil), CouplingEdge (fluid_structure),
                RadiationEdge (radiation)

The mesh group is any object with node_tags, node_coords, a name and its cells (tri for a
region, edges for a boundary); the parts expose them unchanged for the assemblers.

A meshed lp.Model builds its parts with region_part / boundary_part (model.parts).
"""
from __future__ import annotations

import numpy as np

from loudpy.geometry.naming import pml_layer


class Part:
    role: str = ""

    def __init__(self, group):
        self.group = group

    @property
    def name(self) -> str:
        return self.group.name

    @property
    def node_tags(self) -> np.ndarray:
        return self.group.node_tags

    @property
    def node_coords(self) -> np.ndarray:
        return self.group.node_coords

    def __repr__(self):
        return f"{type(self).__name__}({self.name!r}, role={self.role!r}, {len(self.node_tags)} nodes)"


class RegionPart(Part):
    @property
    def tri(self) -> np.ndarray:
        return self.group.tri


class EdgePart(Part):
    @property
    def edges(self) -> np.ndarray:
        return self.group.edges


# ── regions ──────────────────────────────────────────────────────────────────
class SolidPart(RegionPart):
    """Axisymmetric solid: ρ, E, ν; Rayleigh (alpha_ray, beta_ray) and/or hysteretic (eta) damping."""
    role = "solid"
    dofs_per_node = 2

    def __init__(self, group, rho, E, nu, alpha_ray=0.0, beta_ray=0.0, eta=0.0):
        super().__init__(group)
        self.rho, self.E, self.nu = rho, E, nu
        self.alpha_ray, self.beta_ray, self.eta = alpha_ray, beta_ray, eta


class AirPart(RegionPart):
    """Air: ρ, c; orientation ±1 (front / rear side of the structure)."""
    role = "air"
    dofs_per_node = 1

    def __init__(self, group, rho, c, orientation=1.0):
        super().__init__(group)
        self.rho, self.c, self.orientation = rho, c, orientation


class PMLPart(AirPart):
    """PML layer: absorption strength alpha, profile order n, tuned to f_pml; its geometry in `layer`."""
    role = "pml"

    def __init__(self, group, orientation, rho, c, alpha, n, f_pml):
        super().__init__(group, rho, c, orientation)
        self.alpha, self.n = alpha, n
        self.omega_pml = 2 * np.pi * f_pml
        self.layer = pml_layer(self.name)


# ── boundaries ───────────────────────────────────────────────────────────────
class ClampedEdge(EdgePart):
    """Blocked displacement components `direction` (0 = r, 1 = z)."""
    role = "clamped"

    def __init__(self, group, direction):
        super().__init__(group)
        self.direction = direction


class CoilEdge(EdgePart):
    """Where the drive acts, along the components `direction`."""
    role = "coil"

    def __init__(self, group, direction):
        super().__init__(group)
        self.direction = direction


class CouplingEdge(EdgePart):
    """Fluid–structure coupling; `direction` is the orientation ±1 of the air side."""
    role = "fluid_structure"

    def __init__(self, group, direction):
        super().__init__(group)
        self.direction = direction


class RadiationEdge(EdgePart):
    """Exact non-reflecting (DtN) arc; orientation ±1 of the acoustic region it closes."""
    role = "radiation"

    def __init__(self, group, direction=None):
        super().__init__(group)
        self.orientation = 1.0 if direction is None else float(direction)


# ── from the model description ───────────────────────────────────────────────
def _material(region, lib):
    from loudpy.model.materials import Material
    m = lib.resolve(region.material) if isinstance(region.material, Material) or region.material in lib \
        else Material(str(region.material))
    return m.with_(**{k: getattr(region, k, None) for k in ("rho", "E", "nu", "c")})


def region_parameters(region, lib) -> dict:
    """Physical parameters of the part of a region (material, overrides, damping, PML settings)."""
    from loudpy.model.damping import Hysteretic, Rayleigh, Undamped
    from loudpy.model.regions import PML, Air, Solid
    m = _material(region, lib)
    if isinstance(region, Solid):
        p = m.require("rho", "E", "nu")
        d = region.damping
        if isinstance(d, Hysteretic):
            p["eta"] = d.eta if d.eta is not None else m.require("eta")["eta"]
        elif isinstance(d, Undamped):
            p.update(alpha_ray=0.0, beta_ray=0.0)
        elif isinstance(d, Rayleigh):
            p["alpha_ray"] = d.alpha if d.alpha is not None else m.require("alpha_ray")["alpha_ray"]
            p["beta_ray"] = d.beta if d.beta is not None else m.require("beta_ray")["beta_ray"]
        else:
            raise TypeError(f"{region.name}: unknown damping {d!r}")
        return p
    if isinstance(region, PML):
        p = m.require("rho", "c")
        p["alpha"] = region.strength if region.strength is not None else m.require("alpha")["alpha"]
        p["n"] = region.order if region.order is not None else m.require("n")["n"]
        if region.tuned_to is None:
            raise ValueError(f"PML {region.name!r}: give tuned_to= (frequency the absorption is tuned to)")
        p["f_pml"] = region.tuned_to
        return p
    if isinstance(region, Air):
        return m.require("rho", "c")
    raise TypeError(f"unknown region {region!r}")


def region_part(region, group, lib) -> RegionPart:
    """The part of a region on one mesh group."""
    from loudpy.geometry import naming
    from loudpy.model.regions import PML, Solid
    p = region_parameters(region, lib)
    orientation = naming.parse(group.name).orientation
    if isinstance(region, Solid):
        return SolidPart(group, **p)
    if isinstance(region, PML):
        return PMLPart(group, orientation=1.0 if orientation is None else orientation, **p)
    return AirPart(group, orientation=1.0 if orientation is None else orientation, **p)


def boundary_value(boundary, group_name: str):
    """Blocked / forced components, or orientation ±1, of a boundary group (given, or from its name)."""
    from loudpy.geometry import naming
    from loudpy.model.boundaries import Clamped, FluidStructure, Radiation
    if isinstance(boundary, Clamped) and boundary.dofs is not None:
        return naming.DIRECTIONS[boundary.dofs]
    if isinstance(boundary, (FluidStructure, Radiation)) and boundary.side is not None:
        return naming.SIDES[boundary.side]
    info = naming.parse(group_name)
    return info.dofs if info.dofs is not None else info.orientation


def boundary_part(boundary, group) -> EdgePart:
    """The part of a boundary on one mesh group."""
    from loudpy.model.boundaries import Clamped, Coil, FluidStructure, Radiation
    cls = {Clamped: ClampedEdge, Coil: CoilEdge, FluidStructure: CouplingEdge, Radiation: RadiationEdge}[type(boundary)]
    value = boundary_value(boundary, group.name)
    if value is None:
        if cls is RadiationEdge:
            return cls(group)
        raise ValueError(f"{type(boundary).__name__}({boundary.name!r}): group {group.name!r} names no "
                         f"direction (_r_, _z_, _rz_) nor side (front / rear); give dofs= or side=")
    return cls(group, value)
