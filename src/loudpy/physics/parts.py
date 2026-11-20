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
from loudpy.errors import ModelError


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
    """Air: ρ, c (the same formulation in every air region: no front / rear sign)."""
    role = "air"
    dofs_per_node = 1

    def __init__(self, group, rho, c):
        super().__init__(group)
        self.rho, self.c = rho, c


class PMLPart(AirPart):
    """PML layer: absorption strength alpha, profile order n, tuned to f_pml; its geometry in `layer` (a
    PMLLayer stored with the mesh; for meshes written before, read from the name of the surface)."""
    role = "pml"

    def __init__(self, group, rho, c, alpha, n, f_pml, layer=None):
        super().__init__(group, rho, c)
        self.alpha, self.n = alpha, n
        self.omega_pml = 2 * np.pi * f_pml
        self.layer = layer if layer is not None else pml_layer(self.name)


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
    """
    Fluid–structure coupling. `signs` (one per edge): +1 when the edge normal (dz, −dr) given by its
    node order points out of the adjacent air, −1 otherwise, so that every edge couples with the
    outward normal of the air. Set by the model from the mesh (geometry.topology), never from names.
    """
    role = "fluid_structure"

    def __init__(self, group, signs=None):
        super().__init__(group)
        self.signs = None if signs is None else np.asarray(signs, float)


class RadiationEdge(EdgePart):
    """Exact non-reflecting (DtN) arc closing an air region."""
    role = "radiation"


# ── from the model description ───────────────────────────────────────────────
def _material(region, lib):
    from loudpy.model.materials import Material
    m = lib.resolve(region.material) if isinstance(region.material, Material) or region.material in lib \
        else Material(str(region.material))
    return m.with_(**{k: getattr(region, k, None) for k in ("rho", "E", "nu", "c")})


def region_parameters(region, lib) -> dict:
    """Physical parameters of the part of a region (material, overrides, damping, PML settings).

    The damping coefficients not given to lp.Rayleigh / lp.Hysteretic come from the material's default
    fits (Material.rayleigh, Material.loss_factor).
    """
    from loudpy.model.damping import Hysteretic, Rayleigh, Undamped
    from loudpy.model.regions import PML, Air, Solid
    m = _material(region, lib)
    if isinstance(region, Solid):
        p = m.require("rho", "E", "nu")
        d = region.damping
        if isinstance(d, Hysteretic):
            p["eta"] = d.eta if d.eta is not None else m.require("loss_factor")["loss_factor"]
        elif isinstance(d, Undamped):
            p.update(alpha_ray=0.0, beta_ray=0.0)
        elif isinstance(d, Rayleigh):
            alpha, beta = d.alpha, d.beta
            if alpha is None or beta is None:
                fit = m.require("rayleigh")["rayleigh"]
                alpha, beta = (fit[0] if alpha is None else alpha), (fit[1] if beta is None else beta)
            p.update(alpha_ray=alpha, beta_ray=beta)
        else:
            raise ModelError(f"{region.name}: unknown damping {d!r}")
        return p
    if isinstance(region, PML):
        p = m.require("rho", "c")
        p["alpha"], p["n"] = region.strength, region.order
        if region.tuned_to is None:
            raise ModelError(f"PML {region.name!r}: give tuned_to= (frequency the absorption is tuned to)")
        p["f_pml"] = region.tuned_to
        return p
    if isinstance(region, Air):
        return m.require("rho", "c")
    raise ModelError(f"unknown region {region!r}")


def region_part(region, group, lib, layer=None) -> RegionPart:
    """The part of a region on one mesh group (a PML: with the geometry `layer` of its surface)."""
    from loudpy.model.regions import PML, Solid
    p = region_parameters(region, lib)
    if isinstance(region, Solid):
        return SolidPart(group, **p)
    if isinstance(region, PML):
        return PMLPart(group, **p, layer=layer)
    return AirPart(group, **p)


def boundary_part(boundary, group) -> EdgePart:
    """The part of a boundary on one mesh group (components from the boundary's parameters)."""
    from loudpy.model.boundaries import COMPONENTS, Clamped, Coil, FluidStructure, Radiation
    if isinstance(boundary, Clamped):
        return ClampedEdge(group, COMPONENTS[boundary.dofs])
    if isinstance(boundary, Coil):
        return CoilEdge(group, COMPONENTS[boundary.direction])
    if isinstance(boundary, FluidStructure):
        return CouplingEdge(group)
    if isinstance(boundary, Radiation):
        return RadiationEdge(group)
    raise ModelError(f"unknown boundary {boundary!r}")


def orient_couplings(parts) -> None:
    """Per-edge signs of the fluid–structure edges: the outward normal of the adjacent air (from the mesh)."""
    from loudpy.geometry.topology import EdgeSides
    airs = [p for p in parts if p.role in ("air", "pml")]
    couplings = [p for p in parts if p.role == "fluid_structure"]
    if not airs or not couplings:
        return
    sides = EdgeSides(airs)
    for p in couplings:
        rows = {int(t): i for i, t in enumerate(np.asarray(p.node_tags))}
        xy = np.asarray(p.node_coords)[:, :2]
        try:
            _, p.signs = sides.outward(p.edges, lambda t: xy[rows[int(t)]])
        except Exception as e:
            raise ModelError(f"fluid–structure boundary {p.name!r}: every edge must lie on exactly one air "
                             f"region ({e})") from None
