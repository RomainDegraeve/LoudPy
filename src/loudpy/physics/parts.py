"""
Physics objects: one mesh group (composition, not inheritance) with its physical parameters
and its role (registry.py).

    regions     SolidPart (solid), AirPart (air), PMLPart (pml)
    boundaries  ClampedEdge (clamped), CoilEdge (coil), CouplingEdge (fluid_structure),
                RadiationEdge (radiation)

The mesh group is any object with node_tags, node_coords, a name and its cells (tri for a
region, edges for a boundary); the parts expose them unchanged for the assemblers.

The classes of loudpy.SubDomains / loudpy.Interfaces derive from these with their former
names and constructors (the names are written in the result files until phase 4).
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


def material_parameters(cls, mat_name: str, mat_path) -> dict:
    """Material properties of `mat_name` (JSON library) that the constructor of `cls` accepts."""
    import inspect
    import json
    with open(mat_path, "r") as f:
        data = json.load(f)[mat_name]
    sig = inspect.signature(cls).parameters
    return {k: v for k, v in data.items() if k in sig}
