"""
loudpy — fluid–structure finite elements for loudspeakers.

    import loudpy as lp
    model = lp.Model("speaker.geo")
    model.add(lp.Solid("membranne", "Paper"), lp.Clamped("interface_constrained"), lp.Coil("interface_forced"))
    model.mesh("mesh.msh")

The former API (Problem, DomainSpec*, InterfaceSpec*, loudpy.Studies) keeps working while the
refactor proceeds (docs/ARCHITECTURE.md).
"""

# ── new API ──────────────────────────────────────────────────────────────────
from .model import (
    Model, Solid, Air, PML, Clamped, Coil, FluidStructure, Radiation,
    Rayleigh, Hysteretic, Undamped, Force, Voltage,
    Material, MaterialLibrary, load_materials, materials,
)
from .geometry import Frame, Mesh

# ── former API ───────────────────────────────────────────────────────────────
from .fem.shape import fct_form_L3, fct_form_T6, jacobian_L3, jacobian_T6, material_matrix_D
from .fem.quadrature import GAUSS_L3_3pts, GAUSS_TR6_7pts
from .Meshing.MeshManager import MeshManager
from .PhysicalEntitiesManager import PhysicalEntitiesManager
from .SimObjects import (DomainSpec, DomainSpecAcou, DomainSpecMeca, DomainSpecMecaHysteretic,
                         DomainSpecMecaRayleigh, DomainSpecPML, InterfaceSpec, InterfaceSpecAcouMeca,
                         InterfaceSpecClamped, InterfaceSpecForced, InterfaceSpecDtN)


__all__ = [
    # new API
    "Model", "Solid", "Air", "PML", "Clamped", "Coil", "FluidStructure", "Radiation",
    "Rayleigh", "Hysteretic", "Undamped", "Force", "Voltage",
    "Material", "MaterialLibrary", "load_materials", "materials", "Frame", "Mesh",
    # former API
    "PhysicalEntitiesManager", "MeshManager",
    "DomainSpec", "DomainSpecMeca", "DomainSpecMecaRayleigh",
    "DomainSpecMecaHysteretic", "DomainSpecAcou", "DomainSpecPML",
    "InterfaceSpec", "InterfaceSpecAcouMeca",
    "InterfaceSpecClamped", "InterfaceSpecForced", "InterfaceSpecDtN",
    "GAUSS_L3_3pts", "GAUSS_TR6_7pts",
    "fct_form_L3", "fct_form_T6", "jacobian_L3", "jacobian_T6", "material_matrix_D",
]
