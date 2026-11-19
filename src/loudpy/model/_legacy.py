"""
Bridge from the new model description to the current physics objects and studies.

Until the physics layer and the studies are rebuilt (refactor phases 2–5), a Model:
  - builds the same physics objects as PhysicalEntitiesManager (SubDomain*, Interface*), but
    with exact group selection (loudpy.geometry.naming) and an explicit parameter mapping;
  - translates itself to the former specs (DomainSpec*, InterfaceSpec*), which the mesher
    signature and the studies (interfaces, saved specs) still read.
This module disappears with the old studies.
"""
from __future__ import annotations

from loudpy.geometry import naming
from loudpy.model.boundaries import Boundary, Clamped, Coil, FluidStructure, Radiation
from loudpy.model.damping import Hysteretic, Rayleigh, Undamped
from loudpy.model.materials import Material, MaterialLibrary
from loudpy.model.regions import PML, Air, Region, Solid


# ════════════════════════════════════════════════════════════════════════════
#  Resolved material properties of a region
# ════════════════════════════════════════════════════════════════════════════
def _material(region, lib: MaterialLibrary) -> Material:
    m = lib.resolve(region.material) if isinstance(region.material, Material) or region.material in lib \
        else Material(str(region.material))
    return m.with_(**{k: getattr(region, k, None) for k in ("rho", "E", "nu", "c")})


def region_parameters(region: Region, lib: MaterialLibrary) -> dict:
    """Constructor parameters of the physics object of a region (the same values as before)."""
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


# ════════════════════════════════════════════════════════════════════════════
#  Physics objects (what the assemblers consume today)
# ════════════════════════════════════════════════════════════════════════════
def region_object(region: Region, group, lib: MaterialLibrary):
    from loudpy.SubDomains import (SubdomainMeca_Rayleigh, SubdomainMeca_Hysteretic, SubDomainAcou,
                                   SubDomainAcou_PML)
    p = region_parameters(region, lib)
    orientation = naming.parse(group.name).orientation
    if isinstance(region, Solid):
        cls = SubdomainMeca_Hysteretic if isinstance(region.damping, Hysteretic) else SubdomainMeca_Rayleigh
        return cls(group, **p)
    if isinstance(region, PML):
        return SubDomainAcou_PML(group, orientation=1.0 if orientation is None else orientation, **p)
    if orientation is not None:
        return SubDomainAcou(group, orientation=orientation, **p)
    return SubDomainAcou(group, **p)


def boundary_value(boundary: Boundary, group_name: str):
    """DOFs (blocked / forced) or orientation (±1) of a boundary group, as the old name rules."""
    if isinstance(boundary, Clamped) and boundary.dofs is not None:
        return naming.DIRECTIONS[boundary.dofs]
    if isinstance(boundary, (FluidStructure, Radiation)) and boundary.side is not None:
        return naming.SIDES[boundary.side]
    info = naming.parse(group_name)
    return info.dofs if info.dofs is not None else info.orientation


def boundary_object(boundary: Boundary, group):
    from loudpy.Interfaces import InterfaceAcouMeca, InterfaceConstrained, InterfaceForced, InterfaceDtN
    cls = {Clamped: InterfaceConstrained, Coil: InterfaceForced, FluidStructure: InterfaceAcouMeca,
           Radiation: InterfaceDtN}[type(boundary)]
    value = boundary_value(boundary, group.name)
    if value is None:
        if cls is InterfaceDtN:
            return cls(group)
        raise ValueError(f"{type(boundary).__name__}({boundary.name!r}): group {group.name!r} names no "
                         f"direction (_r_, _z_, _rz_) nor side (front / rear); give dofs= or side=")
    return cls(group, value)


# ════════════════════════════════════════════════════════════════════════════
#  Former specs (mesher signature, studies)
# ════════════════════════════════════════════════════════════════════════════
def region_spec(region: Region, lib: MaterialLibrary):
    from loudpy.SimObjects import DomainSpecMecaRayleigh, DomainSpecMecaHysteretic, DomainSpecAcou, DomainSpecPML
    mat = region.material.name if isinstance(region.material, Material) else region.material
    if isinstance(region, Solid):
        if isinstance(region.damping, Hysteretic):
            return DomainSpecMecaHysteretic(region.name, material=mat, size=region.mesh_size)
        return DomainSpecMecaRayleigh(region.name, material=mat, size=region.mesh_size)
    if isinstance(region, PML):
        s = DomainSpecPML(region.name, mat)
        s.t, s.size, s.f_pml, s.n, s.alpha = region.thickness, region.mesh_size, region.tuned_to, \
            region.order, region.strength
        return s
    return DomainSpecAcou(region.name, mat, size=region.mesh_size)


def boundary_spec(boundary: Boundary):
    from loudpy.SimObjects import InterfaceSpecClamped, InterfaceSpecForced, InterfaceSpecAcouMeca, InterfaceSpecDtN
    cls = {Clamped: InterfaceSpecClamped, Coil: InterfaceSpecForced, FluidStructure: InterfaceSpecAcouMeca,
           Radiation: InterfaceSpecDtN}[type(boundary)]
    return cls(boundary.name)
