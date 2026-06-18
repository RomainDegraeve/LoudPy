from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar,  Optional
from loudpy.SubDomains import (
    SubDomainMeca, SubdomainMeca_Hysteretic, SubdomainMeca_Rayleigh,
    SubDomainAcou, SubDomainAcou_PML,
)
from loudpy.Interfaces import (
    InterfaceAcouMeca, InterfaceConstrainted, InterfaceForced,
)

# ── Domain specs ─────────────────────────────────────────────────────────────
@dataclass(eq=False)
class DomainSpec:
    name     : str 
    physics  : ClassVar[type]
    rho = None 
    material : str 

    def material_keys(self):
        keys = []
        for base in type(self).__mro__:
            for k, v in base.__dict__.items():
                if not k.startswith("_") and v is None and k not in keys:
                    keys.append(k)
        return keys


@dataclass(eq=False)
class DomainSpecMeca(DomainSpec):
    physics : ClassVar[type] = SubDomainMeca
    E : Optional[float] = None 
    size : float = 0.002
    
@dataclass(eq=False)
class DomainSpecMecaRayleigh(DomainSpec):
    physics : ClassVar[type] = SubdomainMeca_Rayleigh
    E : Optional[float] = None 
    alpha_ray : Optional[float]= None
    beta_ray : Optional[float]= None
    size : float = 0.002
    
@dataclass(eq=False)
class DomainSpecMecaHysteretic(DomainSpec):
    physics : ClassVar[type] = SubdomainMeca_Hysteretic
    E : Optional[float] = None 
    eta : Optional[float]= None
    size : float = 0.002

@dataclass(eq=False)
class DomainSpecAcou(DomainSpec):
    physics      : ClassVar[type] = SubDomainAcou
    c : Optional[float]= None
    size : float = 0.01
@dataclass(eq=False)
class DomainSpecPML(DomainSpec):
    physics      : ClassVar[type] = SubDomainAcou_PML
    t      : float = 1
    size : float = 0.01
    f_pml : Optional[float] = None
    c : Optional[float]= None
    alpha : Optional[float]= None
    n : Optional[float]= None
    

# ── Interface specs ──────────────────────────────────────────────────────────
@dataclass(eq=False)
class InterfaceSpec:
    name    : str
    physics : ClassVar[type]

@dataclass(eq=False)
class InterfaceSpecAcouMeca(InterfaceSpec):
    physics : ClassVar[type] = InterfaceAcouMeca

@dataclass(eq=False)
class InterfaceSpecClamped(InterfaceSpec):
    physics : ClassVar[type] = InterfaceConstrainted

@dataclass(eq=False)
class InterfaceSpecForced(InterfaceSpec):
    physics : ClassVar[type] = InterfaceForced


__all__ = [
    "DomainSpec", "DomainSpecMeca", "DomainSpecMecaRayleigh",
    "DomainSpecMecaHysteretic", "DomainSpecAcou", "DomainSpecPML",
    "InterfaceSpec", "InterfaceSpecAcouMeca",
    "InterfaceSpecClamped", "InterfaceSpecForced",
]