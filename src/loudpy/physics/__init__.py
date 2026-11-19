"""
loudpy.physics — physics objects and the operators they contribute (no file I/O).

  registry      roles of the physics objects, select(objects, *roles)
  parts         SolidPart, AirPart, PMLPart, ClampedEdge, CoilEdge, CouplingEdge, RadiationEdge
  structural    MecaAssembler, MecaAssemblerFourier, MecaAssemblerNL     (role "solid")
  acoustic      AcouAssembler                                            (roles "air", "pml")
  coupling      FsiAssembler                                             (role "fluid_structure")
  forcing       ForceAssembler                                           (role "coil")
  radiation     DtN boundary operators on an arc / a sphere              (role "radiation")
"""
from loudpy.physics.registry import ROLES, role_of, select
from loudpy.physics.parts import (Part, SolidPart, AirPart, PMLPart, ClampedEdge, CoilEdge, CouplingEdge,
                                  RadiationEdge)

__all__ = ["ROLES", "role_of", "select", "Part", "SolidPart", "AirPart", "PMLPart", "ClampedEdge",
           "CoilEdge", "CouplingEdge", "RadiationEdge"]
