"""
loudpy.physics — physics objects and the operators they contribute (no file I/O).

  registry      roles of the physics objects, select(objects, *roles)
  parts         SolidPart, AirPart, PMLPart, ClampedEdge, CoilEdge, CouplingEdge, RadiationEdge
  structural    MecaAssembler, MecaAssemblerFourier, MecaAssemblerNL     (role "solid")
  acoustic      AcouAssembler                                            (roles "air", "pml")
  coupling      FsiAssembler                                             (role "fluid_structure")
  forcing       ForceAssembler                                           (role "coil")
  radiation     DtN boundary operators on an arc / a sphere              (role "radiation")
  builder       SystemBuilder: fields, terms (functions of s, LowRank blocks), loads; contributions by role
                (frequency and symmetric forms); BlockSolver (eliminate small fields, GMRES on the big one)
  systems       assembled operators of a model: structure, + air, Fourier order m, acoustic, DtN system
  modal         3D cavity modes, driver modes, driver ↔ cavity modal coupling, add_driver_modes (scenes)
  exterior      half-space DtN on a hemisphere, exterior FE, add_exterior, scene_builder, coupled sweeps,
                1D port (scenes)
"""
from loudpy.physics.registry import ROLES, role_of, select
from loudpy.physics.parts import (Part, SolidPart, AirPart, PMLPart, ClampedEdge, CoilEdge, CouplingEdge,
                                  RadiationEdge)

__all__ = ["ROLES", "role_of", "select", "Part", "SolidPart", "AirPart", "PMLPart", "ClampedEdge",
           "CoilEdge", "CouplingEdge", "RadiationEdge"]
