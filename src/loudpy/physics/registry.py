"""
Roles of the physics objects, and selection by role.

Every physics object (a mesh group with its physical parameters, see parts.py) declares one role.
Assemblers and studies select the objects they need by role, instead of testing classes:

    select(objects, "solid")            the structural regions
    select(objects, "air", "pml")       the acoustic regions, PML layers included
    select(objects, "clamped")          the blocked boundaries

A new kind of physics object only declares its role; nothing else has to change.
"""
from __future__ import annotations

ROLES = {
    "solid": "structural region (axisymmetric solid)",
    "air": "acoustic region",
    "pml": "perfectly matched layer (an acoustic region)",
    "clamped": "boundary with blocked displacement components",
    "coil": "boundary where the drive force acts",
    "fluid_structure": "structure ↔ air coupling boundary",
    "radiation": "exact non-reflecting (DtN) boundary",
}


def role_of(obj) -> str | None:
    return getattr(obj, "role", None)


def select(objects, *roles) -> list:
    """The objects whose role is one of `roles`, in their original order."""
    unknown = set(roles) - set(ROLES)
    if unknown:
        raise ValueError(f"unknown roles {sorted(unknown)}; roles: {sorted(ROLES)}")
    return [o for o in objects if role_of(o) in roles]
