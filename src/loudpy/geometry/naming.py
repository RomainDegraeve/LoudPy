"""
Physical-group names of the loudpy .geo convention, and how names given by the user select them.

A group name is a sequence of tokens separated by "_":

    [physics prefix] base tokens [modifiers]          e.g.  submeca_glue_1
                                                            interface_constrained_rz_2
                                                            subacou_rear
                                                            PML_r+_1_front   (+ "__..." added by the mesher)

  physics prefix : "submeca", "subacou" (any first token starting with "sub")
  modifiers      : an index (digits), a direction (r, z, rz, zr, r+, r-, z+, z-),
                   a side (front, rear), "Corner" (PML corners built by the mesher)

A selector matches a group when its tokens appear whole, either at the start of the group or
right after a physics prefix, and every token after them is a modifier:

    "membranne"             → submeca_membranne
    "glue"                  → submeca_glue_1, submeca_glue_2
    "coil"                  → submeca_coil            (not submeca_coil_former)
    "interface_constrained" → interface_constrained_r_1, interface_constrained_rz_1, …
    "PML"                   → every PML layer and corner

    "front"                 → subacou_front           (base tokens after the prefix)

A selector with a wildcard ("*", "?", "[") is a glob on the group name, with or without its
physics prefix ("glue_*" → submeca_glue_1, submeca_glue_12).
"""
from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass

DIRECTIONS = {"r": (0,), "z": (1,), "rz": (0, 1), "zr": (0, 1)}
PML_DIRECTIONS = {"r+", "r-", "z+", "z-"}
SIDES = {"front": +1.0, "rear": -1.0}
_DIGITS = re.compile(r"^\d+$")


def base_name(group: str) -> str:
    """The group name without the "__key=value" parts the mesher appends (PML layers)."""
    return group.strip().split("__")[0]


def tokens(name: str) -> list[str]:
    return [t for t in base_name(name).split("_") if t]


def is_modifier(token: str) -> bool:
    return (bool(_DIGITS.match(token)) or token in DIRECTIONS or token in PML_DIRECTIONS
            or token in SIDES or token == "Corner")


def is_prefix(token: str) -> bool:
    return token.startswith("sub")


def matches(selector: str, group: str) -> bool:
    """Does the user name `selector` select the physical group `group` (rules above)?"""
    if any(c in selector for c in "*?["):
        base = base_name(group)
        g = tokens(base)
        bare = "_".join(g[1:]) if g and is_prefix(g[0]) else base        # without "submeca_" etc.
        return fnmatch.fnmatchcase(base, selector) or fnmatch.fnmatchcase(bare, selector)
    s, g = tokens(selector), tokens(group)
    if not s:
        return False
    for start in (0, 1):
        if start == 1 and not (g and is_prefix(g[0])):
            continue
        if g[start:start + len(s)] == s and all(is_modifier(t) for t in g[start + len(s):]):
            return True
    return False


@dataclass(frozen=True)
class GroupInfo:
    """What a group name says beyond its base: index, direction, side."""
    name: str
    index: int | None
    direction: str | None          # "r", "z", "rz", "zr" (constrained / forced DOFs)
    side: str | None               # "front" / "rear"

    @property
    def dofs(self):
        """Constrained DOF indices (0 = r, 1 = z) from the direction token, or None."""
        return DIRECTIONS.get(self.direction) if self.direction else None

    @property
    def orientation(self):
        """+1 (front) / −1 (rear), or None."""
        return SIDES.get(self.side) if self.side else None


def parse(group: str) -> GroupInfo:
    toks = tokens(group)
    index = next((int(t) for t in toks if _DIGITS.match(t)), None)
    direction = next((t for t in toks if t in DIRECTIONS), None)
    side = next((t for t in toks if t in SIDES), None)
    return GroupInfo(group, index, direction, side)
