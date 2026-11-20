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

Names are labels: the modifiers only matter for matching. No physical setting is read from a name
(clamped components, coil direction and orientations are parameters or come from the mesh).

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

# modifier tokens (matching only)
DIRECTIONS = {"r", "z", "rz", "zr"}
PML_DIRECTIONS = {"r+", "r-", "z+", "z-"}
SIDES = {"front", "rear"}
_DIGITS = re.compile(r"^\d+$")


def base_name(group: str) -> str:
    """The group name without the "__key=value" parts the mesher appends (PML layers)."""
    return group.strip().split("__")[0]


def source_group(mesh_group: str) -> str:
    """The geometry group a mesh group was built from: itself, or for the PML layers and corners made by
    the mesher ("PML_r+_1_front__r0=…", "PML_r+_1_front_Corner__…") the PML curve group."""
    base = base_name(mesh_group)
    return base[:-len("_Corner")] if base.endswith("_Corner") else base


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


# ════════════════════════════════════════════════════════════════════════════
#  PML layers: geometry of a layer, written by the mesher into its surface names
# ════════════════════════════════════════════════════════════════════════════
_FLOAT = r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"


@dataclass(frozen=True)
class PMLLayer:
    """
    Where a PML layer starts and how thick it is, along r and/or z (both for a corner):
    the absorption grows from r0 (z0) outward, in the direction sgn_r (sgn_z), over tr (tz).
    """
    use_r: bool = False
    r0: float = 0.0
    tr: float = 0.0
    sgn_r: float = 0.0
    use_z: bool = False
    z0: float = 0.0
    tz: float = 0.0
    sgn_z: float = 0.0


def pml_layer(group: str) -> PMLLayer | None:
    """The PMLLayer of a mesher PML surface name ("PML_r+_1_front__r0=…__t=…"), None if not a PML."""
    if "PML" not in group:
        return None

    def get(key):
        m = re.search(r"__" + key + r"=" + _FLOAT + r"(?:__|$)", group)
        return float(m.group(1)) if m else None

    kw = {}
    if "__r0=" in group:
        kw.update(use_r=True, r0=get("r0"), sgn_r=-1.0 if "r-" in group else 1.0,
                  tr=get("tr") or get("t") or 0.0)
    if "__z0=" in group:
        kw.update(use_z=True, z0=get("z0"), sgn_z=-1.0 if "z-" in group else 1.0,
                  tz=get("tz") or get("t") or 0.0)
    return PMLLayer(**kw)
