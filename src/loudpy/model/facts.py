"""
What an item type is, declared next to its class (loudpy.model.regions, boundaries, mesh_controls) and read by programs
— lp.suggest, lp.readiness, lp.schema and the interface's Physics tables — so that nothing outside the class needs its
name:

    lp.Solid.facts       Facts(selects="domains", medium="solid", physics="structure", key="material")
    lp.Clamped.facts     Facts(selects="curves", physics="structure", role="held", key="dofs",
                               fits=(Fit("solid|-", "a free edge of the structure: clamped where it is held"), …))
    item_types()         {type name: class} of every item type (the classes that declare facts, in order)
    facts_of(x)          the Facts of an item, an item class or a type name

Where a boundary type fits is a *place*, read from the geometry alone (never from a name): what lies on each side of
a curve (the media of the regions there, "-" for the outside), whether the curve is on the symmetry axis, and its
shape (a straight line r = const or z = const, an arc centred on the axis):

    "solid|air"    between the structure and the air          "air|-"        an outer edge of the air
    "solid|-"      a free edge of the structure               "solid|solid"  between two solid parts
    "air|air"      inside the air                             "solid|axis"   the structure on the symmetry axis
    "air|axis"     the air on the symmetry axis

A new item type is one class with its facts (a Region or a Boundary subclass) and its description for programs
(lp.schema.register): it then shows in the interface's tables, in lp.suggest and in lp.readiness.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

PLACES = {
    "solid|air": "between the structure and the air",
    "air|-": "an outer edge of the air",
    "solid|-": "a free edge of the structure",
    "solid|solid": "between two solid parts",
    "air|air": "inside the air",
    "solid|axis": "the structure on the symmetry axis",
    "air|axis": "the air on the symmetry axis (no condition needed)",
}
MEDIA = ("solid", "air")
SELECTS = ("domains", "curves")
ROLES = {"held": "holds the structure (where it is clamped)", "drive": "where the drive acts",
         "coupling": "couples the structure and the air", "absorbing": "lets the sound leave the air (no reflection)"}
PHYSICS = ("structure", "air", "coupling")
SHAPES = ("", "straight", "arc")


@dataclass(frozen=True)
class Fit:
    """Where an item type fits: a place (PLACES), why (said to the user), the curve shape it needs ("straight": a
    line r = const or z = const; "arc": a circular arc centred on the axis; "": any), the parameter values it takes
    there (e.g. {"dofs": "r"} on the axis), and whether it is the usual choice there (else only possible)."""
    place: str
    why: str
    shape: str = ""
    params: dict = field(default_factory=dict)
    usual: bool = True

    def __post_init__(self):
        if self.place not in PLACES:
            raise ValueError(f"Fit.place must be one of {list(PLACES)}, got {self.place!r}")
        if self.shape not in SHAPES:
            raise ValueError(f"Fit.shape must be one of {SHAPES}, got {self.shape!r}")


@dataclass(frozen=True)
class Facts:
    """What an item type is: what it selects (domain groups or curves), the medium a region fills, the physics
    interface it belongs to, its role for the checklist (ROLES), its key parameter (the one a table shows next to its
    type, e.g. a solid's material) and where it fits (Fit, for boundaries and layers grown from curves);
    starting: values a new item of this type starts with beyond its code defaults; table: shown in the Physics
    tables."""
    selects: str = "domains"
    medium: str | None = None
    physics: str | None = None
    role: str | None = None
    key: str | None = None
    fits: tuple[Fit, ...] = ()
    starting: dict = field(default_factory=dict)
    table: bool = True

    def __post_init__(self):
        if self.selects not in SELECTS:
            raise ValueError(f"Facts.selects must be one of {SELECTS}, got {self.selects!r}")
        if self.medium is not None and self.medium not in MEDIA:
            raise ValueError(f"Facts.medium must be one of {MEDIA} or None, got {self.medium!r}")
        if self.role is not None and self.role not in ROLES:
            raise ValueError(f"Facts.role must be one of {list(ROLES)} or None, got {self.role!r}")
        if self.physics is not None and self.physics not in PHYSICS:
            raise ValueError(f"Facts.physics must be one of {PHYSICS} or None, got {self.physics!r}")

    def fit(self, place: str, shape: str = "") -> Fit | None:
        """How it fits a place of that curve shape (None: it does not)."""
        for f in self.fits:
            if f.place == place and (not f.shape or f.shape == shape):
                return f
        return None


_TYPES: dict[str, Any] = {}             # {name: item class} (classes with a `facts`)


def register(cls) -> type:
    """Make an item class known by its name (done for the classes of loudpy; lp.schema.register does it for a new
    one). Returns the class."""
    facts = getattr(cls, "facts", None)
    if not isinstance(facts, Facts):
        raise ValueError(f"{cls.__name__} declares no facts (a class attribute `facts = Facts(…)`)")
    _TYPES[cls.__name__] = cls
    return cls


def unregister(name: str) -> None:
    """Forget an item type registered by name."""
    _TYPES.pop(name, None)


def item_types() -> dict[str, Any]:
    """{type name: class} of every item type: regions, then boundaries, then the others (mesh controls), each in the
    order they were made known."""
    from loudpy.model.boundaries import Boundary
    from loudpy.model.regions import Region

    def rank(cls) -> int:
        return 0 if issubclass(cls, Region) else 1 if issubclass(cls, Boundary) else 2
    return dict(sorted(_TYPES.items(), key=lambda kv: rank(kv[1])))


def facts_of(x) -> Facts:
    """The Facts of an item, an item class or a type name (KeyError for an unknown name)."""
    if isinstance(x, str):
        return item_types()[x].facts
    facts = getattr(x, "facts", None)
    if not isinstance(facts, Facts):
        raise KeyError(f"{type(x).__name__ if not isinstance(x, type) else x.__name__} declares no facts")
    return facts


def of_role(role: str) -> list[str]:
    """The type names whose role is `role`, in order."""
    return [n for n, c in item_types().items() if c.facts.role == role]


def names(types: list[str]) -> str:
    """"a PML or a Radiation", "a Solid, an Air or a PML": type names as a list in words."""
    words = [("an " if t[:1].lower() in "aeiou" else "a ") + t for t in types]
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " or " + words[-1] if words else ""
