"""
Mesh controls by selection (COMSOL's "Size" nodes): the element size of the domains a selector selects,
next to the per-item mesh_size.

    model.add(lp.MeshSize(on=lp.Group("surround", "spider"), size=0.3e-3))
    model.add(lp.MeshSize(on=lp.Box(r=(0.0, 0.02)), size=1e-3, name="near the axis"))

A control overrides the mesh_size of the regions for the domain groups it selects; when several controls
select a group, the last one added wins. Every control must select at least one meshed domain group.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from loudpy.checks import check
from loudpy.geometry.selectors import Selector, as_selector


@dataclass
class MeshSize:
    """Element size [m] of the domain groups selected by `on` (overrides the regions' mesh_size)."""
    on: Selector | str
    size: float
    name: str | None = field(default=None, kw_only=True)

    def __post_init__(self):
        check(self.size, "mesh_size", f"MeshSize({self.label!r}).size")

    @property
    def selector(self) -> Selector:
        return as_selector(self.on)

    @property
    def label(self) -> str:
        return self.name if self.name is not None else repr(self.on)
