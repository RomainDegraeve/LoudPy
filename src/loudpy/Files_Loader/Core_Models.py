from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from typing import ClassVar, Iterator
import matplotlib.tri as mtri
import numpy as np

from loudpy.Files_Loader.Field_Operations_Helpers import (
    infer_dpn, scatter_dofs, reduce_field
)

@dataclass(frozen=True)
class Mesh:
    tags:   np.ndarray   # (N,) global node tags
    coords: np.ndarray   # (N, 2)
    tris:   np.ndarray   # (M, 3) row indices

    @property
    def n_nodes(self) -> int: return len(self.tags)
    @property
    def n_tris(self)  -> int: return len(self.tris)

    @cached_property
    def tag_to_row(self) -> dict[int, int]:
        return {int(t): i for i, t in enumerate(self.tags)}

    @cached_property
    def triangulation(self) -> mtri.Triangulation:
        return mtri.Triangulation(self.coords[:, 0], self.coords[:, 1], self.tris)

    def nearest_node(self, xy) -> tuple[int, np.ndarray]:
        xy = np.asarray(xy, dtype=float)
        idx = int(np.argmin(np.sum((self.coords - xy) ** 2, axis=1)))
        return idx, self.coords[idx]


@dataclass
class Snapshot(ABC):
    """Any field result on a mesh (freq / mode / time)."""
    label:    str
    fields:   dict[str, np.ndarray]
    dof_maps: dict[str, np.ndarray]
    mesh_id:  str
    meta:     dict = field(default_factory=dict)

    @abstractmethod
    def coordinate(self) -> float: ...

    def nodal(self, name: str, mesh: Mesh, *,
              component: str = "mag", dpn: int | None = None) -> np.ndarray:
        dpn = dpn or infer_dpn(name)
        raw = self.fields[name]
        arr = scatter_dofs(raw, self.dof_maps.get(name), mesh, dpn)
        return reduce_field(arr, component)

    def complex_nodal(self, name: str, mesh: Mesh,
                      dpn: int | None = None) -> np.ndarray:
        dpn = dpn or infer_dpn(name)
        return scatter_dofs(self.fields[name],
                            self.dof_maps.get(name), mesh, dpn)


@dataclass
class FreqSnapshot(Snapshot):
    f: float = 0.0
    def coordinate(self) -> float: return self.f


@dataclass
class ModeSnapshot(Snapshot):
    freq:  float = 0.0
    zeta:  float | None = None
    index: int = 0
    def coordinate(self) -> float: return self.freq


@dataclass
class TimeSnapshot(Snapshot):
    t: float = 0.0
    def coordinate(self) -> float: return self.t


@dataclass
class TimeRun:
    """Bundles all arrays of a single transient run."""
    time:     np.ndarray
    U:        np.ndarray
    V:        np.ndarray
    A:        np.ndarray
    dof_maps: dict[str, np.ndarray]
    mesh_id:  str

    _FIELD_MAP: ClassVar[dict[str, str]] = {
        "u_meca": "U", "v_meca": "V", "a_meca": "A",
    }

    def array_for(self, field_name: str) -> np.ndarray:
        return getattr(self, self._FIELD_MAP.get(field_name, "U"))

    def snapshot_at(self, idx: int, field_name: str = "u_meca") -> TimeSnapshot:
        arr = self.array_for(field_name)
        return TimeSnapshot(
            label    = f"t={self.time[idx]*1e3:.3f}ms",
            fields   = {field_name: arr[idx]},
            dof_maps = self.dof_maps,
            mesh_id  = self.mesh_id,
            t        = float(self.time[idx]),
        )

    def iter_snapshots(self, field_name: str = "u_meca") -> Iterator[TimeSnapshot]:
        for i in range(len(self.time)):
            yield self.snapshot_at(i, field_name)
