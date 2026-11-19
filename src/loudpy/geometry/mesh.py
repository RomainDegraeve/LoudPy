"""
Mesh: one data model for 2D (axisymmetric meridian) and 3D meshes.

    nodes   : node_tags (n,), coords (n, dim)
    groups  : {name: MeshGroup(name, dim, cell_type, cells)} — cells hold node tags

The 2D mesher (meshing2d) and the 3D builders (meshing3d) both produce it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CELL_TYPES = {"line3": (1, 3), "tri3": (2, 3), "tri6": (2, 6), "tet10": (3, 10)}   # name: (dim, nodes per cell)


@dataclass(frozen=True)
class MeshGroup:
    name: str
    dim: int
    cell_type: str
    cells: np.ndarray                  # (n_cells, nodes per cell) node tags

    @property
    def node_tags(self) -> np.ndarray:
        return np.unique(self.cells)

    def __len__(self):
        return len(self.cells)


@dataclass
class Mesh:
    node_tags: np.ndarray              # (n,)
    coords: np.ndarray                 # (n, dim)
    groups: dict[str, MeshGroup] = field(default_factory=dict)
    path: str | None = None            # file it was read from / written to

    @property
    def dim(self) -> int:
        return self.coords.shape[1]

    @property
    def n_nodes(self) -> int:
        return len(self.node_tags)

    def names(self, dim: int | None = None) -> list[str]:
        return [n for n, g in self.groups.items() if dim is None or g.dim == dim]

    def rows_of(self, tags) -> np.ndarray:
        """Row indices (into node_tags / coords) of node tags."""
        pos = self._positions
        return np.array([pos[int(t)] for t in np.asarray(tags).ravel()], dtype=np.int64).reshape(np.shape(tags))

    def triangles(self, groups=None) -> np.ndarray:
        """(n, 3) corner triangles (row indices) of the 2D groups (all, or the named ones), for plotting."""
        names = [n for n in (groups or self.names(2)) if self.groups[n].dim == 2]
        if not names:
            return np.zeros((0, 3), np.int64)
        return np.vstack([self.rows_of(self.groups[n].cells[:, :3]) for n in names])

    def nearest(self, point) -> int:
        """Row of the node nearest to `point`."""
        return int(np.argmin(np.linalg.norm(self.coords - np.asarray(point, float)[:self.dim], axis=1)))

    def coords_of(self, tags) -> np.ndarray:
        """Coordinates of node tags."""
        pos = self._positions
        return self.coords[[pos[int(t)] for t in np.asarray(tags).ravel()]].reshape(
            np.shape(tags) + (self.dim,))

    @property
    def _positions(self) -> dict[int, int]:
        if not hasattr(self, "_pos_cache"):
            object.__setattr__(self, "_pos_cache", {int(t): i for i, t in enumerate(self.node_tags)})
        return self._pos_cache

    def __repr__(self):
        by_dim = {d: len(self.names(d)) for d in sorted({g.dim for g in self.groups.values()})}
        return f"Mesh({self.n_nodes} nodes, dim {self.dim}, groups per dim {by_dim}, path={self.path!r})"

    # ── builders ───────────────────────────────────────────────────────────
    @classmethod
    def from_axisymmetric(cls, domains, boundaries, nodes, path=None) -> "Mesh":
        """From the 2D mesher output (TR6_Surfs domains, L3_Edges boundaries, Nodes)."""
        groups = {}
        for d in domains:
            groups[d.name] = MeshGroup(d.name, 2, "tri6", np.asarray(d.tri))
        for b in boundaries:
            groups[b.name] = MeshGroup(b.name, 1, "line3", np.asarray(b.edges))
        coords = np.asarray(nodes.node_coords).reshape(-1, 3)[:, :2]
        return cls(np.asarray(nodes.node_tags), coords, groups, path)
