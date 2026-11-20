"""
Function spaces: DOFs numbered from the nodes of a high-order mesh.

A `Space` takes the mesh nodes (tags, coordinates), the cells of one or more groups (node tags, gmsh
order) and a Lagrange element whose layout matches the cells. Its DOFs are the sorted unique node tags
of those cells, with `components` interleaved DOFs per node (u_r, u_z, … of node 0, then node 1, …), the
numbering of the former assemblers, so operators can be compared entry by entry. The geometry map is
isoparametric: each cell's own nodes (curved when gmsh placed the high-order nodes on curves).

    space = Space(node_tags, coords, [cells_a, cells_b], lagrange("triangle", 2))
    space.cell_dofs[0]      # (n_e, n · components) DOFs of group a's cells
    space.cell_coords[0]    # (n_e, n, gdim) their node coordinates
"""
from __future__ import annotations

import numpy as np

from loudpy.fem.reference import LagrangeElement, lagrange

__all__ = ["Space", "MESH_CELL_ELEMENTS"]

# loudpy.geometry.mesh cell type names → (reference cell, order)
MESH_CELL_ELEMENTS = {"line3": ("line", 2), "tri3": ("triangle", 1), "tri6": ("triangle", 2),
                      "tet10": ("tetrahedron", 2)}


class Space:
    """Lagrange space over the cells of some mesh groups (see the module docstring)."""

    def __init__(self, node_tags, coords, cells, element: LagrangeElement, components: int = 1):
        node_tags = np.asarray(node_tags).ravel()
        coords = np.asarray(coords, dtype=float)
        if coords.ndim != 2 or len(coords) != len(node_tags):
            raise ValueError("coords must be (n_nodes, gdim), one row per node tag")
        if isinstance(cells, np.ndarray) and cells.ndim == 2:
            cells = [cells]
        cells = [np.asarray(c, dtype=np.int64) for c in cells]
        for c in cells:
            if c.ndim != 2 or c.shape[1] != element.n_nodes:
                raise ValueError(f"cells of shape {c.shape} do not match {element} ({element.n_nodes} nodes)")
        if components < 1:
            raise ValueError("components must be ≥ 1")
        self.element = element
        self.components = int(components)
        self.gdim = coords.shape[1]
        self.tags = np.unique(np.concatenate([c.ravel() for c in cells])) if cells else np.zeros(0, np.int64)

        order = np.argsort(node_tags, kind="stable")
        sorted_tags = node_tags[order]
        pos = np.searchsorted(sorted_tags, self.tags)
        if np.any(pos >= len(sorted_tags)) or np.any(sorted_tags[np.minimum(pos, len(sorted_tags) - 1)] != self.tags):
            raise ValueError("cells refer to node tags that are not in node_tags")
        self.node_coords = np.ascontiguousarray(coords[order[pos]])       # (n_space_nodes, gdim)

        comp = np.arange(self.components)
        self.cell_nodes, self.cell_dofs, self.cell_coords = [], [], []
        for c in cells:
            local = np.searchsorted(self.tags, c)                          # (n_e, n) space node index
            self.cell_nodes.append(local)
            self.cell_dofs.append(np.ascontiguousarray(
                (local[..., None] * self.components + comp).reshape(len(c), -1).astype(np.int32)))
            self.cell_coords.append(np.ascontiguousarray(self.node_coords[local]))

    @classmethod
    def from_mesh(cls, mesh, groups, components: int = 1) -> "Space":
        """Space over named groups of a `loudpy.geometry.mesh.Mesh` (one cell type)."""
        names = [groups] if isinstance(groups, str) else list(groups)
        types = {mesh.groups[n].cell_type for n in names}
        if len(types) != 1:
            raise ValueError(f"groups {names} mix cell types {sorted(types)}")
        element = lagrange(*MESH_CELL_ELEMENTS[types.pop()])
        return cls(mesh.node_tags, mesh.coords, [mesh.groups[n].cells for n in names], element, components)

    @property
    def n_nodes(self) -> int:
        return len(self.tags)

    @property
    def n_dofs(self) -> int:
        return self.n_nodes * self.components

    @property
    def n_cells(self) -> int:
        return sum(len(c) for c in self.cell_nodes)

    def dofs_of(self, tags, component: int | None = None) -> np.ndarray:
        """DOFs of node tags (all components interleaved, or one component)."""
        idx = np.searchsorted(self.tags, np.asarray(tags, dtype=np.int64).ravel())
        if np.any(idx >= self.n_nodes) or np.any(self.tags[np.minimum(idx, self.n_nodes - 1)] != np.ravel(tags)):
            raise ValueError("node tags not in this space")
        if component is not None:
            return idx * self.components + component
        return (idx[:, None] * self.components + np.arange(self.components)).ravel()

    def interpolate(self, f) -> np.ndarray:
        """Nodal values of f (points (n, gdim) → (n,) or (n, components)), in DOF order."""
        return np.asarray(f(self.node_coords), dtype=float).reshape(self.n_dofs)

    def __repr__(self):
        return (f"Space({self.element!r}, {self.n_nodes} nodes × {self.components}, "
                f"{self.n_cells} cells in {len(self.cell_nodes)} groups)")
