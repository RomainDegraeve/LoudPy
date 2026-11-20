"""
Mesh: one data model for 2D (axisymmetric meridian) and 3D meshes.

    nodes   : node_tags (n,), coords (n, dim)
    groups  : {name: MeshGroup(name, dim, cell_type, cells)} — cells hold node tags

The 2D mesher (meshing2d) and the 3D builders (meshing3d) both produce it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CELL_TYPES = {"line2": (1, 2), "line3": (1, 3), "line4": (1, 4), "tri3": (2, 3), "tri6": (2, 6),
              "tri10": (2, 10), "quad4": (2, 4), "quad9": (2, 9), "tet4": (3, 4), "tet10": (3, 10),
              "tet20": (3, 20)}                                     # name: (dim, nodes per cell)
_SUB_TRI6 = np.array([[0, 3, 5], [3, 1, 4], [5, 4, 2], [3, 4, 5]])   # gmsh TRI6 → 4 linear triangles
# linear sub-triangles through every node (plotting / interpolation), gmsh node layouts
_SUB_CELLS = {
    6: _SUB_TRI6,
    10: np.array([[0, 3, 8], [3, 4, 9], [3, 9, 8], [4, 1, 5], [4, 5, 9], [8, 9, 7], [9, 6, 7], [9, 5, 6],
                  [7, 6, 2]]),                                       # TRI10 → 9
    4: np.array([[0, 1, 2], [0, 2, 3]]),                             # QUAD4 → 2
    9: np.array([[0, 4, 8], [0, 8, 7], [4, 1, 5], [4, 5, 8], [7, 8, 6], [7, 6, 3], [8, 5, 2],
                 [8, 2, 6]]),                                        # QUAD9 → 8
}
_CORNERS = {3: 3, 6: 3, 10: 3, 4: 4, 9: 4}


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

    def triangles(self, groups=None, refine: bool = False) -> np.ndarray:
        """
        (n, 3) linear triangles (row indices) of the 2D groups (all, or the named ones), for plotting
        and interpolation: the corner triangles, or with refine=True the 4 sub-triangles of each
        6-node triangle through its mid-side nodes (smoother, uses every node).
        """
        names = [n for n in (groups or self.names(2)) if self.groups[n].dim == 2]
        if not names:
            return np.zeros((0, 3), np.int64)
        out = []
        for n in names:
            cells = self.groups[n].cells
            k = cells.shape[1]
            if refine and k in _SUB_CELLS:
                cells = cells[:, _SUB_CELLS[k]].reshape(-1, 3)
            elif _CORNERS.get(k) == 4:                                 # quadrilaterals: 2 triangles
                cells = cells[:, _SUB_CELLS[4]].reshape(-1, 3)
            out.append(self.rows_of(cells[:, :3]))
        return np.vstack(out)

    def interpolation(self, points, gradient: bool = False):
        """
        Sparse (n_points, n_nodes) matrix W with values(points) = W @ nodal values, exact for the mesh's own
        elements: each point is located in a cell of the mesh's dimension (KD-tree on the cell centres, then
        the reference coordinates by Newton on the isoparametric map, curved cells included) and weighted by
        that cell's Lagrange basis (P1–P3 triangles, Q1–Q2 quadrilaterals, P1–P3 tetrahedra). A point in no
        cell takes the value of the nearest node.

        gradient=True: also the gradient matrices [W_0, …, W_dim-1] (∂/∂r, ∂/∂z in 2D; ∂/∂x, ∂/∂y, ∂/∂z in 3D)
        of the field in the cell containing each point (zero for points in no cell).
        """
        import scipy.sparse as sp
        pts = np.atleast_2d(np.asarray(points, float))[:, :self.dim]
        n = len(pts)
        rows, cols, vals = [], [], []
        found = np.zeros(n, bool)
        grad = [([], [], []) for _ in range(self.dim)] if gradient else None
        if self.n_nodes and not gradient:                  # a point on a node: that node's value, exactly
            from scipy.spatial import cKDTree
            d, near = cKDTree(self.coords).query(pts)
            on_node = d <= 1e-12 * max(self._scale, 1e-300)
            rows += list(np.flatnonzero(on_node))
            cols += list(near[on_node])
            vals += [1.0] * int(on_node.sum())
            found |= on_node
        by_nodes: dict[int, list] = {}
        for g in self.groups.values():
            if g.dim == self.dim and len(g):
                by_nodes.setdefault(np.asarray(g.cells).shape[1], []).append(np.asarray(g.cells))
        for k, lst in by_nodes.items():
            todo = np.flatnonzero(~found)
            if not len(todo):
                break
            R = self._rows(np.vstack(lst))
            cell, W, G = _locate(self.coords[R], self.dim, k, pts[todo], self._scale)
            for j, (i, c, w) in enumerate(zip(todo, cell, W)):
                if c >= 0:
                    rows += [i] * k
                    cols += list(R[c])
                    vals += list(w)
                    found[i] = True
                    if grad is not None:
                        for d in range(self.dim):
                            grad[d][0].extend([i] * k)
                            grad[d][1].extend(R[c])
                            grad[d][2].extend(G[j, :, d])
        for i in np.flatnonzero(~found):
            rows.append(i)
            cols.append(self.nearest(pts[i]))
            vals.append(1.0)
        W = sp.csr_matrix((vals, (rows, cols)), shape=(n, self.n_nodes))
        if grad is None:
            return W
        return W, [sp.csr_matrix((v, (r, c)), shape=(n, self.n_nodes)) for r, c, v in grad]

    @property
    def _scale(self) -> float:
        return float(np.linalg.norm(np.ptp(self.coords, axis=0))) if self.n_nodes else 1.0

    def _rows(self, tags) -> np.ndarray:
        """Rows of node tags (vectorised rows_of)."""
        tags = np.asarray(tags)
        if not hasattr(self, "_sorted_tags"):
            order = np.argsort(self.node_tags, kind="stable")
            object.__setattr__(self, "_sorted_tags", (order, np.asarray(self.node_tags)[order]))
        order, st = self._sorted_tags
        idx = np.searchsorted(st, tags.ravel())
        return order[np.clip(idx, 0, len(st) - 1)].reshape(tags.shape)

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
        from loudpy.fem.reference import cell_type_name
        groups = {}
        for d in domains:
            cells = np.asarray(d.tri)
            groups[d.name] = MeshGroup(d.name, 2, cell_type_name(2, cells.shape[1]), cells)
        for b in boundaries:
            cells = np.asarray(b.edges)
            groups[b.name] = MeshGroup(b.name, 1, cell_type_name(1, cells.shape[1]), cells)
        coords = np.asarray(nodes.node_coords).reshape(-1, 3)[:, :2]
        return cls(np.asarray(nodes.node_tags), coords, groups, path)


def _locate(X, dim, n_nodes, pts, scale, n_candidates=None, tol=1e-9):
    """
    For each point: the index of the cell (rows of X: (n_cells, n_nodes, dim) node coordinates) containing it,
    −1 if none, the basis values there (n_points, n_nodes) and their physical gradients (n_points, n_nodes,
    dim). Candidates: the cells with the nearest centres; reference coordinates by Newton on the
    isoparametric map.
    """
    from scipy.spatial import cKDTree
    from loudpy.fem.reference import element_for_cells
    elem = element_for_cells(dim, n_nodes)
    ref = elem.cell
    nv = len(ref.vertices)
    kk = min(n_candidates or (16 if dim == 2 else 32), len(X))
    _, cand = cKDTree(X[:, :nv].mean(axis=1)).query(pts, k=kk)
    cand = np.asarray(cand).reshape(len(pts), kk)
    P = np.repeat(pts, kk, axis=0)
    Xc = X[cand.ravel()]                                               # (m, n_nodes, dim)
    xi = np.repeat(ref.vertices.mean(axis=0)[None], len(P), axis=0)
    lo, hi = ref.vertices.min(axis=0) - 1.0, ref.vertices.max(axis=0) + 1.0
    for _ in range(30):
        N, dN = elem.tabulate(xi)
        r = P - np.einsum("mk,mkd->md", N, Xc)
        A = np.einsum("mkj,mki->mji", Xc, dN)                          # ∂x_j / ∂ξ_i
        det = np.linalg.det(A)
        A[np.abs(det) < 1e-300] = np.eye(dim)
        step = np.linalg.solve(A, r[..., None])[..., 0]
        xi = np.clip(xi + step, lo, hi)
        if np.abs(step).max() < 1e-13:
            break
    N, dN = elem.tabulate(xi)
    resid = np.linalg.norm(P - np.einsum("mk,mkd->md", N, Xc), axis=1)
    if ref.simplex:
        inside = (xi >= -tol).all(axis=1) & (xi.sum(axis=1) <= 1.0 + tol)
    else:
        inside = (np.abs(xi) <= 1.0 + tol).all(axis=1)
    inside &= resid <= 1e-9 * max(scale, 1e-12)
    inside = inside.reshape(len(pts), kk)
    first = np.where(inside.any(axis=1), inside.argmax(axis=1), -1)
    cells = np.where(first >= 0, cand[np.arange(len(pts)), np.maximum(first, 0)], -1)
    pick = np.arange(len(pts)) * kk + np.maximum(first, 0)
    W = N[pick]
    A = np.einsum("mkj,mki->mji", Xc[pick], dN[pick])                 # ∂x_j / ∂ξ_i at the chosen points
    A[np.abs(np.linalg.det(A)) < 1e-300] = np.eye(dim)
    G = np.einsum("mki,mij->mkj", dN[pick], np.linalg.inv(A))         # ∂N_k / ∂x_j
    return cells, W, G
