"""
Reference cells and Lagrange elements on gmsh's node layouts.

Reference cells (gmsh conventions):

  line            ξ ∈ [-1, 1]                     measure 2
  triangle        ξ, η ≥ 0, ξ + η ≤ 1             measure 1/2
  quadrilateral   (ξ, η) ∈ [-1, 1]²               measure 4
  tetrahedron     ξ, η, ζ ≥ 0, ξ + η + ζ ≤ 1      measure 1/6

A Lagrange element of order p has one node per point of gmsh's layout (corners, then the nodes of the
edges in gmsh's edge order and direction, then faces, then the interior); its basis is the nodal basis
of the polynomial space P_p (simplices: total degree ≤ p) or Q_p (quadrilaterals: degree ≤ p in each
variable), computed from the inverse of the Vandermonde matrix on monomials. `tabulate(points)` returns
N (n_q, n) and ∇N (n_q, n, dim) in reference coordinates.

    elem = lagrange("triangle", 2)          # TRI6, gmsh type 9
    elem = element_for_gmsh_type(21)        # TRI10
    N, dN = elem.tabulate(points)

The node coordinates are written with exact fractions; `tests/test_fem_reference.py` checks every layout
against `gmsh.model.mesh.getElementProperties`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from itertools import product

import numpy as np

__all__ = ["ReferenceCell", "LagrangeElement", "CELLS", "lagrange", "element_for_gmsh_type",
           "GMSH_TYPES"]


@dataclass(frozen=True)
class ReferenceCell:
    """A reference cell: name, dimension, vertices, measure, sub-entities (local vertex indices)."""
    name: str
    dim: int
    vertices: np.ndarray = field(repr=False)
    measure: float
    simplex: bool
    edges: tuple[tuple[int, ...], ...]
    faces: tuple[tuple[int, ...], ...] = ()      # 3D cells only, outward orientation as in gmsh

    @property
    def facets(self) -> tuple[tuple[int, ...], ...]:
        """Sub-entities of dimension dim − 1 (points for a line, edges in 2D, faces in 3D)."""
        if self.dim == 1:
            return ((0,), (1,))
        return self.edges if self.dim == 2 else self.faces

    @property
    def facet_cell(self) -> str | None:
        """Name of the facets' reference cell."""
        return {1: None, 2: "line", 3: "triangle"}[self.dim]


CELLS: dict[str, ReferenceCell] = {
    "line": ReferenceCell("line", 1, np.array([[-1.0], [1.0]]), 2.0, True, ((0, 1),)),
    "triangle": ReferenceCell("triangle", 2, np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), 0.5,
                              True, ((0, 1), (1, 2), (2, 0))),
    "quadrilateral": ReferenceCell("quadrilateral", 2,
                                   np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]),
                                   4.0, False, ((0, 1), (1, 2), (2, 3), (3, 0))),
    "tetrahedron": ReferenceCell("tetrahedron", 3,
                                 np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                                           [0.0, 0.0, 1.0]]), 1.0 / 6.0, True,
                                 ((0, 1), (1, 2), (2, 0), (3, 0), (3, 2), (3, 1)),
                                 ((0, 2, 1), (0, 1, 3), (0, 3, 2), (3, 1, 2))),
}


# --------------------------------------------------------------------------------------- node layouts

def _edge_points(cell: ReferenceCell, p: int) -> list[np.ndarray]:
    """p − 1 equally spaced points per edge, in gmsh's edge order, from the edge's first vertex."""
    pts = []
    for a, b in cell.edges:
        A, B = cell.vertices[a], cell.vertices[b]
        pts += [A + (B - A) * k / p for k in range(1, p)]
    return pts


def _layout(cell_name: str, p: int) -> np.ndarray:
    """gmsh's reference node coordinates of the order-p Lagrange element (p ≤ 3; Q2 for quads)."""
    cell = CELLS[cell_name]
    pts = list(cell.vertices)
    if p == 1:
        return np.array(pts, dtype=float)
    pts += _edge_points(cell, p)
    if cell_name == "quadrilateral":
        if p != 2:
            raise ValueError("quadrilaterals: orders 1 and 2 only")
        pts.append(np.zeros(2))                                   # QUAD9 centre
    elif cell_name == "triangle" and p == 3:
        pts.append(np.array([1.0, 1.0]) / 3.0)                   # TRI10 interior node
    elif cell_name == "tetrahedron" and p == 3:
        for face in cell.faces:                                  # TET20: one node per face (centroid)
            pts.append(cell.vertices[list(face)].mean(axis=0))
    elif p > 3:
        raise ValueError(f"{cell_name}: orders up to 3 only")
    return np.array(pts, dtype=float)


# gmsh element type → (cell, order); the elements loudpy knows
GMSH_TYPES: dict[int, tuple[str, int]] = {
    1: ("line", 1), 8: ("line", 2), 26: ("line", 3),
    2: ("triangle", 1), 9: ("triangle", 2), 21: ("triangle", 3),
    3: ("quadrilateral", 1), 10: ("quadrilateral", 2),
    4: ("tetrahedron", 1), 11: ("tetrahedron", 2), 29: ("tetrahedron", 3),
}
_GMSH_TYPE_OF = {v: k for k, v in GMSH_TYPES.items()}


def _exponents(cell: ReferenceCell, p: int) -> np.ndarray:
    """Monomial exponents spanning P_p (simplices) or Q_p (quadrilaterals), shape (n, dim)."""
    rng = range(p + 1)
    exps = [e for e in product(rng, repeat=cell.dim) if (sum(e) <= p if cell.simplex else True)]
    exps.sort(key=lambda e: (sum(e), tuple(-x for x in e)))
    return np.array(exps, dtype=np.int64)


def _monomials(points: np.ndarray, exps: np.ndarray):
    """Values (n_q, n) and gradients (n_q, n, dim) of the monomials x^e at the points."""
    x = np.asarray(points, dtype=float)[:, None, :]              # (n_q, 1, dim)
    e = exps[None, :, :]                                          # (1, n, dim)
    pw = x ** e
    V = pw.prod(axis=2)
    dim = exps.shape[1]
    G = np.empty(V.shape + (dim,))
    for d in range(dim):
        ed = e[..., d]
        dpw = np.where(ed > 0, ed * x[..., d] ** np.maximum(ed - 1, 0), 0.0)
        others = np.delete(pw, d, axis=2).prod(axis=2) if dim > 1 else 1.0
        G[..., d] = dpw * others
    return V, G


class LagrangeElement:
    """
    Nodal Lagrange element of order `order` on `cell`, nodes in gmsh's layout.

    Attributes: `cell`, `order`, `nodes` (n, dim), `gmsh_type`, `degree` (highest total degree of the
    basis: p for simplices, dim·p for Q_p), `facet_nodes` (local node indices of each facet, ordered as
    the facet element of the same order: corners first, then edge / face nodes as gmsh numbers them).
    """

    def __init__(self, cell: str, order: int):
        if cell not in CELLS:
            raise ValueError(f"unknown reference cell {cell!r}; known: {sorted(CELLS)}")
        if (cell, order) not in _GMSH_TYPE_OF:
            raise ValueError(f"no Lagrange element of order {order} on a {cell}")
        self.cell = CELLS[cell]
        self.order = order
        self.gmsh_type = _GMSH_TYPE_OF[(cell, order)]
        self.nodes = _layout(cell, order)
        self._exps = _exponents(self.cell, order)
        V, _ = _monomials(self.nodes, self._exps)
        self._coef = np.linalg.inv(V)                              # N(x) = m(x) · coef
        self.degree = order if self.cell.simplex else self.cell.dim * order
        self.facet_nodes = self._facet_nodes()

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def dim(self) -> int:
        return self.cell.dim

    def __repr__(self):
        return f"LagrangeElement({self.cell.name!r}, {self.order})  # gmsh type {self.gmsh_type}"

    def tabulate(self, points) -> tuple[np.ndarray, np.ndarray]:
        """Basis values N (n_q, n) and reference gradients ∇N (n_q, n, dim) at the points (n_q, dim)."""
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        if pts.shape[1] != self.dim:
            pts = pts.reshape(-1, self.dim)
        V, G = _monomials(pts, self._exps)
        N = V @ self._coef
        dN = np.einsum("qmd,mn->qnd", G, self._coef)
        return N, dN

    def interpolate(self, f) -> np.ndarray:
        """Nodal values f(nodes) (f takes points (n, dim), returns (n,) or (n, k))."""
        return np.asarray(f(self.nodes))

    def _facet_nodes(self) -> tuple[np.ndarray, ...]:
        """Local nodes on each facet, ordered as the facet element (affine map from the facet's corners)."""
        if self.dim == 1:
            return (np.array([0]), np.array([1]))
        sub = lagrange(self.cell.facet_cell, self.order)
        out = []
        for facet in self.cell.facets:
            X = self.cell.vertices[list(facet)]
            # affine map: facet reference corners → facet vertices (first dim + 1 corners)
            A = np.column_stack([sub.cell.vertices, np.ones(len(sub.cell.vertices))])
            M, *_ = np.linalg.lstsq(A[: sub.dim + 1], X[: sub.dim + 1], rcond=None)
            images = np.column_stack([sub.nodes, np.ones(sub.n_nodes)]) @ M
            d = np.linalg.norm(self.nodes[None, :, :] - images[:, None, :], axis=2)
            idx = d.argmin(axis=1)
            if not np.allclose(d[np.arange(len(idx)), idx], 0.0, atol=1e-12):
                raise AssertionError(f"facet nodes not found on {self}")
            out.append(idx)
        return tuple(out)


@lru_cache(maxsize=None)
def lagrange(cell: str, order: int) -> LagrangeElement:
    """The (cached) Lagrange element of `order` on `cell`."""
    return LagrangeElement(cell, order)


def element_for_gmsh_type(gmsh_type: int) -> LagrangeElement:
    """The Lagrange element matching a gmsh element type (1, 2, 3, 4, 8, 9, 10, 11, 21, 26, 29)."""
    if gmsh_type not in GMSH_TYPES:
        raise ValueError(f"gmsh element type {gmsh_type} is not supported; known: {sorted(GMSH_TYPES)}")
    return lagrange(*GMSH_TYPES[gmsh_type])
