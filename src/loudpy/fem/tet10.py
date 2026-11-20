"""
P2 (10-node) tetrahedra and 6-node triangles on their faces for the 3D acoustics of the scenes, on the
generic core (`loudpy.fem.kernels`): global stiffness / mass assembly and surface quadrature.

    K, M, volume = assemble_p2(nodes, tets)          ∫∇N_i·∇N_j dV, ∫N_i N_j dV (gmsh TET10 node order)
    pts, w, nodes, W = surface_quadrature(nodes, tri6)   points, weights, surface nodes, interpolation

`nodes` are coordinates (n, 3), cells hold row indices. With `curved=True` (the default) the cells are
isoparametric: the geometry follows gmsh's mid-edge nodes (on curved CAD surfaces they lie on the
surface); `curved=False` maps every cell from its 4 (3) corners, as the former straight-sided code did.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from loudpy.fem.kernels import (_boundary_load_kernel, _laplace_kernel, _mass_kernel, quadrature_points,
                                tabulate)
from loudpy.fem.quadrature import QuadratureRule, quadrature
from loudpy.fem.reference import lagrange

__all__ = ["assemble_p2", "surface_quadrature", "CURVED", "straight_cells"]

CURVED = False         # default geometry of the 3D toolkit cells

# 6-point Dunavant rule on triangles (degree 4), barycentric (l1, l2), weights sum to 1: the rule of the
# former surface quadrature, kept so that straight faces give the same numbers
_TRI_QP = np.array([[0.445948490915965, 0.445948490915965], [0.108103018168070, 0.445948490915965],
                    [0.445948490915965, 0.108103018168070], [0.091576213509771, 0.091576213509771],
                    [0.816847572980459, 0.091576213509771], [0.091576213509771, 0.816847572980459]])
_TRI_W = np.array([0.223381589678011] * 3 + [0.109951743655322] * 3)
_DUNAVANT6 = QuadratureRule("triangle", 4, _TRI_QP, _TRI_W / 2.0)


def straight_cells(element, ec) -> np.ndarray:
    """Cell coordinates with the P2 edge nodes moved to the edge midpoints (straight-sided cells)."""
    ec = np.array(ec, dtype=float, copy=True)
    k = element.cell.dim + 1 if element.cell.simplex else len(element.cell.vertices)
    for j, (a, b) in enumerate(element.cell.edges):
        ec[:, k + j] = 0.5 * (ec[:, a] + ec[:, b])
    return ec


def assemble_p2(nodes, tets, curved: bool | None = None):
    """Stiffness ∫∇N·∇N, mass ∫N N (sparse csc over all nodes) and the volume of P2 tetrahedra."""
    curved = CURVED if curved is None else curved
    nodes = np.asarray(nodes, dtype=float)
    tets = np.asarray(tets, dtype=np.int64)
    elem = lagrange("tetrahedron", 2)
    ec = np.ascontiguousarray(nodes[tets])
    if not curved:
        ec = straight_cells(elem, ec)
    N, dN, w = tabulate(elem, quadrature("tetrahedron", 4))     # exact for straight cells
    one = np.ones((len(ec), len(w)))
    Ke = np.empty((len(ec), 100))
    Me = np.empty((len(ec), 100))
    _laplace_kernel(ec, N, dN, w, one, False, Ke)
    _mass_kernel(ec, N, dN, w, one, False, Me)
    rows = np.repeat(tets, 10, axis=1).ravel()
    cols = np.tile(tets, (1, 10)).ravel()
    n = len(nodes)
    K = sp.csc_matrix((Ke.ravel(), (rows, cols)), shape=(n, n))
    M = sp.csc_matrix((Me.ravel(), (rows, cols)), shape=(n, n))
    return K, M, float(Me.sum())


def surface_quadrature(nodes, tri, curved: bool | None = None):
    """
    Quadrature on a TRI6 surface (6 points per face): points xq (n_q, 3), weights wq (n_q,), the surface
    nodes and the sparse interpolation W (n_q × n_surface_nodes). ∫_S N_i f dS = (Wᵀ (wq f))_i.
    """
    curved = CURVED if curved is None else curved
    nodes = np.asarray(nodes, dtype=float)
    tri = np.asarray(tri, dtype=np.int64)
    elem = lagrange("triangle", 2)
    ec = np.ascontiguousarray(nodes[tri])
    if not curved:
        ec = straight_cells(elem, ec)
    N, dN, w = tabulate(elem, _DUNAVANT6)
    n_q = len(w)
    # weights: ∫ 1 · δ_q over each face → the measure at each point (unit coefficient at one point)
    wq = np.empty((len(ec), n_q))
    dx = np.empty((len(ec), n_q))
    for q in range(n_q):
        coef = np.zeros((len(ec), n_q))
        coef[:, q] = 1.0
        _boundary_load_kernel(ec, N, dN, w, coef, False, dx)
        wq[:, q] = dx.sum(axis=1)                                   # Σ_a N_a = 1
    xq = quadrature_points(ec, N).reshape(-1, 3)
    surf_nodes, loc = np.unique(tri, return_inverse=True)
    loc = loc.reshape(tri.shape)
    W = sp.csr_matrix((np.tile(N, (len(tri), 1)).ravel(),
                       (np.repeat(np.arange(len(xq)), 6), np.tile(loc, (1, n_q)).reshape(-1))),
                      shape=(len(xq), len(surf_nodes)))
    return xq, wq.ravel(), surf_nodes, W
