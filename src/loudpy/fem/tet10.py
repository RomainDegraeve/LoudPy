"""
P2 (10-node) tetrahedra with straight edges, and 6-node triangles on their faces: exact reference
integrals, global stiffness / mass assembly, surface quadrature (3D acoustics of the scenes).

    K, M, volume = assemble_p2(nodes, tets)          ∫∇N_i·∇N_j dV, ∫N_i N_j dV (gmsh TET10 node order)
    pts, w, N = surface_quadrature(nodes, tri6)       points, weights and shape values on TRI6 faces
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

_TET_EDGES = [(0, 1), (1, 2), (0, 2), (0, 3), (2, 3), (1, 3)]      # gmsh TET10 node order


def _tet10_tables():
    from math import factorial
    Q = np.zeros((10, 4, 4))                       # N_p = λᵀ Q_p λ (homogeneous in λ)
    one = np.ones(4)
    for i in range(4):
        e = np.eye(4)[i]
        Q[i] = 2 * np.outer(e, e) - 0.5 * (np.outer(e, one) + np.outer(one, e))
    for p, (a, b) in enumerate(_TET_EDGES, start=4):
        Q[p, a, b] = Q[p, b, a] = 2.0
    L2 = (np.ones((4, 4)) + np.eye(4)) / 20.0      # ∫ λ_f λ_g dV / V
    A = 2 * Q                                      # ∂N_p/∂λ_e = Σ_f A[p,e,f] λ_f
    T = np.einsum("pef,qhg,fg->pqeh", A, A, L2)    # ∫ c_pe c_qh dV / V
    I4 = np.zeros((4, 4, 4, 4))                    # ∫ λ_a λ_b λ_c λ_d dV / V
    for idx in np.ndindex(4, 4, 4, 4):
        cnt = np.bincount(idx, minlength=4)
        I4[idx] = 6.0 * np.prod([factorial(c) for c in cnt]) / factorial(7)
    Mref = np.einsum("pab,qcd,abcd->pq", Q, Q, I4)
    return T, Mref


_T10, _M10 = _tet10_tables()

# 6-point Dunavant rule on triangles (degree 4), barycentric (l1, l2), weights sum to 1
_TRI_QP = np.array([[0.445948490915965, 0.445948490915965], [0.108103018168070, 0.445948490915965],
                    [0.445948490915965, 0.108103018168070], [0.091576213509771, 0.091576213509771],
                    [0.816847572980459, 0.091576213509771], [0.091576213509771, 0.816847572980459]])
_TRI_W = np.array([0.223381589678011] * 3 + [0.109951743655322] * 3)


def _tri6_shape(l1, l2):
    """TRI6 shape functions (gmsh order: corners 0, 1, 2, mid-edges 01, 12, 20)."""
    l0 = 1.0 - l1 - l2
    return np.array([l0 * (2 * l0 - 1), l1 * (2 * l1 - 1), l2 * (2 * l2 - 1),
                     4 * l0 * l1, 4 * l1 * l2, 4 * l2 * l0])


def assemble_p2(nodes, tets):
    """Stiffness ∫∇N·∇N, mass ∫N N (sparse, straight P2 tetrahedra) and the volume."""
    X = nodes[tets[:, :4]]                                     # (n_e, 4, 3)
    J = np.transpose(X[:, 1:] - X[:, :1], (0, 2, 1))           # columns X_i − X_0
    det = np.linalg.det(J)
    Ji = np.linalg.inv(J)                                      # rows: ∇λ_1..3
    g = np.concatenate([-Ji.sum(axis=1, keepdims=True), Ji], axis=1)   # (n_e, 4, 3)
    G = np.einsum("nai,nbi->nab", g, g)
    V = np.abs(det) / 6.0
    Ke = V[:, None, None] * np.einsum("pqeh,neh->npq", _T10, G)
    Me = V[:, None, None] * _M10[None]
    rows = np.repeat(tets, 10, axis=1).ravel()
    cols = np.tile(tets, (1, 10)).ravel()
    n = len(nodes)
    K = sp.csc_matrix((Ke.ravel(), (rows, cols)), shape=(n, n))
    M = sp.csc_matrix((Me.ravel(), (rows, cols)), shape=(n, n))
    return K, M, float(V.sum())


def surface_quadrature(nodes, tri):
    """
    Quadrature on a TRI6 surface (straight faces, 6 points per face): points xq (n_q, 3),
    weights wq (n_q,), the surface nodes and the sparse interpolation W (n_q × n_surface_nodes).
    ∫_S N_i f dS = (Wᵀ (wq f))_i.
    """
    Xc = nodes[tri[:, :3]]
    area = 0.5 * np.linalg.norm(np.cross(Xc[:, 1] - Xc[:, 0], Xc[:, 2] - Xc[:, 0]), axis=1)
    Nq = np.array([_tri6_shape(l1, l2) for l1, l2 in _TRI_QP])                        # (6q, 6n)
    L = np.column_stack([1 - _TRI_QP.sum(1), _TRI_QP])                                # (6q, 3)
    xq = np.einsum("qa,fad->fqd", L, Xc).reshape(-1, 3)
    wq = (area[:, None] * _TRI_W[None]).ravel()
    surf_nodes, loc = np.unique(tri, return_inverse=True)
    loc = loc.reshape(tri.shape)
    W = sp.csr_matrix((np.tile(Nq, (len(tri), 1)).ravel(),
                       (np.repeat(np.arange(len(xq)), 6), np.tile(loc, (1, 6)).reshape(-1))),
                      shape=(len(xq), len(surf_nodes)))
    return xq, wq, surf_nodes, W
