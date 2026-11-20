"""
Quadrature rules.

Generic rules, exact for polynomials of total degree ≤ `degree` on the reference cells of
`loudpy.fem.reference` (weights sum to the cell's measure):

    rule = quadrature("triangle", 5)        # rule.points (n_q, 2), rule.weights (n_q,)

  line, quadrilateral   Gauss–Legendre, tensor products (quads: exact to degree q in each variable)
  triangle, tetrahedron collapsed (Duffy) Gauss–Jacobi rules, positive weights, interior points

The legacy arrays below (7-point triangle rule with 8 digits, 3- and 4-point Gauss lines) are the rules
the former P2 kernels use; they stay as they are so that the former operators can be reproduced exactly.
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.special import roots_jacobi


GAUSS_TR6_7pts = np.array([
        [1/3, 1/3, 0.225],
        [0.05971587, 0.47014206, 0.13239415],
        [0.47014206, 0.05971587, 0.13239415],
        [0.47014206, 0.47014206, 0.13239415],
        [0.79742699, 0.10128651, 0.12593918],
        [0.10128651, 0.79742699, 0.12593918],
        [0.10128651, 0.10128651, 0.12593918]
    ]) # Sum of weights is 1 but area is 0.5 so a division by two is later necessary










GAUSS_L3_4pts =  np.array([
    [0.339981043584856, 0.652145154862546], 
    [-0.339981043584856, 0.652145154862546], 
    [0.861136311594053,  0.347854845137454], 
    [-0.861136311594053, 0.347854845137454]
                      ])

GAUSS_L3_3pts = np.array([
    [ 0.000000000000000, 0.888888888888889],
    [ 0.774596669241483, 0.555555555555556],
    [-0.774596669241483, 0.555555555555556]
])


# ------------------------------------------------------------------------------------- generic rules

@dataclass(frozen=True)
class QuadratureRule:
    """Points (n_q, dim) in reference coordinates and weights (n_q,); exact to total degree `degree`."""
    cell: str
    degree: int
    points: np.ndarray
    weights: np.ndarray

    def __len__(self):
        return len(self.weights)


def _gauss_jacobi(n: int, alpha: float = 0.0):
    """n-point Gauss–Jacobi rule on [-1, 1], weight (1 − x)^alpha."""
    if alpha == 0.0:
        return np.polynomial.legendre.leggauss(n)
    x, w = roots_jacobi(n, alpha, 0.0)
    return x, w


@lru_cache(maxsize=None)
def _rule(cell: str, degree: int) -> QuadratureRule:
    n = max(1, (degree + 2) // 2)                    # 2n − 1 ≥ degree
    if cell == "line":
        x, w = _gauss_jacobi(n)
        pts, wts = x[:, None], w
    elif cell == "quadrilateral":
        x, w = _gauss_jacobi(n)
        X, Y = np.meshgrid(x, x, indexing="ij")
        pts = np.column_stack([X.ravel(), Y.ravel()])
        wts = np.outer(w, w).ravel()
    elif cell == "triangle":
        # (u, v) ∈ [-1, 1]²: ξ = (1 + u)(1 − v)/4, η = (1 + v)/2, dξ dη = (1 − v)/8 du dv
        u, wu = _gauss_jacobi(n)
        v, wv = _gauss_jacobi(n, 1.0)
        U, V = np.meshgrid(u, v, indexing="ij")
        pts = np.column_stack([((1 + U) * (1 - V) / 4).ravel(), ((1 + V) / 2).ravel()])
        wts = np.outer(wu, wv).ravel() / 8.0
    elif cell == "tetrahedron":
        # ξ = (1+u)(1−v)(1−w)/8, η = (1+v)(1−w)/4, ζ = (1+w)/2, Jacobian (1−v)(1−w)²/64
        u, wu = _gauss_jacobi(n)
        v, wv = _gauss_jacobi(n, 1.0)
        t, wt = _gauss_jacobi(n, 2.0)
        U, V, T = np.meshgrid(u, v, t, indexing="ij")
        pts = np.column_stack([((1 + U) * (1 - V) * (1 - T) / 8).ravel(),
                               ((1 + V) * (1 - T) / 4).ravel(), ((1 + T) / 2).ravel()])
        wts = np.einsum("i,j,k->ijk", wu, wv, wt).ravel() / 64.0
    else:
        raise ValueError(f"no quadrature on {cell!r}")
    pts.setflags(write=False)
    wts.setflags(write=False)
    return QuadratureRule(cell, degree, pts, wts)


def quadrature(cell: str, degree: int) -> QuadratureRule:
    """A rule on `cell` exact for polynomials of total degree ≤ `degree` (cached, read-only arrays)."""
    if degree < 0:
        raise ValueError("degree must be ≥ 0")
    return _rule(cell, int(degree))


def legacy_triangle7() -> QuadratureRule:
    """The former 7-point triangle rule (8-digit values), weights scaled to sum to ½."""
    return QuadratureRule("triangle", 5, GAUSS_TR6_7pts[:, :2].copy(), GAUSS_TR6_7pts[:, 2] / 2.0)


def legacy_line(n: int) -> QuadratureRule:
    """The former 3- or 4-point Gauss line rules (15 digits), on [-1, 1]."""
    table = {3: GAUSS_L3_3pts, 4: GAUSS_L3_4pts}[n]
    return QuadratureRule("line", 2 * n - 1, table[:, :1].copy(), table[:, 1].copy())
