"""
Exact non-reflecting (Dirichlet-to-Neumann, DtN) condition on a spherical
exterior boundary, axisymmetric T6/L3 acoustics, time convention e^{st}
(s = jω).

Outside the sphere |x − x0| = R the outgoing pressure is

    p(r, θ) = Σ_n p_n(R) [k_n(s r/c) / k_n(s R/c)] P_n(cos θ)

(k_n: modified spherical Bessel function of the 2nd kind, θ from the axis),
so on the boundary, order by order,

    ∂p/∂r = Σ_n D_n(s) p_n P_n(cos θ),     D_n(s) = (s/c) k_n'(z)/k_n(z),  z = sR/c.

With k_n(z) ∝ e^{−z} z^{−n−1} θ_n(z) (θ_n: reverse Bessel polynomial of
degree n, simple roots z_i in the left half-plane) this kernel is EXACTLY
rational in s:

    D_n(s) = −s/c − 1/R + (1/R) Σ_{i=1..n} z_i / (sR/c − z_i)

(n = 0: −s/c − 1/R, i.e. Bayliss–Turkel order 1).  Each term is linearised
with one auxiliary unknown, so the coupled problem stays a polynomial
(quadratic) eigenvalue problem in s with no approximation of the radiation
condition beyond the truncation order.

Weak form: the acoustic row gains −∮ (∂p/∂n) q dS = −Σ_n D_n(s) β_n (b_nᵀp) b_n
with b_n,i = ∮ P_n(cos θ) N_i dS  (dS = 2π r ds, as in H and Q) and
    β_n = (2n+1)/(4πR²)  full sphere (all n),
    β_n = (2n+1)/(2πR²)  hemisphere on a rigid baffle through the centre
                         (field even about the baffle → even n only).
"""
from __future__ import annotations

from math import factorial

import numpy as np
from loudpy.errors import MeshError
from loudpy.fem.kernels import boundary_load_kernel, quadrature_points, tabulate
from loudpy.fem.quadrature import quadrature
from loudpy.fem.reference import element_for_cells


# ── Reverse Bessel polynomials θ_n and their roots ─────────────────────────────
def _theta_and_derivative(n: int, z):
    """θ_n(z), θ_n'(z) by the recurrence θ_n = (2n−1)θ_{n−1} + z²θ_{n−2}."""
    z = np.asarray(z, dtype=complex)
    t0, d0 = np.ones_like(z), np.zeros_like(z)
    if n == 0:
        return t0, d0
    t1, d1 = z + 1.0, np.ones_like(z)
    for m in range(2, n + 1):
        t0, t1 = t1, (2 * m - 1) * t1 + z * z * t0
        d0, d1 = d1, (2 * m - 1) * d1 + 2 * z * t0 + z * z * d0
    return t1, d1


def reverse_bessel_roots(n: int) -> np.ndarray:
    """The n roots of θ_n (poles of the DtN kernel of order n), Newton-refined."""
    if n == 0:
        return np.empty(0, dtype=complex)
    # θ_n(z) = Σ_m a_m z^{n−m},  a_m = (n+m)! / (m! (n−m)! 2^m); scale z = n·w
    a = np.array([factorial(n + m) / (factorial(m) * factorial(n - m) * 2.0 ** m)
                  for m in range(n + 1)], dtype=float)
    scaled = a / float(n) ** np.arange(n + 1)          # coefficient of w^{n−m}
    roots = np.roots(scaled / scaled[0]) * n
    for _ in range(8):
        t, d = _theta_and_derivative(n, roots)
        roots = roots - t / d
    return roots


def _legendre_all(n_max: int, x: np.ndarray) -> np.ndarray:
    """P_0..P_nmax at x, shape (n_max+1, len(x))."""
    P = np.empty((n_max + 1, x.size))
    P[0] = 1.0
    if n_max >= 1:
        P[1] = x
    for n in range(1, n_max):
        P[n + 1] = ((2 * n + 1) * x * P[n] - n * P[n - 1]) / (n + 1)
    return P


def dtn_boundary_vectors(edges: np.ndarray, node_tags: np.ndarray,
                         node_coords: np.ndarray, tag_to_local: dict,
                         n_dof: int, center, orders) -> tuple:
    """
    b_n (dense over the acoustic DOFs, only boundary entries non-zero) for
    every n in `orders`, and the boundary radius R.

    edges        : (n_e, n) gmsh node tags of the line edges (corners first, gmsh order)
    node_coords  : coordinates matching node_tags (r, z)
    center       : (r0, z0) of the sphere, r0 must be 0 (on the axis)
    """
    xy = {int(t): np.asarray(c)[:2] for t, c in zip(node_tags, node_coords)}
    r0, z0 = center
    dist = np.array([np.hypot(xy[int(t)][0] - r0, xy[int(t)][1] - z0) for t in node_tags])
    R = float(dist.mean())
    if np.ptp(dist) > 1e-3 * R:
        raise MeshError(f"DtN boundary is not a circle about {center}: "
                         f"radius {dist.min():.6g}–{dist.max():.6g} m")

    orders = list(orders)
    n_max = max(orders)
    edges = np.asarray(edges, dtype=np.int64)
    n_p = edges.shape[1]
    ec = np.ascontiguousarray(np.array([[xy[int(t)] for t in e] for e in edges]).reshape(len(edges), n_p, 2))
    loc = np.vectorize(lambda t: tag_to_local[int(t)])(edges) if len(edges) else edges
    N, dN, w = tabulate(element_for_cells(1, n_p), quadrature("line", 15))     # 8-point Gauss
    x = quadrature_points(ec, N)                                                # (n_e, n_q, 2)
    cos_t = (x[..., 1] - z0) / np.hypot(x[..., 0] - r0, x[..., 1] - z0)
    P = _legendre_all(n_max, cos_t.ravel()).reshape(n_max + 1, *cos_t.shape)
    B = np.zeros((len(orders), n_dof))
    be = np.empty((len(ec), n_p))
    for j, n in enumerate(orders):                    # b_n,i = ∮ P_n(cos θ) N_i 2πr ds
        boundary_load_kernel(ec, N, dN, w, np.ascontiguousarray(P[n]), True, be)
        np.add.at(B[j], loc, be)
    return B, R


def dtn_orders(R: float, f_max: float, c: float, half: bool, n_max: int | None = None):
    """Truncation N ≈ ⌈k_max R⌉ + 10; even orders only for a baffled hemisphere."""
    if n_max is None:
        n_max = int(np.ceil(2 * np.pi * f_max / c * R)) + 10
    return list(range(0, n_max + 1, 2 if half else 1)), n_max
