"""
Generic axisymmetric elasticity kernels (Numba), any Lagrange element on the meridian (r, z).

Inputs as in `loudpy.fem.kernels` (ec, N, dN, w), plus the material:

  axisymmetric_kernel(ec, N, dN, w, D, out)          K = ∫ Bᵀ D B 2πr dA, 2 DOF per node (u_r, u_z),
                                                     strains [ε_r, ε_z, γ_rz, ε_θ = u_r / r], D 4×4
  fourier_kernel(ec, N, dN, w, D6, m, out)           azimuthal order m, 3 DOF per node (U_r, U_z, U_θ),
                                                     strains [ε_r, ε_z, γ_rz, ε_θ, γ_rθ, γ_zθ], D 6×6,
                                                     ∫ … r dθ = 2π (m = 0) or π (m ≥ 1)
  total_lagrangian_kernel(ec, ue, N, dN, w, D, rho, alpha, beta, K, M, C, F)
                                                     geometrically nonlinear axisymmetric element:
                                                     tangent K_T = K_mat + K_geo, consistent M,
                                                     C = α M + β K_mat, internal force F_int

The matrix D and the strain ordering are those of `loudpy.fem.shape.material_matrix_D` and of
`material_matrix_D6`. 1/r is evaluated at the quadrature points only (never on the axis nodes).
Outputs are flattened row-major (n_e, (c·n)²), DOFs interleaved per node.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from loudpy.fem.kernels import TWO_PI, _det_inv, _gradients, _jacobian, _radius

__all__ = ["axisymmetric_kernel", "fourier_kernel", "total_lagrangian_kernel", "material_matrix_D6"]


@njit(cache=True)
def material_matrix_D6(E, nu):
    """Isotropic elasticity, strains [ε_r, ε_z, γ_rz, ε_θ, γ_rθ, γ_zθ] (6×6)."""
    f = E / ((1 + nu) * (1 - 2 * nu))
    g = (1 - 2 * nu) / 2
    D = np.zeros((6, 6))
    D[0, 0] = D[1, 1] = D[3, 3] = 1 - nu
    D[0, 1] = D[1, 0] = D[0, 3] = D[3, 0] = D[1, 3] = D[3, 1] = nu
    D[2, 2] = D[4, 4] = D[5, 5] = g
    return f * D


@njit(parallel=True, cache=True, fastmath=False)
def axisymmetric_kernel(ec, N, dN, w, D, out):
    n_e, n, _ = ec.shape
    n_q = N.shape[0]
    m = 2 * n
    for e in prange(n_e):
        J = np.empty((2, 2))
        Ji = np.empty((2, 2))
        G = np.empty((n, 2))
        B = np.zeros((4, m))
        Ke = np.zeros((m, m))
        for q in range(n_q):
            _jacobian(ec[e], dN[q], J)
            det = _det_inv(J, Ji)
            _gradients(dN[q], J, Ji, det, G)
            r = _radius(ec[e], N[q])
            inv_r = 1.0 / r
            for a in range(n):
                B[0, 2 * a] = G[a, 0]
                B[1, 2 * a + 1] = G[a, 1]
                B[2, 2 * a] = G[a, 1]
                B[2, 2 * a + 1] = G[a, 0]
                B[3, 2 * a] = N[q, a] * inv_r
            s = TWO_PI * r * w[q] * abs(det)
            Ke += s * (B.T @ D @ B)
        out[e] = Ke.ravel()


@njit(parallel=True, cache=True, fastmath=False)
def fourier_kernel(ec, N, dN, w, D6, m, out):
    n_e, n, _ = ec.shape
    n_q = N.shape[0]
    k = 3 * n
    tf = TWO_PI if m == 0 else np.pi
    for e in prange(n_e):
        J = np.empty((2, 2))
        Ji = np.empty((2, 2))
        G = np.empty((n, 2))
        B = np.zeros((6, k))
        Ke = np.zeros((k, k))
        for q in range(n_q):
            _jacobian(ec[e], dN[q], J)
            det = _det_inv(J, Ji)
            _gradients(dN[q], J, Ji, det, G)
            r = _radius(ec[e], N[q])
            ir = 1.0 / r
            for a in range(n):
                cr, cz, ct = 3 * a, 3 * a + 1, 3 * a + 2
                Na = N[q, a]
                B[0, cr] = G[a, 0]
                B[1, cz] = G[a, 1]
                B[2, cr] = G[a, 1]
                B[2, cz] = G[a, 0]
                B[3, cr] = Na * ir
                B[3, ct] = m * Na * ir
                B[4, cr] = -m * Na * ir
                B[4, ct] = G[a, 0] - Na * ir
                B[5, cz] = -m * Na * ir
                B[5, ct] = G[a, 1]
            s = tf * r * w[q] * abs(det)
            Ke += s * (B.T @ D6 @ B)
        out[e] = Ke.ravel()


@njit(parallel=True, cache=True, fastmath=False)
def total_lagrangian_kernel(ec, ue, N, dN, w, D, rho, alpha, beta, K_out, M_out, C_out, F_out):
    n_e, n, _ = ec.shape
    n_q = N.shape[0]
    m = 2 * n
    for e in prange(n_e):
        J = np.empty((2, 2))
        Ji = np.empty((2, 2))
        G = np.empty((n, 2))
        K_mat = np.zeros((m, m))
        K_geo = np.zeros((m, m))
        M_e = np.zeros((m, m))
        F_int = np.zeros(m)
        B = np.zeros((4, m))
        E = np.empty(4)
        u = ue[e]
        for q in range(n_q):
            _jacobian(ec[e], dN[q], J)
            det = _det_inv(J, Ji)
            _gradients(dN[q], J, Ji, det, G)
            r = 0.0
            u_r = 0.0
            du_dr = 0.0
            du_dz = 0.0
            dw_dr = 0.0
            dw_dz = 0.0
            for a in range(n):
                r += N[q, a] * ec[e, a, 0]
                u_r += N[q, a] * u[2 * a]
                du_dr += G[a, 0] * u[2 * a]
                du_dz += G[a, 1] * u[2 * a]
                dw_dr += G[a, 0] * u[2 * a + 1]
                dw_dz += G[a, 1] * u[2 * a + 1]
            if r < 1e-8:
                r = 1e-8
            # Green–Lagrange strains
            E[0] = du_dr + 0.5 * (du_dr * du_dr + dw_dr * dw_dr)
            E[1] = dw_dz + 0.5 * (du_dz * du_dz + dw_dz * dw_dz)
            E[2] = du_dz + dw_dr + (du_dr * du_dz + dw_dr * dw_dz)
            ut = u_r / r
            E[3] = ut + 0.5 * ut * ut
            S = D @ E
            B[:] = 0.0
            for a in range(n):
                B[0, 2 * a] = G[a, 0] * (1.0 + du_dr)
                B[0, 2 * a + 1] = G[a, 0] * dw_dr
                B[1, 2 * a] = G[a, 1] * du_dz
                B[1, 2 * a + 1] = G[a, 1] * (1.0 + dw_dz)
                B[2, 2 * a] = G[a, 1] * (1.0 + du_dr) + G[a, 0] * du_dz
                B[2, 2 * a + 1] = G[a, 0] * (1.0 + dw_dz) + G[a, 1] * dw_dr
                B[3, 2 * a] = (N[q, a] / r) * (1.0 + u_r / r)
            dV = np.pi * r * abs(det) * (2.0 * w[q])
            K_mat += (B.T @ D @ B) * dV
            F_int += (B.T @ S) * dV
            m_scal = rho * dV
            hoop = S[3] / (r * r) * dV
            for a in range(n):
                for b in range(n):
                    nn = N[q, a] * N[q, b]
                    M_e[2 * a, 2 * b] += m_scal * nn
                    M_e[2 * a + 1, 2 * b + 1] += m_scal * nn
                    base = (G[a, 0] * S[0] * G[b, 0] + G[a, 1] * S[1] * G[b, 1]
                            + G[a, 0] * S[2] * G[b, 1] + G[a, 1] * S[2] * G[b, 0]) * dV
                    K_geo[2 * a, 2 * b] += base + nn * hoop
                    K_geo[2 * a + 1, 2 * b + 1] += base
        K_T = K_mat + K_geo
        C_e = alpha * M_e + beta * K_mat
        K_out[e] = K_T.ravel()
        M_out[e] = M_e.ravel()
        C_out[e] = C_e.ravel()
        F_out[e] = F_int
