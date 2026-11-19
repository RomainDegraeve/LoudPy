"""
TR6_axi_fourier.py — T6 elasticity elements for an axisymmetric solid under a
non-axisymmetric displacement of azimuthal (Fourier) order m.

The 3D displacement of order m is written on the meridian (r, z) mesh as

    u_r(r, θ, z) = U_r(r, z) cos mθ
    u_z(r, θ, z) = U_z(r, z) cos mθ
    u_θ(r, θ, z) = U_θ(r, z) sin mθ          (the sin/cos family gives the same modes)

so each order is an independent 2D problem with 3 DOF per node, ordered (U_r, U_z, U_θ).
The six engineering strains, ordered [ε_r, ε_z, γ_rz, ε_θ, γ_rθ, γ_zθ], are

    ε_r  = ∂U_r/∂r                  ε_θ  = (U_r + m U_θ) / r
    ε_z  = ∂U_z/∂z                  γ_rθ = ∂U_θ/∂r − (U_θ + m U_r) / r
    γ_rz = ∂U_r/∂z + ∂U_z/∂r        γ_zθ = ∂U_θ/∂z − m U_z / r

(times cos mθ for the first four, sin mθ for the last two). The θ-integral of cos² / sin²
is π for m ≥ 1 and 2π for m = 0 (for m = 0, U_θ is the torsional, θ-independent field),
so the element matrices are those of the full 3D body for this order.
"""
import numpy as np
from numba import njit

from loudpy.fem.shape import fct_form_T6, jacobian_T6
from loudpy.fem.quadrature import GAUSS_TR6_7pts as gp

len_gp = gp.shape[0]
_N = np.empty((len_gp, 6))
_dNdxi = np.empty((len_gp, 6))
_dNdeta = np.empty((len_gp, 6))
_w = np.empty(len_gp)
for _g in range(len_gp):
    _Nv, _dxi, _det = fct_form_T6(gp[_g, 0], gp[_g, 1])
    _N[_g], _dNdxi[_g], _dNdeta[_g] = _Nv, _dxi, _det
    _w[_g] = gp[_g, 2] * 0.5           # reference triangle area 1/2 folded into the weights


@njit(cache=True)
def material_matrix_D6(E, nu):
    """Isotropic elasticity, strains [ε_r, ε_z, γ_rz, ε_θ, γ_rθ, γ_zθ] (6x6)."""
    f = E / ((1 + nu) * (1 - 2 * nu))
    g = (1 - 2 * nu) / 2
    D = np.zeros((6, 6))
    D[0, 0] = D[1, 1] = D[3, 3] = 1 - nu
    D[0, 1] = D[1, 0] = D[0, 3] = D[3, 0] = D[1, 3] = D[3, 1] = nu
    D[2, 2] = D[4, 4] = D[5, 5] = g
    return f * D


@njit(cache=True)
def _theta_factor(m):
    return 2.0 * np.pi if m == 0 else np.pi


@njit(cache=True, fastmath=True)
def compute_K_T6_fourier(coords, D6, m):
    """Stiffness matrix (18x18) of order m, DOF per node (U_r, U_z, U_θ)."""
    K_e = np.zeros((18, 18))
    B = np.zeros((6, 18))
    tf = _theta_factor(m)
    for g in range(len_gp):
        N = _N[g]
        J00, J01, J10, J11, detJ = jacobian_T6(_dNdxi[g], _dNdeta[g], coords)
        inv = 1.0 / detJ
        dNr = (J11 * _dNdxi[g] - J01 * _dNdeta[g]) * inv
        dNz = (-J10 * _dNdxi[g] + J00 * _dNdeta[g]) * inv
        r = 0.0
        for i in range(6):
            r += N[i] * coords[i, 0]
        ir = 1.0 / r
        B[:] = 0.0
        for i in range(6):
            cr, cz, ct = 3 * i, 3 * i + 1, 3 * i + 2
            B[0, cr] = dNr[i]
            B[1, cz] = dNz[i]
            B[2, cr] = dNz[i]
            B[2, cz] = dNr[i]
            B[3, cr] = N[i] * ir
            B[3, ct] = m * N[i] * ir
            B[4, cr] = -m * N[i] * ir
            B[4, ct] = dNr[i] - N[i] * ir
            B[5, cz] = -m * N[i] * ir
            B[5, ct] = dNz[i]
        s = tf * r * _w[g] * abs(detJ)
        K_e += s * (B.T @ D6 @ B)
    return K_e


@njit(cache=True, fastmath=True)
def compute_M_T6_fourier(coords, rho, m):
    """Consistent mass matrix (18x18) of order m, DOF per node (U_r, U_z, U_θ)."""
    M_e = np.zeros((18, 18))
    tf = _theta_factor(m)
    for g in range(len_gp):
        N = _N[g]
        _, _, _, _, detJ = jacobian_T6(_dNdxi[g], _dNdeta[g], coords)
        r = 0.0
        for i in range(6):
            r += N[i] * coords[i, 0]
        s = tf * max(abs(r), 1e-12) * _w[g] * abs(detJ) * rho
        for i in range(6):
            for j in range(6):
                v = s * N[i] * N[j]
                for c in range(3):
                    M_e[3 * i + c, 3 * j + c] += v
    return M_e
