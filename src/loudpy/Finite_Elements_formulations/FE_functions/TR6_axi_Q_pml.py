import numpy as np
from numba import njit
from loudpy.Finite_Elements_formulations.FE_kernels import fct_form_T6, jacobian_T6
from loudpy.Finite_Elements_formulations.Gauss_Quadratures import GAUSS_TR6_7pts as gp

len_gp = gp.shape[0]

_N      = np.empty((len_gp, 6))
_dNdxi  = np.empty((len_gp, 6))
_dNdeta = np.empty((len_gp, 6))
_w      = np.empty(len_gp)

for _g in range(len_gp):
    _xi, _eta, _ww = gp[_g]
    _Nv, _dxi, _det = fct_form_T6(_xi, _eta)
    _N[_g], _dNdxi[_g], _dNdeta[_g], _w[_g] = _Nv, _dxi, _det, _ww

_NN = np.empty((len_gp, 6, 6))
for _g in range(len_gp):
    _NN[_g] = np.outer(_N[_g], _N[_g])

PI = np.pi

# ─────────────────────────────────────────────────────────────────────────────
# EXACT COMSOL-STYLE PML STRETCHING
# ─────────────────────────────────────────────────────────────────────────────
@njit(cache=True, fastmath=True, inline="always")
def _eval_stretch(r_gp, z_gp,
                  use_r, r0, tr, sgn_r,
                  use_z, z0, tz, sgn_z,
                  alpha, n, lambda0):
    """Evaluate sr, sz, r_tilde at a Gauss point, supporting corner overlap."""
    sr = complex(1.0, 0.0)
    sz = complex(1.0, 0.0)
    r_tilde = complex(r_gp, 0.0)

    if use_r:
        sr, r_tilde = _stretch(r_gp, r0, tr, sgn_r, alpha, n, lambda0)
    if use_z:
        sz, _ = _stretch(z_gp, z0, tz, sgn_z, alpha, n, lambda0)
    return sr, sz, r_tilde


@njit(cache=True, fastmath=True, inline="always")
def _stretch(x, x0, t, sign, alpha, n, lambda0):
    xi = sign * (x - x0) / t
    if xi <= 0.0:
        return complex(1.0, 0.0), complex(x, 0.0)
    if xi > 1.0:
        xi = 1.0
    xi_nm1 = xi ** (n - 1)
    xi_n   = xi_nm1 * xi
    
    # s NEVER takes the sign. It must always have a negative imaginary part.
    s       = 1.0 - 1j * alpha * lambda0 * (n / t) * xi_nm1
    x_tilde = x - 1j * alpha * lambda0 * xi_n * sign
    return s, x_tilde
# ─────────────────────────────────────────────────────────────────────────────
# MASS MATRIX (Q) — Axisymmetric PML
# ─────────────────────────────────────────────────────────────────────────────
@njit(cache=True, fastmath=True)
def compute_Q_T6_pml(coords, c, use_r, r0, tr, sgn_r, 
                     use_z, z0, tz, sgn_z, alpha, n, omega, sign_val):
    lambda0 = 2.0 * PI * c / omega
    Q = np.zeros((6, 6), dtype=np.complex128)

    for g in range(len_gp):
        N_vec   = _N[g]
        dN_dxi  = _dNdxi[g]
        dN_deta = _dNdeta[g]
        w       = _w[g]
        Q_local = _NN[g]

        # We only need detJ for the integration area
        _, _, _, _, detJ = jacobian_T6(dN_dxi, dN_deta, coords)

        r_gp = 0.0
        z_gp = 0.0
        for i in range(6):
            r_gp += N_vec[i] * coords[i, 0]
            z_gp += N_vec[i] * coords[i, 1]

        sr, sz, r_tilde = _eval_stretch(r_gp, z_gp,  
                                        use_r, r0, tr, sgn_r,
                                        use_z, z0, tz, sgn_z,
                                        alpha, n, lambda0)

        # Integration measure: dV = 2*pi * r_tilde * sr * sz * dr * dz

        scale = PI * w * (sign_val*abs(detJ)) * r_tilde * sr * sz
        Q += scale * Q_local

    Q *= 1.0 / (c * c)
    return Q