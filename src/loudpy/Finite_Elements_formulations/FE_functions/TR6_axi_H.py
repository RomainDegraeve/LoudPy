from numba import njit
from loudpy.Finite_Elements_formulations.FE_kernels import fct_form_T6, jacobian_T6
import numpy as np
from loudpy.Finite_Elements_formulations.Gauss_Quadratures import GAUSS_TR6_7pts as gp

len_gp = gp.shape[0]

# ── Precompute shape functions at all Gauss points ────────────────────────────
_N      = np.empty((len_gp, 6))   # N_i(ξ_g, η_g)
_dNdxi  = np.empty((len_gp, 6))   # ∂N_i/∂ξ at each gp
_dNdeta = np.empty((len_gp, 6))   # ∂N_i/∂η at each gp
_w      = np.empty(len_gp)        # quadrature weights

for _g in range(len_gp):
    _xi, _eta, _ww = gp[_g]
    _Nv, _dxi, _det = fct_form_T6(_xi, _eta)
    _N[_g]      = _Nv
    _dNdxi[_g]  = _dxi
    _dNdeta[_g] = _det
    _w[_g]      = _ww

# Constant used in axisymmetric integration (2π / 2 = π)
PI = np.pi


@njit(cache=True, fastmath=True)
def compute_H_T6(coords, sign_val):
    """Compute element stiffness matrix H for axisymmetric T6 acoustic element."""
    H = np.zeros((6, 6))

    for g in range(len_gp):
        # ← no fct_form_T6 call: read precomputed values
        N_vec   = _N[g]
        dN_dxi  = _dNdxi[g]
        dN_deta = _dNdeta[g]
        w       = _w[g]

        J00, J01, J10, J11, detJ = jacobian_T6(dN_dxi, dN_deta, coords)
        inv_detJ = 1.0 / detJ                       # ← divide once
        

        dN_dr = ( J11 * dN_dxi - J01 * dN_deta) * inv_detJ
        dN_dz = (-J10 * dN_dxi + J00 * dN_deta) * inv_detJ



        H_local = np.outer(dN_dr, dN_dr) + np.outer(dN_dz, dN_dz)

        r = 0
        for i in range(6):
            r += N_vec[i] * coords[i, 0]

        # Fold 2π/2 = π into a single scalar
        s = PI * w * (sign_val*abs(detJ)) * r
        H += s * H_local

    return H


