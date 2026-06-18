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

# Precompute outer(N, N) at each gp (element-independent)
_NN = np.empty((len_gp, 6, 6))
for _g in range(len_gp):
    _NN[_g] = np.outer(_N[_g], _N[_g])

# Constant used in axisymmetric integration (2π / 2 = π)
PI = np.pi


@njit(cache=True, fastmath=True)
def compute_Q_T6(coords, c, sign_val):
    """Compute element mass matrix Q for axisymmetric T6 acoustic element."""
    Q = np.zeros((6, 6))

    for g in range(len_gp):
        # ← no fct_form_T6 call, no np.outer: all precomputed
        N_vec   = _N[g]
        dN_dxi  = _dNdxi[g]
        dN_deta = _dNdeta[g]
        w       = _w[g]
        Q_local = _NN[g]

        _, _, _, _, detJ = jacobian_T6(dN_dxi, dN_deta, coords)

        r = 0
        for i in range(6):
            r += N_vec[i] * coords[i, 0]

        # Fold 2π/2 = π into a single scalar
        s = PI * w * (sign_val*abs(detJ)) * r
        Q += s * Q_local

    Q *= 1.0 / (c * c)   # divide once at the end
    return Q



