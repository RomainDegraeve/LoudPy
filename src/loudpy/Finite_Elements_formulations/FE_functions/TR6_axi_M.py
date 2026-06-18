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

# Precompute outer(N, N) at each gp (constant across elements)
_NN = np.empty((len_gp, 6, 6))
for _g in range(len_gp):
    _NN[_g] = np.outer(_N[_g], _N[_g])

PI = np.pi


@njit(cache=True, fastmath=True) # Force cache off to ensure update
def compute_M_T6(coords, rho):
    M_e = np.zeros((12, 12))
    for g in range(len_gp):
        N_vec   = _N[g]
        dN_dxi  = _dNdxi[g]
        dN_deta = _dNdeta[g]
        w       = _w[g]
        outer   = _NN[g]

        _, _, _, _, detJ = jacobian_T6(dN_dxi, dN_deta, coords)

        # Calculate radius r (assuming x-coordinate is r)
        r = 0.0
        for i in range(6):
            r += N_vec[i] * coords[i, 0]

        # Use absolute values for everything to guarantee positive mass
        # Max(abs(r), 1e-12) prevents zero-mass on the axis
        abs_r = max(abs(r), 1e-12)
        abs_detJ = abs(detJ)
        
        M_scalar = PI * abs_r * w * abs_detJ * rho

        M_e[0::2, 0::2] += M_scalar * outer
        M_e[1::2, 1::2] += M_scalar * outer

    return M_e