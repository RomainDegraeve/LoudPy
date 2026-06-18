from loudpy.Finite_Elements_formulations.Gauss_Quadratures import GAUSS_L3_3pts as gp
import numpy as np
from loudpy.Finite_Elements_formulations.FE_kernels import fct_form_L3
from numba import njit

len_gp = gp.shape[0]

PI = np.pi

_N = np.empty((len_gp, 3))
_dNdxi = np.empty((len_gp, 3))
_w = np.empty(len_gp)        # quadratur

for _g in range(len_gp):
    _xi, _ww = gp[_g]
    _Nv, _dxi = fct_form_L3(_xi)
    _N[_g] = _Nv
    _dNdxi[_g] = _dxi
    _w[_g] = _ww

@njit(cache=True, fastmath=True)
def compute_FSI_T6(edge_coords):

    """
    Axisymmetric fluid-structure coupling for a curved TR6 edge.
    """
    M_x = np.zeros((3, 3))   # ADD THIS
    M_y = np.zeros((3, 3))   # ADD THIS
    for g in range(len_gp):
        
        N_vec   = _N[g]
        dN_dxi  = _dNdxi[g]
        w       = _w[g]
        
        r     = np.dot(N_vec,  np.ascontiguousarray(edge_coords[:, 0]))
        dr_dxi = np.dot(dN_dxi, np.ascontiguousarray(edge_coords[:, 0]))
        dz_dxi = np.dot(dN_dxi, np.ascontiguousarray(edge_coords[:, 1]))

      
        # outward normal * ds (no need to normalise since we want n*ds)
        # n = (dz/dxi, -dr/dxi) / |t|  →  n*ds = (dz_dxi, -dr_dxi)
        nr_ds =  dz_dxi
        nz_ds = -dr_dxi

        common = 2.0 * PI * r * w   

        outer = np.outer(N_vec, N_vec)
        M_x += common * nr_ds * outer
        M_y += common * nz_ds * outer

    return M_x, M_y



