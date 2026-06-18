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
def compute_unit_force_T6(edge_coords):
    """
    Axisymmetric nodal forces for unit pressure on a curved 3-node edge.
    """
    F = np.zeros(3)

    for g in range(len_gp):
        
        N_vec   = _N[g]
        dN_dxi  = _dNdxi[g]
        w       = _w[g]

        r      = np.dot(N_vec,  edge_coords[:, 0])
        dr_dxi = np.dot(dN_dxi, edge_coords[:, 0])
        dz_dxi = np.dot(dN_dxi, edge_coords[:, 1])

        # curved edge jacobian
        ds = np.sqrt(dr_dxi**2 + dz_dxi**2)

        F += 2.0 * PI * r * w * ds * N_vec

    return F
