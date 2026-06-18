from numba import njit
import numpy as np
from loudpy.Finite_Elements_formulations.FE_kernels import fct_form_T6, jacobian_T6
from loudpy.Finite_Elements_formulations.Gauss_Quadratures import GAUSS_TR6_7pts as gp

len_gp = gp.shape[0]

# ── Precompute shape functions at all Gauss points ────────────────────────────
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

PI =  np.pi # not 2 pi beacause sum(w) is 1 and should be 0.5




@njit(cache=True, fastmath=False)   # ← the critical change
def _spatial_grads(dN_dxi, dN_deta, J00, J01, J10, J11):
    invdet = 1.0 / (J00*J11 - J01*J10)
    # Return as tuples to avoid heap allocation
    # Numba will treat these as registers
    dN_dr = ( (J11*dN_dxi[0] - J01*dN_deta[0]) * invdet,
              (J11*dN_dxi[1] - J01*dN_deta[1]) * invdet,
              (J11*dN_dxi[2] - J01*dN_deta[2]) * invdet,
              (J11*dN_dxi[3] - J01*dN_deta[3]) * invdet,
              (J11*dN_dxi[4] - J01*dN_deta[4]) * invdet,
              (J11*dN_dxi[5] - J01*dN_deta[5]) * invdet )
    
    dN_dz = ( (-J10*dN_dxi[0] + J00*dN_deta[0]) * invdet,
              (-J10*dN_dxi[1] + J00*dN_deta[1]) * invdet,
              (-J10*dN_dxi[2] + J00*dN_deta[2]) * invdet,
              (-J10*dN_dxi[3] + J00*dN_deta[3]) * invdet,
              (-J10*dN_dxi[4] + J00*dN_deta[4]) * invdet,
              (-J10*dN_dxi[5] + J00*dN_deta[5]) * invdet )
    return dN_dr, dN_dz

@njit(cache=True, fastmath=False)   # ← the critical change
def _disp_grads(N, dN_dr, dN_dz, u_e, coords):
    r = u_r = du_dr = du_dz = dw_dr = dw_dz = 0.0
    for i in range(6):
        r     += N[i]     * coords[i, 0]
        u_r   += N[i]     * u_e[2*i]
        du_dr += dN_dr[i] * u_e[2*i]
        du_dz += dN_dz[i] * u_e[2*i]
        dw_dr += dN_dr[i] * u_e[2*i+1] # Note: Ensure these indices
        dw_dz += dN_dz[i] * u_e[2*i+1] # match your local DOF mapping
    return r, u_r, du_dr, du_dz, dw_dr, dw_dz


@njit(cache=True, fastmath=False)   # ← the critical change
def _green_lagrange(u_r, r, du_dr, du_dz, dw_dr, dw_dz):
    E = np.empty(4)
    E[0] = du_dr + 0.5*(du_dr*du_dr + dw_dr*dw_dr)
    E[1] = dw_dz + 0.5*(du_dz*du_dz + dw_dz*dw_dz)
    E[2] = du_dz + dw_dr + (du_dr*du_dz + dw_dr*dw_dz)
    ut   = u_r / r
    E[3] = ut + 0.5*ut*ut
    return E


@njit(cache=True, fastmath=False)   # ← the critical change
def _B_matrix(N, dN_dr, dN_dz, r, u_r, du_dr, du_dz, dw_dr, dw_dz):
    B = np.zeros((4, 12))
    for i in range(6):
        B[0, 2*i  ] = dN_dr[i]*(1.0 + du_dr)
        B[0, 2*i+1] = dN_dr[i]*dw_dr
        B[1, 2*i  ] = dN_dz[i]*du_dz
        B[1, 2*i+1] = dN_dz[i]*(1.0 + dw_dz)
        B[2, 2*i  ] = dN_dz[i]*(1.0 + du_dr) + dN_dr[i]*du_dz
        B[2, 2*i+1] = dN_dr[i]*(1.0 + dw_dz) + dN_dz[i]*dw_dr
        B[3, 2*i  ] = (N[i]/r)*(1.0 + u_r/r)
    return B


"""
@njit(cache=True, fastmath=False)   # ← the critical change
def _accumulate_K_geo(K_geo, dN_dr, dN_dz, N, S, r, dV):
    for i in range(6):
        for j in range(6):
            base = (dN_dr[i]*S[0]*dN_dr[j]
                  + dN_dz[i]*S[1]*dN_dz[j]
                  + dN_dr[i]*S[2]*dN_dz[j]
                  + dN_dz[i]*S[2]*dN_dr[j]) * dV
            K_geo[2*i,   2*j  ] += base
            K_geo[2*i+1, 2*j+1] += base
            K_geo[2*i,   2*j  ] += S[3] * (N[i]/r) * (N[j]/r) * dV
"""


@njit(cache=True, fastmath=False, inline='always')
def _accumulate_K_geo(K_geo, dN_dr, dN_dz, N, S, r, dV):
    hoop_scal = S[3] / (r * r) * dV
    for i in range(6):
        for j in range(6):
            base = (dN_dr[i]*S[0]*dN_dr[j]
                  + dN_dz[i]*S[1]*dN_dz[j]
                  + dN_dr[i]*S[2]*dN_dz[j]
                  + dN_dz[i]*S[2]*dN_dr[j]) * dV
            K_geo[2*i,   2*j  ] += base + N[i]*N[j]*hoop_scal
            K_geo[2*i+1, 2*j+1] += base



@njit(cache=True, fastmath=False)   # ← the critical change
def compute_element_TL(coords, u_e, D, rho, alpha, beta):
    """
    Total Lagrangian Axisymmetric T6 — physical Rayleigh damping.
        K_T = K_mat + K_geo
        C_e = alpha*M_e + beta*K_mat
    Returns
    -------
    K_T   : (12,12) tangent stiffness  (material + geometric)
    M_e   : (12,12) consistent mass
    C_e   : (12,12) Rayleigh damping
    F_int : (12,)   internal force
    """
    K_mat = np.zeros((12, 12))
    K_geo = np.zeros((12, 12))
    M_e   = np.zeros((12, 12))
    F_int = np.zeros(12)

    for g in range(len_gp):
        N       = _N[g]
        dN_dxi  = _dNdxi[g]
        dN_deta = _dNdeta[g]
        w       = _w[g]
        NN      = _NN[g]

        J00, J01, J10, J11, detJ = jacobian_T6(dN_dxi, dN_deta, coords)
        dN_dr, dN_dz = _spatial_grads(dN_dxi, dN_deta, J00, J01, J10, J11)

        r, u_r, du_dr, du_dz, dw_dr, dw_dz = _disp_grads(N, dN_dr, dN_dz, u_e, coords)
        if r < 1e-8:
            r = 1e-8
        E = _green_lagrange(u_r, r, du_dr, du_dz, dw_dr, dw_dz)
        S = D @ E
        B = _B_matrix(N, dN_dr, dN_dz, r, u_r, du_dr, du_dz, dw_dr, dw_dz)

        dV = PI * r * abs(detJ) * w

        K_mat += (B.T @ D @ B) * dV
        F_int += (B.T @ S)     * dV

        m_scal = rho * dV
        M_e[0::2, 0::2] += m_scal * NN
        M_e[1::2, 1::2] += m_scal * NN

        _accumulate_K_geo(K_geo, dN_dr, dN_dz, N, S, r, dV)

    K_T = K_mat + K_geo
    C_e = alpha * M_e + beta * K_mat
    return K_T, M_e, C_e, F_int
