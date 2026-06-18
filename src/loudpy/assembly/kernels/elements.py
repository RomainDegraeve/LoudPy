from numba import njit, prange
from loudpy.Finite_Elements_formulations import compute_H_T6, compute_M_T6, compute_Q_T6, compute_K_T6, compute_FSI_T6, compute_unit_force_T6, compute_element_TL, compute_H_T6_pml, compute_Q_T6_pml

# =============================================================================
@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
def _batch_K(ec, D, out):
    for e in prange(ec.shape[0]):
        out[e] = compute_K_T6(ec[e], D).ravel()

@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
def _batch_M(ec, rho, out):
    for e in prange(ec.shape[0]):
        out[e] = compute_M_T6(ec[e], rho).ravel()

@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
def _batch_H(ec, sign_val, out):
    for e in prange(ec.shape[0]):
        out[e] = compute_H_T6(ec[e], sign_val).ravel()

@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
def _batch_Q(ec, c, sign_val, out):
    for e in prange(ec.shape[0]):
        out[e] = compute_Q_T6(ec[e], c, sign_val).ravel()

@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
def _batch_F(ec, out):
    for e in prange(ec.shape[0]):
        out[e] = compute_unit_force_T6(ec[e])

@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
def _batch_FSI(ec, out_x, out_y):
    for e in prange(ec.shape[0]):
        Mx, My = compute_FSI_T6(ec[e])
        out_x[e] = Mx.ravel()
        out_y[e] = My.ravel()


@njit(cache=True, fastmath=True, parallel=True)
def _batch_TL(ec, u_e_all, D, rho, alpha, beta,
              Ke_out, Me_out, Ce_out, Fe_out):
    """
    Vectorised Total-Lagrangian element loop (axisymmetric T6).

    Parameters
    ----------
    ec        : (n_e, 6, 2)   element nodal coords (R, Z)
    u_e_all   : (n_e, 12)     element displacement vectors (gathered)
    D         : (4, 4)        material tangent (axisym)
    rho, alpha, beta : floats (subdomain-constant)

    Outputs (preallocated)
    ----------------------
    Ke_out, Me_out, Ce_out : (n_e, 144)  flattened tangent / mass / damping
    Fe_out                 : (n_e, 12)   element internal force vector

    Notes
    -----
    The 12x12 element matrices are flattened in C order (row-major),
    matching the index layout produced by `_fill_ij_square`.
    """
    n_e = ec.shape[0]
    for e in prange(n_e):
        K_T, M_e, C_e, F_int = compute_element_TL(
            ec[e], u_e_all[e], D, rho, alpha, beta
        )
        for i in range(12):
            Fe_out[e, i] = F_int[i]
            for j in range(12):
                Ke_out[e, i * 12 + j] = K_T[i, j]
                Me_out[e, i * 12 + j] = M_e[i, j]
                Ce_out[e, i * 12 + j] = C_e[i, j]


@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
def _batch_H_pml(ec,
                 use_r, r0, tr, sgn_r,
                 use_z, z0, tz, sgn_z,
                 alpha, n, omega, c, sign_val,out):
    for e in prange(ec.shape[0]):
        out[e] = compute_H_T6_pml(
            ec[e],
            use_r, r0, tr, sgn_r,
            use_z, z0, tz, sgn_z,
            alpha, n, omega, c, sign_val
        ).ravel()

@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
def _batch_Q_pml(ec, c,
                 use_r, r0, tr, sgn_r,
                 use_z, z0, tz, sgn_z,
                 alpha, n, omega, sign_val, out):
    for e in prange(ec.shape[0]):
        out[e] = compute_Q_T6_pml(
            ec[e], c,
            use_r, r0, tr, sgn_r,
            use_z, z0, tz, sgn_z,
            alpha, n, omega, sign_val
        ).ravel()
        