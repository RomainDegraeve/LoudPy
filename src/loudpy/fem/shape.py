from numba import njit
import numpy as np 

@njit(cache=True, fastmath=True)
def fct_form_T6(xi, eta):
   
    N = np.array([
        1 - 3*xi - 3*eta + 2*xi**2 + 4*xi*eta + 2*eta**2,
        2*xi**2 - xi,
        2*eta**2 - eta,
        4*xi - 4*xi**2 - 4*xi*eta,
        4*xi*eta,
        4*eta - 4*xi*eta - 4*eta**2
    ])
    
    # Derivatives
    dN_dxi = np.array([-3 + 4*xi + 4*eta, 4*xi - 1, 0, 4 - 8*xi - 4*eta, 4*eta, -4*eta])
    dN_deta = np.array([-3 + 4*xi + 4*eta, 0, 4*eta - 1, -4*xi, 4*xi, 4 - 4*xi - 8*eta])
    return N, dN_dxi, dN_deta






@njit(cache=True, fastmath=True)
def jacobian_T6(dN_dxi, dN_deta, coords):
    r = coords[:, 0]
    z = coords[:, 1]
    J00 = dN_dxi  @ r
    J01 = dN_dxi  @ z
    J10 = dN_deta @ r
    J11 = dN_deta @ z
    detJ = J00 * J11 - J01 * J10
    return J00, J01, J10, J11, detJ

@njit(cache=True, fastmath = True)
def fct_form_L3(xi):

    N = np.array([-0.5 * xi * (1.0 - xi), 0.5 * xi * (1.0 + xi), 1.0 - xi * xi])
    dN_dxi = np.array([xi - 0.5,xi + 0.5, -2.0 * xi ])

    return N, dN_dxi

# STILL NEED TO REMOVE L3 FUNC OUT OF JACOBIAN FOR EFFICIENCY !
@njit(cache=True, fastmath=True)
def jacobian_L3(xi, coords):

    _, dN_dxi = fct_form_L3(xi)

    j11 = np.dot(dN_dxi, coords[:, 0])
    j12 = np.dot(dN_dxi, coords[:, 1])

    detJ = np.sqrt(j11**2 + j12**2)

    return detJ

@njit(cache=True, fastmath=True)
def material_matrix_D(E, nu):
    """Material stiffness matrix for axisymmetric case (4x4)
       Ordered for strain vector [e_r, e_z, gamma_rz, e_theta]
    """
    factor = E / ((1 + nu) * (1 - 2 * nu))
    D = factor * np.array([
        [1 - nu,   nu,          0,           nu],      # sigma_r
        [  nu,   1 - nu,        0,           nu],      # sigma_z
        [   0,      0,   (1 - 2*nu)/2,        0],      # tau_rz
        [  nu,     nu,          0,         1 - nu]     # sigma_theta
    ])
    return D
