from numba import njit, prange

@njit(parallel=True, cache=True)
def _fill_ij_square(gdofs, I, J, s0):
    n_e, m = gdofs.shape
    for e in prange(n_e):
        base = s0 + e * m * m
        for i in range(m):
            row = base + i * m
            for j in range(m):
                I[row + j] = gdofs[e, i]
                J[row + j] = gdofs[e, j]

@njit(parallel=True, cache=True)
def _fill_ij_rect(row_dofs, col_dofs, I, J, s0):
    n_e, n_r = row_dofs.shape
    n_c = col_dofs.shape[1]
    for e in prange(n_e):
        base = s0 + e * n_r * n_c
        for i in range(n_r):
            row = base + i * n_c
            for j in range(n_c):
                I[row + j] = row_dofs[e, i]
                J[row + j] = col_dofs[e, j]

@njit(parallel=True, cache=True)
def _fill_dof_idx(gdofs, IDX, s0):
    n_e, m = gdofs.shape
    for e in prange(n_e):
        base = s0 + e * m
        for i in range(m):
            IDX[base + i] = gdofs[e, i]

