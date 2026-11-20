"""Fluid–structure coupling kernel: interleave the (r, z) blocks and apply the orientation signs."""
from numba import njit, prange


@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
def _fsi_local_to_V(out_x, out_y, signs, V):
    """
    Vectorized interleave + sign application, written directly into V.
    out_x, out_y : (n_e, n_p*n_p) flat
    V            : (n_e * n_p * (n_p*2),) flat
    """
    n_e = out_x.shape[0]
    npp = out_x.shape[1]                       # n_p * n_p
    n_p = 0
    # infer n_p from npp (small, computed once)
    while n_p * n_p != npp:
        n_p += 1

    block = n_p * (n_p * 2)
    for e in prange(n_e):
        s = signs[e]
        base = e * block
        for i in range(n_p):
            row_off = base + i * (n_p * 2)
            xrow = i * n_p
            for j in range(n_p):
                V[row_off + 2 * j]     = s * out_x[e, xrow + j]
                V[row_off + 2 * j + 1] = s * out_y[e, xrow + j]


