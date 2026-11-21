"""
loudpy.fem.sparse_kernels — Numba kernels of the sparse assembly: the map from element entries to CSR
positions (build_mapping), duplicate summation of COO entries and the scatter-add of element blocks.
"""
from numba import njit

@njit(cache=True)
def build_mapping(I, J, indptr, indices, mapping):
    for k in range(len(I)):
        row, col = I[k], J[k]
        for p in range(indptr[row], indptr[row + 1]):
            if indices[p] == col:
                mapping[k] = p
                break

@njit(cache=True)
def sum_duplicates_coo(V, mapping, out):
    for k in range(len(V)):
        out[mapping[k]] += V[k]

@njit(cache=True)
def scatter_add(V, IDX, out):
    for k in range(len(V)):
        out[IDX[k]] += V[k]
