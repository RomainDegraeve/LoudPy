import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from loudpy.assembly.kernels.csr import _build_mapping, _sum_duplicates

def _build_csr_mapping(I, J, V_size, shape):
    """Build CSR sparsity pattern + scatter mapping for COO→CSR accumulation."""
    dummy = coo_matrix((np.ones(V_size, np.int8), (I, J)), shape=shape).tocsr()
    mapping = np.empty(V_size, dtype=np.int32)
    _build_mapping(I, J, dummy.indptr, dummy.indices, mapping)
    return dummy.indptr, dummy.indices, dummy.nnz, mapping


def _accumulate_csr(V, mapping, indptr, indices, nnz, shape):
    """Sum contributions into CSR matrix using precomputed mapping."""
    is_complex = np.iscomplexobj(V)
    data = np.zeros(nnz, dtype=V.dtype)
    if is_complex:
        _sum_duplicates(V.real, mapping, data.real)
        _sum_duplicates(V.imag, mapping, data.imag)
    else:
        _sum_duplicates(V, mapping, data)
    return csr_matrix((data, indices, indptr), shape=shape)

