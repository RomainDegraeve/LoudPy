"""Sparse assembly: COO element blocks → CSR with a precomputed scatter map (pattern built once)."""
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from loudpy.fem.sparse_kernels import build_mapping, sum_duplicates_coo

def _build_csr_mapping(I, J, V_size, shape):
    """Build CSR sparsity pattern + scatter mapping for COO→CSR accumulation."""
    dummy = coo_matrix((np.ones(V_size, np.int8), (I, J)), shape=shape).tocsr()
    mapping = np.empty(V_size, dtype=np.int32)
    build_mapping(I, J, dummy.indptr, dummy.indices, mapping)
    return dummy.indptr, dummy.indices, dummy.nnz, mapping


def _accumulate_csr(V, mapping, indptr, indices, nnz, shape):
    """Sum contributions into CSR matrix using precomputed mapping."""
    is_complex = np.iscomplexobj(V)
    data = np.zeros(nnz, dtype=V.dtype)
    if is_complex:
        sum_duplicates_coo(V.real, mapping, data.real)
        sum_duplicates_coo(V.imag, mapping, data.imag)
    else:
        sum_duplicates_coo(V, mapping, data)
    return csr_matrix((data, indices, indptr), shape=shape)



def local_coords(node_tags, node_coords, flat_tags, n_e, n_pts, dim=None):
    """Map global tags → local coordinates, reshape to (n_e, n_pts, dim)."""
    tags = np.asarray(node_tags)
    coords = np.asarray(node_coords)
    sort_idx = np.argsort(tags)
    local_idx = sort_idx[np.searchsorted(tags[sort_idx], flat_tags)]
    ec = coords[local_idx].reshape(n_e, n_pts, -1)
    if dim is not None:
        ec = ec[:, :, :dim]
    return np.ascontiguousarray(ec)


class SparseAssemblerBase:
    """
    Base class providing CSR pattern/mapping construction utilities
    for any global FE matrix assembled from element COO triplets.
    """
    def _finalize_pattern(self, I, J, V_size, shape):
        self.shape = shape
        self.V_size = V_size
        self.indptr, self.indices, self.nnz, self.mapping = \
            _build_csr_mapping(I, J, V_size, shape)

    def _to_csr(self, V):
        return _accumulate_csr(V, self.mapping, self.indptr,
                               self.indices, self.nnz, self.shape)
