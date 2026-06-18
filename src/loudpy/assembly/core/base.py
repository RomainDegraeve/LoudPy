from .csr_pattern import _accumulate_csr, _build_csr_mapping

class _SparseAssemblerBase:
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
