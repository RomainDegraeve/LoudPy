"""
Square element-by-element assembly over T6 regions (no physics here); the element blocks come from
the generic kernels (`loudpy.fem.kernels`, `loudpy.fem.solid`) through `tabulation`.

An ElementAssembler numbers the nodes of its regions (sorted unique tags, `dofs_per_node`
interleaved DOFs per node), builds the CSR pattern of all its element blocks once, and keeps per
region the element coordinates `ec`, the element DOFs `gdofs` and the slice [s0, s1) of its
values. Physics assemblers fill the values (assemble) and call _to_csr.
"""
from __future__ import annotations

import numpy as np

from loudpy.fem.indexing import _fill_ij_square
from loudpy.fem.kernels import tabulate
from loudpy.fem.quadrature import legacy_triangle7
from loudpy.fem.reference import lagrange
from loudpy.fem.sparse import _SparseAssemblerBase, _local_coords


class ElementAssembler(_SparseAssemblerBase):
    PTS_PER_TRI = 6
    dofs_per_node = 1

    @property
    def tabulation(self):
        """(N, dN, w) of the cells' element (TRI6) at the quadrature rule of the former kernels."""
        if not hasattr(self, "_tabulation"):
            self._tabulation = tabulate(lagrange("triangle", 2), legacy_triangle7())
        return self._tabulation

    def _setup(self, subdomains):
        """Node numbering and pattern for `subdomains` (objects with node_tags, node_coords, tri)."""
        self.subdomains = subdomains
        self.ndof_e = self.PTS_PER_TRI * self.dofs_per_node
        self.block = self.ndof_e ** 2
        self.unique_tags = np.unique(np.concatenate([sd.node_tags for sd in subdomains]))
        self.tag_to_local = {int(t): i for i, t in enumerate(self.unique_tags)}
        self.matrix_size = len(self.unique_tags) * self.dofs_per_node
        self._build_topology()

    def _build_topology(self):
        n_elem_total = sum(len(sd.tri) for sd in self.subdomains)
        V_size = n_elem_total * self.block
        I = np.empty(V_size, dtype=np.int32)
        J = np.empty(V_size, dtype=np.int32)

        self.prep = []
        offset = 0
        for sd in self.subdomains:
            n_e = len(sd.tri)
            tri_flat = np.asarray(sd.tri).ravel()

            ec = _local_coords(sd.node_tags, sd.node_coords, tri_flat, n_e, self.PTS_PER_TRI)

            gidx = np.searchsorted(self.unique_tags, tri_flat).reshape(n_e, self.PTS_PER_TRI)
            gdofs = np.ascontiguousarray(
                (gidx[..., None] * self.dofs_per_node + np.arange(self.dofs_per_node))
                .reshape(n_e, self.ndof_e)
                .astype(np.int32)
            )

            s0 = offset * self.block
            s1 = s0 + n_e * self.block
            _fill_ij_square(gdofs, I, J, s0)

            self.prep.append((sd, ec, gdofs, s0, s1, n_e))
            offset += n_e

        self._finalize_pattern(I, J, V_size, (self.matrix_size, self.matrix_size))
