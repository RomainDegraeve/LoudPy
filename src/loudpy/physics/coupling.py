"""
Fluid–structure coupling on the boundaries with role "fluid_structure".

FsiAssembler: rectangular matrix (n_dof_acou × n_dof_meca), C_ij = ∫ N_i (N_j e · n) 2πr ds with n the
outward normal of the air on each edge (CouplingEdge.signs, from the mesh). The coupled problem is then

    (K − ω²M + iωC) u − Cᵀ p = F          (pressure load −p n_s = p n on the structure)
    −ρω² C u + (H − ω²Q) p = 0            (∂p/∂n = ρω² u·n on the air boundary)

in every air region alike.
"""
from __future__ import annotations

import numpy as np

from loudpy.fem.batch_coupling import _fsi_local_to_V
from loudpy.fem.indexing import _fill_ij_rect
from loudpy.fem.kernels import _boundary_normal_mass_kernel, tabulate
from loudpy.fem.quadrature import legacy_line
from loudpy.fem.reference import lagrange
from loudpy.fem.sparse import _SparseAssemblerBase, _local_coords
from loudpy.physics.registry import select
from loudpy.errors import StateError


class FsiAssembler(_SparseAssemblerBase):
    PTS_PER_EDGE = 3
    PTS_PER_TRI = 6

    def __init__(self, interfaces, meca_assembler, acou_assembler):
        interfaces = select(interfaces, "fluid_structure")
        self.interfaces = interfaces
        self.meca_unique = meca_assembler.unique_tags
        self.acou_unique = acou_assembler.unique_tags

        n_dof_meca = meca_assembler.matrix_size
        n_dof_acou = acou_assembler.matrix_size

        n_p = self.PTS_PER_EDGE
        n_u = n_p * 2
        block = n_p * n_u
        self.block = block
        self.n_pts = n_p

        total_e = sum(len(iface.edges) for iface in interfaces)
        V_size = total_e * block
        I = np.empty(V_size, dtype=np.int32)
        J = np.empty(V_size, dtype=np.int32)

        all_ec, all_signs = [], []
        offset = 0
        for iface in interfaces:
            edges = np.asarray(iface.edges, dtype=np.int64)
            n_e = edges.shape[0]
            flat_tags = edges.ravel()

            ec = _local_coords(iface.node_tags, iface.node_coords, flat_tags, n_e, n_p, dim=2)
            if iface.signs is None:
                raise StateError(f"fluid–structure boundary {iface.name!r} is not oriented (no air next to it?)")
            signs = np.asarray(iface.signs, dtype=np.float64)

            # acoustic row DOFs (1 per node)
            p_gdofs = np.searchsorted(self.acou_unique, flat_tags).reshape(n_e, n_p).astype(np.int32)
            # structural column DOFs (2 per node, interleaved r, z)
            u_gidx = np.searchsorted(self.meca_unique, flat_tags).reshape(n_e, n_p).astype(np.int32)
            u_gdofs = np.empty((n_e, n_u), dtype=np.int32)
            u_gdofs[:, 0::2] = u_gidx * 2
            u_gdofs[:, 1::2] = u_gidx * 2 + 1

            _fill_ij_rect(p_gdofs, u_gdofs, I, J, offset * block)
            all_ec.append(ec)
            all_signs.append(signs)
            offset += n_e

        self.ec = np.concatenate(all_ec, axis=0)
        self.signs = np.concatenate(all_signs, axis=0)
        self._finalize_pattern(I, J, V_size, (n_dof_acou, n_dof_meca))

    def assemble(self):
        n_e, n_p = self.ec.shape[0], self.n_pts
        N, dN, w = tabulate(lagrange("line", 2), legacy_line(3))
        out = np.empty((n_e, 2, n_p * n_p), dtype=np.float64)
        if n_e:
            _boundary_normal_mass_kernel(self.ec, N, dN, w, True, out)
        out_x, out_y = np.ascontiguousarray(out[:, 0]), np.ascontiguousarray(out[:, 1])
        V = np.empty(n_e * self.block, dtype=np.float64)
        _fsi_local_to_V(out_x, out_y, self.signs, V)
        return self._to_csr(V)
