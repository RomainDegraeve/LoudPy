"""
Fluid–structure coupling on the boundaries with role "fluid_structure".

FsiAssembler: rectangular matrix (n_dof_acou × n_dof_meca); each edge carries the orientation ±1
(`direction`) of the air side it couples.
"""
from __future__ import annotations

import numpy as np

from loudpy.fem.batch import _batch_FSI
from loudpy.fem.batch_coupling import _fsi_local_to_V
from loudpy.fem.indexing import _fill_ij_rect
from loudpy.fem.sparse import _SparseAssemblerBase, _local_coords
from loudpy.physics.registry import select


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
            signs = np.full(n_e, iface.direction, dtype=np.float64)

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
        out_x = np.empty((n_e, n_p * n_p), dtype=np.float64)
        out_y = np.empty((n_e, n_p * n_p), dtype=np.float64)
        _batch_FSI(self.ec, out_x, out_y)
        V = np.empty(n_e * self.block, dtype=np.float64)
        _fsi_local_to_V(out_x, out_y, self.signs, V)
        return self._to_csr(V)
