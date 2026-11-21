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

from loudpy.fem.batch_coupling import fsi_local_to_V
from loudpy.fem.indexing import fill_ij_rect
from loudpy.fem.assembly import cell_tabulation
from loudpy.fem.kernels import boundary_normal_mass_kernel
from loudpy.fem.sparse import SparseAssemblerBase, local_coords
from loudpy.physics.registry import select
from loudpy.errors import StateError


class FsiAssembler(SparseAssemblerBase):
    """Assembler of the fluid–structure coupling Cam = ∫ N_u n N_p 2πr ds on the coupling edges, with the
    normal of the adjacent air on every edge."""
    PTS_PER_EDGE = 3
    PTS_PER_TRI = 6

    def __init__(self, interfaces, meca_assembler, acou_assembler):
        interfaces = select(interfaces, "fluid_structure")
        self.interfaces = interfaces
        self.meca_unique = meca_assembler.unique_tags
        self.acou_unique = acou_assembler.unique_tags

        n_dof_meca = meca_assembler.matrix_size
        n_dof_acou = acou_assembler.matrix_size

        widths = {np.asarray(iface.edges).shape[1] for iface in interfaces}
        if len(widths) > 1:
            raise ValueError(f"fluid–structure edges of different orders ({sorted(widths)} nodes)")
        n_p = self.PTS_PER_EDGE = widths.pop() if widths else self.PTS_PER_EDGE
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

            ec = local_coords(iface.node_tags, iface.node_coords, flat_tags, n_e, n_p, dim=2)
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

            fill_ij_rect(p_gdofs, u_gdofs, I, J, offset * block)
            all_ec.append(ec)
            all_signs.append(signs)
            offset += n_e

        self.ec = np.concatenate(all_ec, axis=0)
        self.signs = np.concatenate(all_signs, axis=0)
        self._finalize_pattern(I, J, V_size, (n_dof_acou, n_dof_meca))

    def assemble(self):
        """The coupling matrix (structure DOFs × air nodes)."""
        n_e, n_p = self.ec.shape[0], self.n_pts
        N, dN, w = cell_tabulation(1, n_p)
        out = np.empty((n_e, 2, n_p * n_p), dtype=np.float64)
        if n_e:
            boundary_normal_mass_kernel(self.ec, N, dN, w, True, out)
        out_x, out_y = np.ascontiguousarray(out[:, 0]), np.ascontiguousarray(out[:, 1])
        V = np.empty(n_e * self.block, dtype=np.float64)
        fsi_local_to_V(out_x, out_y, self.signs, V)
        return self._to_csr(V)
