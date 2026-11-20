"""
Coil force: the consistent nodal force of a unit total load on the boundaries with role "coil",
along their components `direction`, normalised so that the forces sum to 1.
"""
from __future__ import annotations

import numpy as np

from loudpy.fem.indexing import _fill_dof_idx
from loudpy.fem.assembly import cell_tabulation
from loudpy.fem.kernels import _boundary_load_kernel
from loudpy.fem.sparse import _local_coords
from loudpy.fem.sparse_kernels import _scatter_add
from loudpy.physics.registry import select


class ForceAssembler:
    PTS_PER_EDGE = 3

    def __init__(self, forced_boundaries, meca, dofs_per_node=2):
        forced_boundaries = select(forced_boundaries, "coil")
        self.forced_boundaries = forced_boundaries
        self.dofs_per_node = dofs_per_node
        unique_tags = np.array(sorted(meca.tag_to_local.keys()), dtype=np.int64)
        self.total_dofs = len(unique_tags) * dofs_per_node

        ec_list, idx_list = [], []
        for bd in forced_boundaries:
            d_off = bd.direction
            edges = np.asarray(bd.edges)
            n_e, n_pts = edges.shape
            flat_tags = edges.ravel()

            ec = _local_coords(bd.node_tags, bd.node_coords, flat_tags, n_e, n_pts)
            gidx = np.searchsorted(unique_tags, flat_tags).reshape(n_e, n_pts)
            gdofs = (gidx * dofs_per_node + d_off).astype(np.int32)

            IDX = np.empty(n_e * n_pts, dtype=np.int32)
            _fill_dof_idx(gdofs, IDX, 0)
            ec_list.append(ec)
            idx_list.append(IDX)

        self.ec = np.concatenate(ec_list, axis=0)
        self.IDX = np.concatenate(idx_list, axis=0)

    def assemble(self):
        n_pts = self.ec.shape[1]
        N, dN, w = cell_tabulation(1, n_pts)
        Fe = np.empty((self.ec.shape[0], n_pts), dtype=np.float64)
        ec = np.ascontiguousarray(self.ec[:, :, :2], dtype=np.float64)
        _boundary_load_kernel(ec, N, dN, w, np.ones((len(ec), len(w))), True, Fe)
        F = np.zeros(self.total_dofs, dtype=np.float64)
        _scatter_add(Fe.ravel(), self.IDX, F)
        total = F.sum()
        if total > 0.0:
            F /= total
        return F
