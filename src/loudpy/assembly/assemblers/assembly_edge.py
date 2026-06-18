
import numpy as np
from loudpy.assembly.core.helpers import _local_coords
from loudpy.assembly.kernels.indexing import _fill_dof_idx, _fill_ij_rect
from loudpy.assembly.kernels.fsi import _compute_signs_jit, _fsi_local_to_V
from loudpy.assembly.core.base import _SparseAssemblerBase
from loudpy.assembly.kernels.elements import _batch_F, _batch_FSI
from loudpy.assembly.kernels.csr import _scatter_add
from loudpy.Interfaces import InterfaceForced, InterfaceAcouMeca

# =============================================================================
class ForceAssembler:
    """Assembles a unit-normalized force vector on prescribed boundaries."""

    
    PTS_PER_EDGE = 3

    def __init__(self, forced_boundaries, meca, dofs_per_node=2):
        forced_boundaries = [f for f in forced_boundaries if isinstance(f,InterfaceForced )]
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

            ec = _local_coords(bd.node_tags, bd.node_coords,
                               flat_tags, n_e, n_pts)

            gidx = np.searchsorted(unique_tags, flat_tags).reshape(n_e, n_pts)
            gdofs = (gidx * dofs_per_node + d_off).astype(np.int32)

            IDX = np.empty(n_e * n_pts, dtype=np.int32)
            _fill_dof_idx(gdofs, IDX, 0)

            ec_list.append(ec)
            idx_list.append(IDX)

        self.ec = np.concatenate(ec_list, axis=0)
        self.IDX = np.concatenate(idx_list, axis=0)

    def assemble(self):
        Fe = np.empty((self.ec.shape[0], self.PTS_PER_EDGE), dtype=np.float64)
        _batch_F(self.ec, Fe)

        F = np.zeros(self.total_dofs, dtype=np.float64)
        _scatter_add(Fe.ravel(), self.IDX, F)

        total = F.sum()
        if total > 0.0:
            F /= total
        return F

# =============================================================================
# FSI Assembler
# =============================================================================
class FsiAssembler(_SparseAssemblerBase):
    """
    Rectangular fluid–structure coupling matrix
    of shape (n_dof_acou, n_dof_meca).
    """
    PTS_PER_EDGE = 3
    PTS_PER_TRI  = 6

    def __init__(self, interfaces, meca_assembler, acou_assembler):
        interfaces = [i for i in interfaces if isinstance(i, InterfaceAcouMeca )]
        self.interfaces = interfaces
        self.meca_unique = meca_assembler.unique_tags
        self.acou_unique = acou_assembler.unique_tags

        n_dof_meca = meca_assembler.matrix_size
        n_dof_acou = acou_assembler.matrix_size

        n_p   = self.PTS_PER_EDGE
        n_u   = n_p * 2
        block = n_p * n_u
        self.block = block
        self.n_pts = n_p

        # --- Allocate global I/J arrays -----------------------------------
        total_e = sum(len(iface.edges) for iface in interfaces)
        V_size  = total_e * block
        I = np.empty(V_size, dtype=np.int32)
        J = np.empty(V_size, dtype=np.int32)

        all_ec, all_signs = [], []
        offset = 0

        for iface in interfaces:
           
            edges = np.asarray(iface.edges, dtype=np.int64)
            n_e = edges.shape[0]
            flat_tags = edges.ravel()

            ec = _local_coords(iface.node_tags, iface.node_coords,
                               flat_tags, n_e, n_p, dim=2)

            # JIT-ed sign computation
            name = iface.name.lower()

            sign_val = iface.direction
            

           

            signs = np.full(n_e, sign_val, dtype=np.float64)

            # Acoustic row DOFs (1/node)
            p_gdofs = np.searchsorted(self.acou_unique, flat_tags) \
                        .reshape(n_e, n_p).astype(np.int32)

            # Mechanic col DOFs (2/node, interleaved r,z)
            u_gidx = np.searchsorted(self.meca_unique, flat_tags) \
                       .reshape(n_e, n_p).astype(np.int32)
            u_gdofs = np.empty((n_e, n_u), dtype=np.int32)
            u_gdofs[:, 0::2] = u_gidx * 2        # r
            u_gdofs[:, 1::2] = u_gidx * 2 + 1    # z

            s0 = offset * block
            _fill_ij_rect(p_gdofs, u_gdofs, I, J, s0)

            all_ec.append(ec)
            all_signs.append(signs)
            offset += n_e

        self.ec    = np.concatenate(all_ec, axis=0)
        self.signs = np.concatenate(all_signs, axis=0)
        self._finalize_pattern(I, J, V_size, (n_dof_acou, n_dof_meca))

    # ------------------------------------------------------------------
    @staticmethod
    def _build_fluid_neighbor_map_csr(acou_assembler):
        """
        Returns a CSR-style adjacency:
            indptr[node_tag]     → first slot
            indptr[node_tag + 1] → end slot
            neigh_tris[k]        : (6,) triangle tags of neighbor k
            neigh_centers[k]     : (2,) centroid of neighbor k
        Indexed directly by global node tag (no Python dict).
        """
        all_centers, all_tris = [], []
        max_tag = 0
        for sd in acou_assembler.subdomains:
            tri    = np.asarray(sd.tri, dtype=np.int64)         # (n_tri, 6)
            tags   = np.asarray(sd.node_tags, dtype=np.int64)
            coords = np.asarray(sd.node_coords)[:, :2]

            sort_idx = np.argsort(tags)
            local = sort_idx[np.searchsorted(tags[sort_idx], tri.ravel())]
            tri_coords = coords[local].reshape(tri.shape[0], 6, 2)
            centers = tri_coords.mean(axis=1)                   # (n_tri, 2)

            all_tris.append(tri)
            all_centers.append(centers)
            if tri.size:
                max_tag = max(max_tag, int(tri.max()))

        tris    = np.concatenate(all_tris,    axis=0)           # (N_tri, 6)
        centers = np.concatenate(all_centers, axis=0)           # (N_tri, 2)

        # For each triangle, it contributes to 6 nodes → flatten
        n_tri = tris.shape[0]
        flat_nodes = tris.ravel()                               # (6*N_tri,)
        tri_idx    = np.repeat(np.arange(n_tri, dtype=np.int64), 6)

        # Sort by node tag → all entries for a given node become contiguous
        order = np.argsort(flat_nodes, kind='stable')
        sorted_nodes = flat_nodes[order]
        sorted_tri_idx = tri_idx[order]

        # Build indptr indexed by global tag (size = max_tag + 2)
        indptr = np.zeros(max_tag + 2, dtype=np.int64)
        # count occurrences per tag
        np.add.at(indptr, sorted_nodes + 1, 1)
        np.cumsum(indptr, out=indptr)

        # Reorder neighbor data so that they are sorted by node tag
        neigh_tris    = tris[sorted_tri_idx]                    # (6*N_tri, 6)
        neigh_centers = centers[sorted_tri_idx]                 # (6*N_tri, 2)
        
        return indptr, neigh_tris, neigh_centers

    # ------------------------------------------------------------------
    def assemble(self):
        n_e, n_p = self.ec.shape[0], self.n_pts
        out_x = np.empty((n_e, n_p * n_p), dtype=np.float64)
        out_y = np.empty((n_e, n_p * n_p), dtype=np.float64)
        _batch_FSI(self.ec, out_x, out_y)

        # Write directly into V (no temporaries)
        V = np.empty(n_e * self.block, dtype=np.float64)
        _fsi_local_to_V(out_x, out_y, self.signs, V)

        return self._to_csr(V)


