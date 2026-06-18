import numpy as np
from dataclasses import dataclass

@dataclass
class MeshTopology:
    """Global mesh snapshot, deduplicated across subdomains by node tag."""
    tags:   np.ndarray   # (n_nodes,)        unique global tags
    coords: np.ndarray   # (n_nodes, dim)    coords aligned with tags
    tris:   np.ndarray   # (n_elem, 6)       row indices into `tags`/`coords`

    # tag -> row index, useful when scattering DOF vectors back to mesh
    @property
    def tag_to_row(self):
        return {int(t): i for i, t in enumerate(self.tags)}

    # ------------------------------------------------------------------
    @classmethod
    def from_fem_objects(cls, fem_objects, filter_cls=None):
        subs = [d for d in fem_objects
                if (filter_cls is None or isinstance(d, filter_cls))
                and hasattr(d, "tri")]

        if not subs:
            raise ValueError("No subdomains matched.")

        # 1) union of node tags across selected subdomains
        all_tags  = np.concatenate([sd.node_tags   for sd in subs])
        all_coord = np.concatenate([sd.node_coords for sd in subs], axis=0)

        tags, first_idx = np.unique(all_tags, return_index=True)
        coords = all_coord[first_idx]

        # 2) remap each subdomain's tri (tag-based) into row indices
        tri_blocks = []
        for sd in subs:
            tri_flat = np.asarray(sd.tri).ravel()
            rows = np.searchsorted(tags, tri_flat)
            tri_blocks.append(rows.reshape(sd.tri.shape))
        tris = np.vstack(tri_blocks).astype(np.int32)

        return cls(tags=tags, coords=coords, tris=tris)

    # ------------------------------------------------------------------
    def dof_rows(self, asm_unique_tags):
        """Map an assembler's `unique_tags` to rows in this topology."""
        return np.searchsorted(self.tags, np.asarray(asm_unique_tags))

    def scatter(self, asm_unique_tags, values, dtype=float):
        """Place a per-DOF solution vector onto the global node array."""
        out = np.zeros(len(self.tags), dtype=dtype)
        out[self.dof_rows(asm_unique_tags)] = values
        return out
    