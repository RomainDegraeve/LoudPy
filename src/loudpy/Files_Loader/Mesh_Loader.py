from __future__ import annotations
import h5py
import numpy as np

from loudpy.Files_Loader.Load_Domains import Domain
from loudpy.Files_Loader.Core_Models import Mesh

class MeshLoader:
    """Loads a Mesh from any loudpy HDF5 layout."""

    @staticmethod
    def load(h5: h5py.File, mesh_id: str, domain: Domain | str) -> Mesh:
        dom = Domain.coerce(domain)
        g = h5[f"meshes/{mesh_id}"]

        subgroups = MeshLoader._find_subgroups(g, dom)
        if not subgroups and {"tags", "tris"} <= set(g.keys()):
            subgroups = [g]
        if not subgroups:
            raise ValueError(f"No mesh data for domain '{dom.value}' in {mesh_id}")

        all_tags, all_coords, all_tris = [], [], []
        for sg in subgroups:
            tags, coords, tris_tags = MeshLoader._extract_subgroup(sg)
            all_tags.append(tags)
            all_coords.append(coords)
            all_tris.append(tris_tags)

        return MeshLoader._merge(np.concatenate(all_tags),
                                 np.concatenate(all_coords),
                                 np.concatenate(all_tris))

    @staticmethod
    def _find_subgroups(g, dom: Domain) -> list:
        return [g[k] for k in g.keys()
                if isinstance(g[k], h5py.Group)
                and any(k.startswith(p) for p in dom.subgroup_prefixes)]

    @staticmethod
    def _extract_subgroup(sg) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tags      = sg["tags"][:]
        coord_key = "coords" if "coords" in sg else "nodes"
        coords    = sg[coord_key][:, :2]
        tris      = sg["tris"][:, :3].astype(np.int64)
        tris_tags = tris if tris.max() >= len(tags) else tags[tris]
        return tags, coords, tris_tags

    @staticmethod
    def _merge(all_tags, all_coords, all_tris_tags) -> Mesh:
        unique_tags, first_occ = np.unique(all_tags, return_index=True)
        unique_coords = all_coords[first_occ]
        tag2row = {int(t): i for i, t in enumerate(unique_tags)}
        tris_idx = np.vectorize(tag2row.get)(all_tris_tags)
        ok = ((tris_idx[:, 0] != tris_idx[:, 1])
              & (tris_idx[:, 1] != tris_idx[:, 2])
              & (tris_idx[:, 0] != tris_idx[:, 2]))
        return Mesh(unique_tags, unique_coords, tris_idx[ok])

