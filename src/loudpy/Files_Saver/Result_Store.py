from pathlib import Path
from datetime import datetime
import json
import numpy as np
from typing import Any, Mapping
import h5py
from loudpy.Files_Saver.Saving_Helpers import _mesh_hash, _snap_key, _to_jsonable, _write_complex
from loudpy.Files_Saver.Mesh_topology import MeshTopology


class ResultStore:
    """Pure HDF5 writer. Scatter logic lives in StudySaver."""

    # ── lifecycle ──────────────────────────────────────────────────────────────
    def __init__(self, path: str | Path, mode: str = "w"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = h5py.File(self.path, mode)
        if mode in ("w", "w-", "x"):
            self.f.attrs["created"]       = datetime.now().isoformat()
            self.f.attrs["loudpy_format"] = "1.0"
        self._mesh_cache: dict[str, str] = {}

    def close(self):         self.f.flush(); self.f.close()
    def __enter__(self):     return self
    def __exit__(self, *exc): self.close()

    # ── metadata ───────────────────────────────────────────────────────────────
    def set_metadata(self, **kw):
        for k, v in kw.items():
            self.f.attrs[k] = (_to_jsonable(v)
                               if not isinstance(v, (int, float, str, bool, np.ndarray))
                               else v)
        self.f.attrs["updated"] = datetime.now().isoformat()

    def save_specs(self, specs: Any):
        g = self.f.require_group("meta")
        if "specs" in g: del g["specs"]
        g.create_dataset("specs", data=json.dumps(_to_jsonable(specs), indent=2))

    # ── mesh ───────────────────────────────────────────────────────────────────
    def save_mesh(self,
                  coords: np.ndarray,
                  tris: np.ndarray,
                  tags: np.ndarray,
                  per_physics: Mapping[str, dict] | None = None,
                  attrs: Mapping | None = None,
                  mesh_id: str | None = None) -> str:
        """Save a mesh; returns its mesh_id. Identical meshes are deduplicated."""
        h = _mesh_hash(coords, tris)
        if h in self._mesh_cache:
            return self._mesh_cache[h]

        mesh_id = mesh_id or f"m_{h}"
        g = self.f.require_group(f"meshes/{mesh_id}")
        if "nodes" not in g:
            g.create_dataset("nodes", data=coords, compression="gzip", shuffle=True)
            g.create_dataset("tris",  data=tris,   compression="gzip", shuffle=True)
            g.create_dataset("tags",  data=np.asarray(tags, dtype=np.int64))
            g.attrs["hash"]    = h
            g.attrs["n_nodes"] = len(tags)
            g.attrs["n_tris"]  = len(tris)

        if per_physics:
            for phys, sub in per_physics.items():
                sg = g.require_group(phys)
                for key, val in sub.items():
                    if key in sg: del sg[key]
                    sg.create_dataset(key, data=np.asarray(val),
                                      compression="gzip", shuffle=True)

        if attrs:
            for k, v in attrs.items():
                g.attrs[k] = v

        self._mesh_cache[h] = mesh_id
        return mesh_id

    def save_mesh_from_problem(self, problem, attrs: Mapping | None = None) -> str:
        """
        Extract and save mesh from a Problem:
        - global mesh (all subdomains merged)
        - per-physics submeshes keyed by class name
        - per-named-subdomain entries under subdomains/<name>
        """
        topo = MeshTopology.from_fem_objects(problem.fem_objects)

        per_physics: dict = {}
        groups: dict[str, list] = {}
        for d in problem.fem_objects:
            phys = type(d).__name__.lower() or "other"
            groups.setdefault(phys, []).append(d)

        for phys, doms in groups.items():
            try:
                sub = MeshTopology.from_fem_objects(doms)
            except ValueError:
                continue
            per_physics[phys] = {"tags": sub.tags, "tris": sub.tris,
                                  "coords": sub.coords}

        mesh_id = self.save_mesh(topo.coords, topo.tris, topo.tags,
                                 per_physics=per_physics, attrs=attrs)

        g = self.f.require_group(f"meshes/{mesh_id}/subdomains")
        for d in problem.fem_objects:
            if not hasattr(d, "tri") or not hasattr(d, "name"):
                continue
            name = d.name
            if name in g:
                continue
            sd       = g.require_group(name)
            tags     = np.asarray(d.node_tags, dtype=np.int64)
            crds     = np.asarray(d.node_coords)[:, :2]
            tri_tags = np.asarray(d.tri)[:, :3]
            tag2row  = {int(t): i for i, t in enumerate(tags)}
            tris_idx = np.vectorize(tag2row.get)(tri_tags).astype(np.int32)
            sd.create_dataset("tags",   data=tags,     compression="gzip")
            sd.create_dataset("coords", data=crds,     compression="gzip")
            sd.create_dataset("tris",   data=tris_idx, compression="gzip")
            sd.attrs["physics"] = type(d).__name__.lower()

        return mesh_id

    # ── frequency snapshots ────────────────────────────────────────────────────
    def save_snapshot(self,
                      kind: str,
                      value: float,
                      mesh_id: str,
                      fields: Mapping[str, np.ndarray],
                      attrs: Mapping | None = None,
                      idx: int | None = None) -> str:
        """Write node-indexed fields for one frequency solve."""
        sid = _snap_key(kind, value, idx=idx)
        g   = self.f.require_group(f"fields/{kind}/{sid}")
        g.attrs["mesh_id"] = mesh_id
        g.attrs["value"]   = float(value)
        if attrs:
            for k, v in attrs.items(): g.attrs[k] = v

        for name, arr in fields.items():
            if name in g: del g[name]
            arr = np.asarray(arr)
            _write_complex(g, name, arr)

        return sid

    # ── time histories ─────────────────────────────────────────────────────────
    def save_time_history(self, *, kind, time, U, V, A, input,
                          mesh_id, dpn: int = 1, attrs=None):
        """
        Write a node-indexed time-domain run.
        U/V/A must be (n_t, n_nodes) or (n_t, n_nodes, dpn).
        For raw DOF arrays use StudySaver.save_time_runs() (chunked, RAM-safe).
        """
        root = self.f.require_group(f"time_histories/{kind}")
        run  = root.create_group(f"run_{len(root.keys()):06d}")

        run.create_dataset("time", data=np.ascontiguousarray(time),
                           compression="gzip", compression_opts=4)

        for name, arr in (("U", U), ("V", V), ("A", A)):
            arr = np.ascontiguousarray(arr)
            run.create_dataset(name, data=arr,
                               compression="gzip", compression_opts=4,
                               shuffle=True,
                               chunks=(1,) + arr.shape[1:] if arr.ndim >= 2 else None)

        inp = run.create_group("input")
        inp.create_dataset("force_signal",
                           data=np.ascontiguousarray(input["force_signal"]),
                           compression="gzip", compression_opts=4)
        inp.attrs["force_amplitude"] = float(input["force_amplitude"])
        inp.attrs["force_direction"] = str(input["force_direction"])

        run.attrs["mesh_id"]      = mesh_id
        run.attrs["node_indexed"] = True
        run.attrs["dpn"]          = dpn
        for k, v in (attrs or {}).items():
            run.attrs[k] = v

    # ── modal results ──────────────────────────────────────────────────────────
    def save_modes(self,
                   kind: str,
                   freqs: np.ndarray,
                   shapes: np.ndarray,
                   mesh_id: str,
                   zeta: np.ndarray | None = None,
                   node_indexed: bool = True,
                   attrs: Mapping | None = None):
        """
        Write modal results. shapes must be (n_modes, n_nodes[, dpn]) —
        node-indexed, produced by StudySaver.save_eigen_modes().
        """
        g = self.f.require_group(f"modes/{kind}")
        for k in ("freqs", "shapes", "zeta"):
            if k in g: del g[k]
        g.create_dataset("freqs", data=np.asarray(freqs))
        _write_complex(g, "shapes", shapes)
        if zeta is not None:
            g.create_dataset("zeta", data=np.asarray(zeta))
        g.attrs["mesh_id"]      = mesh_id
        g.attrs["node_indexed"] = node_indexed
        if attrs:
            for k, v in attrs.items(): g.attrs[k] = v

    # ── interfaces ─────────────────────────────────────────────────────────────
    def save_interfaces(self, interface_meta: dict):
        gi = self.f.require_group("interfaces")
        for name, data in interface_meta.items():
            g = gi.require_group(name)
            g.attrs["kind"] = data.get("kind", "")
            tags = data.get("node_tags", np.array([]))
            if len(tags) > 0:
                if "node_tags" in g: del g["node_tags"]
                g.create_dataset("node_tags", data=np.asarray(tags, dtype=np.int64))
