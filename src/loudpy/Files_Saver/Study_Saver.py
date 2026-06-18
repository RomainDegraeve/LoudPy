"""
Central scatter + save logic for all loudpy study types.

ResultStore is a pure HDF5 writer.
StudySaver owns everything that requires knowledge of DOF layout:
  - scattering DOF vectors onto global mesh nodes
  - chunked time-history writing (RAM-safe)
"""
from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
import h5py

from loudpy.Files_Saver.Mesh_topology import MeshTopology
from loudpy.Files_Loader.Field_Operations_Helpers import infer_dpn

if TYPE_CHECKING:
    from loudpy.Files_Saver.Result_Store import ResultStore


class StudySaver:
    """
    Scatter + persist results for any study type.

    Parameters
    ----------
    problem : Problem
        Must have a meshed state (problem.fem_objects set).
    store : ResultStore
        Open HDF5 store to write into.
    """

    def __init__(self, problem, store: "ResultStore"):
        self._problem = problem
        self._store   = store
        self._topo    = MeshTopology.from_fem_objects(problem.fem_objects)

    # ── internal scatter helpers ───────────────────────────────────────────

    @property
    def _n_nodes(self) -> int:
        return len(self._topo.tags)

    def _rows(self, unique_tags: np.ndarray) -> np.ndarray:
        return self._topo.dof_rows(unique_tags)

    def _scatter_vector(self, raw: np.ndarray,
                        rows: np.ndarray, dpn: int) -> np.ndarray:
        """(n_dofs,) → (n_nodes,) or (n_nodes, dpn)."""
        raw = np.asarray(raw)
        n   = len(rows)
        if dpn == 1:
            out = np.zeros(self._n_nodes, dtype=raw.dtype)
            out[rows] = raw[:n]
        else:
            out = np.zeros((self._n_nodes, dpn), dtype=raw.dtype)
            out[rows] = raw[:dpn * n].reshape(n, dpn)
        return out

    def _scatter_fields(self, fields: dict, dmaps: dict) -> dict:
        """
        Scatter each field onto its own domain's node-indexed array.
        Output shape is (n_domain_nodes,) or (n_domain_nodes, dpn),
        sorted by node tag — matching what MeshLoader returns for that domain.
        """
        out = {}
        for name, raw in fields.items():
            dpn = infer_dpn(name)
            dm  = dmaps.get(name)
            raw = np.asarray(raw)
            if dm is None:
                out[name] = raw
                continue
            asm_tags = np.asarray(dm["unique_tags"])   # assembler node order
            n        = len(asm_tags)
            sort_idx = np.argsort(asm_tags)            # sort by tag → matches stored mesh
            if dpn == 1:
                out[name] = raw[:n][sort_idx]
            else:
                out[name] = raw[:dpn * n].reshape(n, dpn)[sort_idx]
        return out

    def _scatter_mode_shapes(self, raw_modes: np.ndarray,
                              asm_tags: np.ndarray, dpn: int) -> np.ndarray:
        """(n_modes, n_dofs) → (n_modes, n_domain_nodes[, dpn]), sorted by tag."""
        n_modes  = raw_modes.shape[0]
        n        = len(asm_tags)
        sort_idx = np.argsort(asm_tags)
        if dpn == 1:
            out = raw_modes[:, :n][:, sort_idx]
        else:
            out = raw_modes[:, :dpn * n].reshape(n_modes, n, dpn)[:, sort_idx, :]
        return out

    # ── public save methods ────────────────────────────────────────────────

    def save_freq_snapshots(self, results: list, dmaps: dict, mesh_id: str):
        """Scatter + write all frequency snapshots."""
        for r in results:
            node_fields = self._scatter_fields(r["fields"], dmaps)
            self._store.save_snapshot(
                kind    = r["kind"],
                value   = r["value"],
                mesh_id = mesh_id,
                fields  = node_fields,
                attrs   = r["attrs"],
            )
            self._store.save_interfaces(r["interfaces"])

    def save_eigen_modes(self, results: list, dmaps: dict, mesh_id: str):
        """Scatter + write modal results."""
        for r in results:
            dm       = dmaps[r["dof_key"]]
            dpn      = dm["dofs_per_node"]
            asm_tags = np.asarray(dm["unique_tags"])

            raw = np.asarray(r["modes"])
            # normalise to (n_modes, n_dofs)
            raw_shapes = raw if raw.shape[0] == len(r["freqs"]) else raw.T

            shapes = self._scatter_mode_shapes(raw_shapes, asm_tags, dpn)

            self._store.save_modes(
                kind         = r["kind"],
                freqs        = r["freqs"],
                shapes       = shapes,
                zeta         = r["zeta"],
                mesh_id      = mesh_id,
                node_indexed = True,
                attrs        = {
                    "dofs_per_node": dpn,
                    "n_dof_full":    dm["n_dof_full"],
                    "complex":       np.iscomplexobj(raw),
                },
            )
            self._store.save_interfaces(r["interfaces"])

    def save_time_runs(self, results: list, dmaps: dict, mesh_id: str,
                       chunk_size: int = 256):
        """
        Write time histories with chunked scatter — peak extra RAM is
        chunk_size × n_nodes × dpn instead of n_timesteps × n_nodes × dpn.
        """
        dm       = next(iter(dmaps.values())) if dmaps else None
        dpn      = dm["dofs_per_node"] if dm else 1
        asm_tags = np.asarray(dm["unique_tags"]) if dm else None
        sort_idx = np.argsort(asm_tags) if asm_tags is not None else None
        n_nodes  = len(asm_tags) if asm_tags is not None else 0

        h5 = self._store.f   # raw h5py.File — StudySaver is in the same package

        for r in results:
            time = np.asarray(r["time"])
            n_t  = len(time)

            root = h5.require_group(f"time_histories/{r['kind']}")
            idx  = len(root.keys())
            run  = root.create_group(f"run_{idx:06d}")

            run.create_dataset("time", data=np.ascontiguousarray(time),
                               compression="gzip", compression_opts=4)

            field_shape = (n_t, n_nodes) if dpn == 1 else (n_t, n_nodes, dpn)
            ch          = (min(chunk_size, n_t),) + field_shape[1:]

            for arr_name, raw in (("U", r["U"]), ("V", r["V"]), ("A", r["A"])):
                raw = np.asarray(raw)
                ds  = run.create_dataset(arr_name, shape=field_shape, dtype=raw.dtype,
                                         compression="gzip", compression_opts=4,
                                         shuffle=True, chunks=ch)
                if sort_idx is not None:
                    for start in range(0, n_t, chunk_size):
                        sl    = slice(start, min(start + chunk_size, n_t))
                        block = raw[sl]                         # (c, n_dofs)
                        c     = block.shape[0]
                        n     = n_nodes
                        if dpn == 1:
                            buf = block[:, :n][:, sort_idx]
                        else:
                            buf = block[:, :dpn*n].reshape(c, n, dpn)[:, sort_idx, :]
                        ds[sl] = buf
                else:
                    ds[:] = raw

            inp = run.create_group("input")
            inp.create_dataset("force_signal",
                               data=np.ascontiguousarray(r["input"]["force_signal"]),
                               compression="gzip", compression_opts=4)
            inp.attrs["force_amplitude"] = float(r["input"]["force_amplitude"])
            inp.attrs["force_direction"] = str(r["input"]["force_direction"])

            run.attrs["mesh_id"]      = mesh_id
            run.attrs["node_indexed"] = True
            run.attrs["dpn"]          = dpn
            for k, v in (r.get("attrs") or {}).items():
                run.attrs[k] = v

            self._store.save_interfaces(r["interfaces"])
