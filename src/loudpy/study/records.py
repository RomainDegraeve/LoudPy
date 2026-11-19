"""
From solver records to Results (in memory, no file): the DOF vectors of a solve become node-indexed
fields on the mesh of their physics, in the node order of the assemblers (sorted node tags) —
the same arrays the former StudySaver wrote to files.
"""
from __future__ import annotations

import numpy as np

from loudpy.geometry.mesh import Mesh, MeshGroup
from loudpy.physics.registry import select
from loudpy.results.results import Block, FrequencyResults, ModeResults, TimeResults, canonical

ROLE_OF_FIELD = {"u": ("solid",), "p": ("air", "pml")}
PHYSICS_OF_FIELD = {"u": "solid", "p": "air"}


def _base(name: str) -> str:
    """Field of a DOF-map key: "u_meca" / "u_meca_m2" → "u", "p_acou" → "p"."""
    return canonical(name.split("_m")[0] if name.startswith("u_meca_m") else name)


def physics_mesh(fem_objects, roles, node_tags) -> Mesh:
    """Mesh of the regions with `roles`, nodes in the order `node_tags`, one group per region."""
    parts = select(fem_objects, *roles)
    coords = {}
    for p in parts:
        for t, c in zip(np.asarray(p.node_tags), np.asarray(p.node_coords)):
            coords[int(t)] = c[:2]
    tags = np.asarray(node_tags)
    groups = {p.name: MeshGroup(p.name, 2, "tri6", np.asarray(p.tri)) for p in parts}
    return Mesh(tags, np.array([coords[int(t)] for t in tags]), groups)


def meshes_for(fem_objects, dof_maps) -> dict[str, Mesh]:
    out = {}
    for key, dm in dof_maps.items():
        f = _base(key)
        phys = PHYSICS_OF_FIELD.get(f)
        if phys and phys not in out:
            out[phys] = physics_mesh(fem_objects, ROLE_OF_FIELD[f], dm["unique_tags"])
    return out


def nodal(raw, dm) -> np.ndarray:
    """DOF vectors (..., n_dof) → node-indexed (..., n_nodes[, dpn]) in the assembler node order."""
    raw = np.asarray(raw)
    n, dpn = len(dm["unique_tags"]), int(dm["dofs_per_node"])
    order = np.argsort(np.asarray(dm["unique_tags"]))             # identity for sorted tags
    if dpn == 1:
        return raw[..., :n][..., order]
    return raw[..., :dpn * n].reshape(raw.shape[:-1] + (n, dpn))[..., order, :]


def boundaries_of(interface_meta) -> dict:
    out = {}
    for name, d in (interface_meta or {}).items():
        tags = np.asarray(d.get("node_tags", []), dtype=np.int64)
        edges = np.asarray(d.get("edges", np.zeros((0, 3))), dtype=np.int64)
        out[name] = {"node_tags": tags, "edges": tags[edges] if len(edges) and len(tags) else edges}
    return out


def _scalars(dicts):
    keys = sorted({k for d in dicts for k in d})
    out = {}
    for k in keys:
        vals = [d.get(k, np.nan) for d in dicts]
        if all(np.isscalar(v) and not isinstance(v, (str, bytes)) for v in vals):
            out[k] = np.array(vals)
    return out


def frequency_block(records, dof_maps, fem_objects, meta=None, label="") -> Block:
    meshes = meshes_for(fem_objects, dof_maps)
    fields = {}
    for key in records[0]["fields"]:
        fields[_base(key)] = np.array([nodal(r["fields"][key], dof_maps[key]) for r in records])
    return Block(meshes, np.array([r["value"] for r in records], dtype=float), fields,
                 _scalars([r.get("attrs", {}) for r in records]), dict(meta or {}), label)


def mode_blocks(records, dof_maps, fem_objects) -> list[Block]:
    meshes = meshes_for(fem_objects, dof_maps)
    blocks = []
    for r in records:
        freqs = np.asarray(r["freqs"])
        extra = {"zeta": np.asarray(r["zeta"])} if r.get("zeta") is not None else {}
        meta = {k: v for k, v in r.get("solver_params", {}).items() if np.isscalar(v)}
        if "fields" in r:                                          # coupled modes
            fields = {_base(k): nodal(np.asarray(v) if np.shape(v)[0] == len(freqs) else np.asarray(v).T,
                                      dof_maps[k]) for k, v in r["fields"].items()}
            extra.update({k: np.asarray(v) for k, v in (r.get("extras") or {}).items()
                          if np.shape(v)[:1] == freqs.shape})
        else:
            raw = np.asarray(r["modes"])
            raw = raw if raw.shape[0] == len(freqs) else raw.T
            fields = {_base(r["dof_key"]): nodal(raw, dof_maps[r["dof_key"]])}
        kind = r["kind"]
        if kind == "meca":
            meta["order"], label = 0, "m0"
        elif kind.startswith("meca_m"):
            meta["order"], label = int(kind[6:]), kind[5:]
        else:
            label = kind
        blocks.append(Block(meshes, freqs, fields, extra, meta, label))
    blocks.sort(key=lambda b: (0, b.meta["order"]) if "order" in b.meta else (1, b.label))
    return blocks


def time_blocks(records, dof_maps, fem_objects) -> list[Block]:
    meshes = meshes_for(fem_objects, dof_maps)
    dm = dof_maps["u_meca"]
    out = []
    for i, r in enumerate(records):
        fields = {n: nodal(np.asarray(r[K]), dm) for n, K in (("u", "U"), ("v", "V"), ("a", "A"))}
        sig = np.asarray(r["input"]["force_signal"])
        extra = {"force": r["input"]["force_amplitude"] * np.real(sig)} if len(sig) == len(r["time"]) else {}
        meta = {k: v for k, v in r.get("solver_params", {}).items() if np.isscalar(v)}
        out.append(Block(meshes, np.asarray(r["time"]), fields, extra, meta, f"run{i}"))
    return out


def frequency_results(blocks, meta, boundaries):
    return FrequencyResults(blocks, meta, boundaries)


def mode_results(blocks, meta, boundaries):
    return ModeResults(blocks, meta, boundaries)


def time_results(blocks, meta, boundaries):
    return TimeResults(blocks, meta, boundaries)
