"""
Reader of the loudpy result format 1 (files written by the former Problem-based studies through
ResultStore / StudySaver, loudpy_format = "1.0") into Results.

Format 1 names the physics by the Python class of the physics objects ("subdomainmeca_rayleigh",
"subdomainacou") and the fields "u_meca" / "p_acou"; they become the physics "solid" / "air" and
the fields "u" / "p". Snapshots are grouped into one block per mesh.
"""
from __future__ import annotations

import json

import h5py
import numpy as np

from loudpy.geometry.mesh import Mesh, MeshGroup
from loudpy.results.results import Block, FrequencyResults, ModeResults, TimeResults, canonical
from loudpy.errors import ResultsFormatError

PHYSICS_PREFIX = {"solid": "subdomainmeca", "air": "subdomainacou"}


def _decode(arr):
    arr = np.asarray(arr)
    if arr.dtype.names and {"r", "i"} <= set(arr.dtype.names):
        return arr["r"] + 1j * arr["i"]
    return arr


def _attr(v):
    if isinstance(v, bytes):
        return v.decode(errors="replace")
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def _merged_mesh(g, prefix):
    """(unique tags, coords, triangles as rows) of the subgroups of a format-1 mesh whose name
    starts with `prefix` (the merge of the former MeshLoader: same node order)."""
    subgroups = [g[k] for k in g.keys() if isinstance(g[k], h5py.Group) and k.startswith(prefix)]
    if not subgroups and {"tags", "tris"} <= set(g.keys()):
        subgroups = [g]
    all_tags, all_coords, all_tris = [], [], []
    for sg in subgroups:
        tags = sg["tags"][:]
        coords = sg["coords" if "coords" in sg else "nodes"][:, :2]
        tris = sg["tris"][:, :3].astype(np.int64)
        all_tags.append(tags)
        all_coords.append(coords)
        all_tris.append(tris if tris.max() >= len(tags) else tags[tris])
    all_tags, all_coords, all_tris = map(np.concatenate, (all_tags, all_coords, all_tris))
    tags, first = np.unique(all_tags, return_index=True)
    row = {int(t): i for i, t in enumerate(tags)}
    idx = np.vectorize(row.get)(all_tris)
    ok = (idx[:, 0] != idx[:, 1]) & (idx[:, 1] != idx[:, 2]) & (idx[:, 0] != idx[:, 2])
    return tags, all_coords[first], idx[ok]


def _meshes(f, mesh_id) -> dict[str, Mesh]:
    """Per-physics meshes of a format-1 mesh group (same node order as the former MeshLoader)."""
    g = f[f"meshes/{mesh_id}"]
    sub = g.get("subdomains", {})
    out = {}
    for phys, prefix in PHYSICS_PREFIX.items():
        if not any(k.startswith(prefix) for k in g.keys()):
            continue
        m_tags, m_coords, m_tris = _merged_mesh(g, prefix)
        tags = m_tags
        groups = {}
        for name, sg in sub.items():
            if not str(sg.attrs.get("physics", "")).startswith(prefix):
                continue
            stags = sg["tags"][()]
            groups[name] = MeshGroup(name, 2, "tri3", stags[sg["tris"][()]])
        if not groups:
            groups["all"] = MeshGroup("all", 2, "tri3", tags[m_tris])
        out[phys] = Mesh(tags, m_coords[:, :2], groups, str(f.filename))
    return out


def _fit(arr, n_nodes):
    """Node-indexed array (n_nodes[, dpn]); pad / cut like the former scatter_dofs."""
    arr = np.asarray(arr)
    if arr.shape[0] == n_nodes:
        return arr
    out = np.zeros((n_nodes,) + arr.shape[1:], dtype=arr.dtype)
    k = min(n_nodes, arr.shape[0])
    out[:k] = arr[:k]
    return out


def _boundaries(f):
    out = {}
    for name, g in f.get("interfaces", {}).items():
        tags = g["node_tags"][()] if "node_tags" in g else np.zeros(0, np.int64)
        edges = g["edges"][()] if "edges" in g else np.zeros((0, 3), np.int64)
        out[name] = {"node_tags": tags, "edges": tags[edges] if len(edges) and len(tags) else edges}
    return out


def _file_meta(f):
    meta = {k: _attr(v) for k, v in f.attrs.items()}
    if "meta/specs" in f:
        try:
            meta["specs"] = json.loads(f["meta/specs"][()])
        except Exception:
            pass
    return meta


def read(path):
    with h5py.File(path, "r") as f:
        meta, boundaries = _file_meta(f), _boundaries(f)
        if "fields/freq" in f:
            return FrequencyResults(_frequency_blocks(f), meta, boundaries, source=str(path))
        if "modes" in f:
            return ModeResults(_mode_blocks(f), meta, boundaries, source=str(path))
        if "time_histories" in f:
            return TimeResults(_time_blocks(f), meta, boundaries, source=str(path))
    raise ResultsFormatError(f"{path}: no frequency, mode or time results found (format 1)")


def _frequency_blocks(f):
    snaps = [f["fields/freq"][k] for k in sorted(f["fields/freq"])]
    blocks, cache = [], {}
    for mesh_id in dict.fromkeys(s.attrs["mesh_id"] for s in snaps):
        sel = [s for s in snaps if s.attrs["mesh_id"] == mesh_id]
        meshes = cache.setdefault(mesh_id, _meshes(f, mesh_id))
        names = list(dict.fromkeys(n for s in sel for n in s))
        fields = {}
        for n in names:
            phys = "air" if "acou" in n else "solid"
            nn = meshes[phys].n_nodes if phys in meshes else None
            arrs = [_decode(s[n][()]) for s in sel]
            fields[canonical(n)] = np.array([_fit(a, nn) if nn else a for a in arrs])
        extra = {}
        for k in sorted({k for s in sel for k in s.attrs} - {"mesh_id", "value"}):
            vals = [s.attrs.get(k, np.nan) for s in sel]
            if all(np.isscalar(v) and not isinstance(v, (str, bytes)) for v in vals):
                extra[k] = np.array(vals)
        blocks.append(Block(meshes, np.array([float(s.attrs["value"]) for s in sel]), fields, extra,
                            {"mesh_id": str(mesh_id)}, label=str(mesh_id)))
    return blocks


def _mode_blocks(f):
    blocks, cache = [], {}
    for kind in f["modes"]:
        g = f[f"modes/{kind}"]
        attrs = {k: _attr(v) for k, v in g.attrs.items()}
        mesh_id = attrs.get("mesh_id", next(iter(f["meshes"])))
        meshes = cache.setdefault(mesh_id, _meshes(f, mesh_id))
        freqs = g["freqs"][()]
        extra = {"zeta": g["zeta"][()]} if "zeta" in g else {}
        if "fields" in g.attrs:                                    # coupled modes, several fields
            names = [str(n) for n in g.attrs["fields"]]
            fields = {canonical(n): _decode(g[f"shapes_{n}"][()]) for n in names}
            skip = {"freqs", "zeta"} | {f"shapes_{n}" for n in names}
            extra.update({n: g[n][()] for n in g if n not in skip and g[n].shape[:1] == freqs.shape})
        else:
            shapes = _decode(g["shapes"][()])
            if shapes.shape[0] != len(freqs):
                shapes = shapes.T
            name = "p" if kind == "acou" else "u"
            if not attrs.get("node_indexed", False) and "dof_map" in g:
                mesh = meshes["air" if name == "p" else "solid"]
                dpn = int(attrs.get("dofs_per_node", 1 if name == "p" else 2))
                rows = mesh.rows_of(g["dof_map"][()])
                out = np.zeros((len(freqs), mesh.n_nodes) + ((dpn,) if dpn > 1 else ()), shapes.dtype)
                out[:, rows] = shapes[:, :len(rows) * dpn].reshape((len(freqs), len(rows)) + out.shape[2:])
                shapes = out
            fields = {name: shapes}
        if kind == "meca":
            attrs["order"], label = 0, "m0"
        elif kind.startswith("meca_m") and kind[6:].isdigit():
            attrs["order"], label = int(kind[6:]), f"m{int(kind[6:])}"
        else:
            label = kind
        blocks.append(Block(meshes, freqs, fields, extra, attrs, label))
    blocks.sort(key=lambda b: (0, b.meta["order"]) if "order" in b.meta else (1, b.label))
    return blocks


def _time_blocks(f):
    blocks, cache = [], {}
    for kind, root in f["time_histories"].items():
        for run in sorted(root):
            g = root[run]
            attrs = {k: _attr(v) for k, v in g.attrs.items()}
            meshes = cache.setdefault(attrs["mesh_id"], _meshes(f, attrs["mesh_id"]))
            fields = {canonical(k): g[K][()] for k, K in (("u_meca", "U"), ("v_meca", "V"), ("a_meca", "A"))
                      if K in g}
            extra = {}
            if "input" in g:
                sig = g["input/force_signal"][()]
                if len(sig) == len(g["time"]):
                    extra["force_signal"] = sig
                attrs.update({f"input_{k}": _attr(v) for k, v in g["input"].attrs.items()})
            blocks.append(Block(meshes, g["time"][()], fields, extra, attrs, f"{kind}/{run}"))
    return blocks
