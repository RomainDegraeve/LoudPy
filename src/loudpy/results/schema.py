"""
loudpy result format 2 (HDF5). Self-describing, independent of Python class names.

    /                       attrs: loudpy_format = "2.0", kind ("frequency" | "modes" | "time"),
                                   created, meta (JSON)
    /blocks/<i>/            attrs: label, meta (JSON)
        values              (n,) coordinate of the entries (frequency [Hz] or time [s])
        meshes/<physics>/   physics = "solid" | "air"
            node_tags       (n_nodes,)
            coords          (n_nodes, dim)
            groups/<name>   (n_cells, k) node tags;  attrs: dim, cell_type
        fields/<name>       (n, n_nodes[, dpn]), real or complex    name = "u" | "p" | "v" | "a"
        extra/<name>        (n,) per-entry quantities (zeta, I_coil, …)
    /boundaries/<name>/     node_tags, edges (node tags)

Version 1 files (former ResultStore / StudySaver) are read by loudpy.results.legacy.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

from loudpy.geometry.mesh import Mesh, MeshGroup

FORMAT = "2.0"


def _json(obj):
    def default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (complex, np.complexfloating)):
            return [float(np.real(o)), float(np.imag(o))]
        if isinstance(o, bytes):
            return o.decode(errors="replace")
        return repr(o)
    return json.dumps(obj, default=default)


def _dataset(g, name, data):
    data = np.ascontiguousarray(data)
    kw = dict(compression="gzip", compression_opts=4, shuffle=True) if data.size > 64 else {}
    return g.create_dataset(name, data=data, **kw)


def write(results, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["loudpy_format"] = FORMAT
        f.attrs["kind"] = results.kind
        f.attrs["created"] = datetime.now().isoformat()
        f.attrs["meta"] = _json(results.meta)
        for i, b in enumerate(results.blocks):
            g = f.create_group(f"blocks/{i:04d}")
            g.attrs["label"] = b.label
            g.attrs["meta"] = _json(b.meta)
            _dataset(g, "values", b.values)
            g.create_group("meshes").attrs["order"] = list(b.meshes)
            for phys, mesh in b.meshes.items():
                mg = g.create_group(f"meshes/{phys}")
                _dataset(mg, "node_tags", mesh.node_tags)
                _dataset(mg, "coords", mesh.coords)
                mg.create_group("groups").attrs["order"] = list(mesh.groups)
                for name, grp in mesh.groups.items():
                    d = _dataset(mg["groups"], name, grp.cells)
                    d.attrs["dim"] = grp.dim
                    d.attrs["cell_type"] = grp.cell_type
            g.create_group("fields").attrs["order"] = list(b.fields)
            for name, arr in b.fields.items():
                _dataset(g["fields"], name, arr)
            for name, arr in b.extra.items():
                _dataset(g.require_group("extra"), name, np.asarray(arr))
        for name, bd in results.boundaries.items():
            bg = f.create_group(f"boundaries/{name}")
            for k in ("node_tags", "edges"):
                if k in bd and len(bd[k]):
                    _dataset(bg, k, np.asarray(bd[k]))


def _ordered(group):
    """Members of a group in the order they were written (HDF5 lists them alphabetically)."""
    if group is None:
        return []
    names = [str(n) for n in group.attrs.get("order", [])] or list(group)
    return [(n, group[n]) for n in names]


def read(path):
    from loudpy.results.results import KINDS, Block
    with h5py.File(path, "r") as f:
        kind = str(f.attrs["kind"])
        blocks = []
        for key in sorted(f["blocks"]):
            g = f["blocks"][key]
            meshes = {}
            for phys, mg in _ordered(g.get("meshes")):
                groups = {n: MeshGroup(n, int(d.attrs["dim"]), str(d.attrs["cell_type"]), d[()])
                          for n, d in _ordered(mg.get("groups"))}
                meshes[phys] = Mesh(mg["node_tags"][()], mg["coords"][()], groups, str(path))
            blocks.append(Block(meshes, g["values"][()],
                                {n: d[()] for n, d in _ordered(g.get("fields"))},
                                {n: d[()] for n, d in g.get("extra", {}).items()},
                                json.loads(g.attrs.get("meta", "{}")), str(g.attrs.get("label", ""))))
        boundaries = {n: {k: bg[k][()] for k in bg} for n, bg in f.get("boundaries", {}).items()}
        meta = json.loads(f.attrs.get("meta", "{}"))
    return KINDS[kind](blocks, meta, boundaries, source=str(path))
