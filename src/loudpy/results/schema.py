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
    /attachments/<name>/    arrays + attrs (JSON "meta"): data a result needs besides its fields
                            (modal bases of 3D drivers, cavity modes, DtN settings …)
    attrs results_class     the Results subclass to rebuild (e.g. "SceneResults"), optional

Large datasets (more than LAZY_ELEMENTS values) are read lazily (LazyArray): 3D pressure fields
stay on disk and only the rows / columns asked for are read. StreamWriter writes a result file
entry by entry (3D sweeps), without holding the fields in memory.

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
LAZY_ELEMENTS = 5_000_000


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
    from loudpy.results.lazy import LazyArray
    if isinstance(data, LazyArray):                       # copy on disk, chunk by chunk
        with h5py.File(data.path, "r") as src:
            ds_src = src[data.name]
            ds = g.create_dataset(name, shape=ds_src.shape, dtype=ds_src.dtype, chunks=ds_src.chunks or True,
                                  compression="gzip", compression_opts=4, shuffle=True)
            step = max(1, (ds_src.chunks or (64,))[0])
            for i in range(0, ds_src.shape[0], step):
                ds[i:i + step] = ds_src[i:i + step]
        return ds
    data = np.ascontiguousarray(data)
    if data.dtype.kind == "U":
        data = data.astype("S")
    kw = dict(compression="gzip", compression_opts=4, shuffle=True) if data.size > 64 else {}
    return g.create_dataset(name, data=data, **kw)


def _write_attachments(f, attachments):
    for name, att in (attachments or {}).items():
        ag = f.require_group(f"attachments/{name}")
        for k, v in att.items():
            if k == "meta":
                ag.attrs["meta"] = _json(v)
            else:
                _dataset(ag, k, v)


def _read_attachments(f, path):
    from loudpy.results.lazy import LazyArray
    out = {}
    for name, ag in f.get("attachments", {}).items():
        att = {"meta": json.loads(ag.attrs.get("meta", "{}"))}
        for k, d in ag.items():
            if d.size > LAZY_ELEMENTS:
                att[k] = LazyArray(path, d.name)
            else:
                v = d[()]
                att[k] = v.astype("U") if getattr(v, "dtype", None) is not None and v.dtype.kind == "S" else v
        out[name] = att
    return out


def write(results, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["loudpy_format"] = FORMAT
        f.attrs["kind"] = results.kind
        f.attrs["created"] = datetime.now().isoformat()
        f.attrs["meta"] = _json(results.meta)
        f.attrs["results_class"] = type(results).__name__
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
        _write_attachments(f, getattr(results, "attachments", None))


def _ordered(group):
    """Members of a group in the order they were written (HDF5 lists them alphabetically)."""
    if group is None:
        return []
    names = [str(n) for n in group.attrs.get("order", [])] or list(group)
    return [(n, group[n]) for n in names]


def read(path):
    from loudpy.results.lazy import LazyArray
    from loudpy.results.results import KINDS, Block, result_class
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
                                {n: (LazyArray(path, d.name) if d.size > LAZY_ELEMENTS else d[()])
                                 for n, d in _ordered(g.get("fields"))},
                                {n: d[()] for n, d in g.get("extra", {}).items()},
                                json.loads(g.attrs.get("meta", "{}")), str(g.attrs.get("label", ""))))
        boundaries = {n: {k: bg[k][()] for k in bg} for n, bg in f.get("boundaries", {}).items()}
        meta = json.loads(f.attrs.get("meta", "{}"))
        attachments = _read_attachments(f, path)
        cls = result_class(str(f.attrs.get("results_class", "")), kind)
    out = cls(blocks, meta, boundaries, source=str(path))
    out.attachments = attachments
    return out


class StreamWriter:
    """
    Write a format-2 result file entry by entry (fields not held in memory):

        w = StreamWriter(path, "frequency", meta, results_class="SceneResults")
        b = w.block(label, values, meshes, fields={"p": (n_nodes,)}, dtype=np.complex64)
        w.row(b, "p", i, p_i)                       # as the solver goes
        w.extra(b, "q", q_block); w.close(attachments=…, boundaries=…, meta=…)
    The file is flushed after every row: an interrupted run keeps what it computed.
    """

    def __init__(self, path, kind, meta=None, results_class=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = h5py.File(self.path, "w")
        self.f.attrs["loudpy_format"] = FORMAT
        self.f.attrs["kind"] = kind
        self.f.attrs["created"] = datetime.now().isoformat()
        self.f.attrs["meta"] = _json(meta or {})
        if results_class:
            self.f.attrs["results_class"] = results_class
        self.n_blocks = 0

    def block(self, label, values, meshes, fields, dtype=np.complex64, meta=None) -> int:
        i = self.n_blocks
        g = self.f.create_group(f"blocks/{i:04d}")
        g.attrs["label"] = label
        g.attrs["meta"] = _json(meta or {})
        _dataset(g, "values", np.asarray(values))
        g.create_group("meshes").attrs["order"] = list(meshes)
        for phys, mesh in meshes.items():
            mg = g.create_group(f"meshes/{phys}")
            _dataset(mg, "node_tags", mesh.node_tags)
            _dataset(mg, "coords", mesh.coords)
            mg.create_group("groups").attrs["order"] = list(mesh.groups)
            for name, grp in mesh.groups.items():
                d = _dataset(mg["groups"], name, grp.cells)
                d.attrs["dim"], d.attrs["cell_type"] = grp.dim, grp.cell_type
        fg = g.create_group("fields")
        fg.attrs["order"] = list(fields)
        n = len(values)
        for name, shape in fields.items():
            full = (n,) + tuple(shape)
            chunk = (min(n, 16),) + tuple(min(s, 4096) for s in shape) if n else None
            fg.create_dataset(name, shape=full, dtype=dtype, chunks=chunk)
        self.n_blocks += 1
        return i

    def row(self, block, field, i, data):
        self.f[f"blocks/{block:04d}/fields/{field}"][i] = data
        self.f.flush()

    def extra(self, block, name, data):
        g = self.f[f"blocks/{block:04d}"].require_group("extra")
        if name in g:
            del g[name]
        _dataset(g, name, np.asarray(data))
        self.f.flush()

    def close(self, attachments=None, boundaries=None, meta=None):
        if meta is not None:
            self.f.attrs["meta"] = _json(meta)
        for name, bd in (boundaries or {}).items():
            bg = self.f.create_group(f"boundaries/{name}")
            for k in ("node_tags", "edges"):
                if k in bd and len(bd[k]):
                    _dataset(bg, k, np.asarray(bd[k]))
        _write_attachments(self.f, attachments)
        self.f.close()
