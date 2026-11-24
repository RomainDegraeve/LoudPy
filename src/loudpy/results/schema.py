"""
loudpy result format 3 (HDF5). Self-describing, independent of Python class names, self-contained.

    /                       attrs: loudpy_format = "3.0", kind ("frequency" | "modes" | "time" | "sweep"),
                                   created, meta (JSON)
    /model/                 attrs: description (JSON: loudpy version, geometry, parameters, regions with
                                   their materials and damping, boundaries, mesh controls, mesh, the model
                                   tree, and "groups": the physical data of every mesh group)
        files/<i>           the geometry files (uint8; attrs name) or the builder's code
    /study/                 attrs: settings (JSON: the study type and its settings)
        arrays/<name>       large settings (e.g. the drive signal of a transient)
    /blocks/<i>/            attrs: label, meta (JSON)
        values              (n,) coordinate of the entries (frequency [Hz] or time [s])
        meshes/<physics>/   physics = "solid" | "air"
            node_tags       (n_nodes,)
            coords          (n_nodes, dim)
            groups/<name>   (n_cells, k) node tags;  attrs: dim, cell_type
        fields/<name>       (n, n_nodes[, dpn]), real or complex    name = "u" | "p" | "v" | "a"
        extra/<name>        (n,) per-entry quantities (zeta, I_coil, …)
        boundaries/<name>/  node_tags, edges of the boundaries on this block's meshes (remeshed sweeps)
    /boundaries/<name>/     node_tags, edges (node tags; of the last block's meshes when blocks have their own)
    /attachments/<name>/    arrays + attrs (JSON "meta"): data a result needs besides its fields
                            (modal bases of 3D drivers, cavity modes, DtN settings …)
    attrs results_class     the Results subclass to rebuild (e.g. "SceneResults"), optional

Large datasets (more than LAZY_ELEMENTS values) are read lazily (LazyArray): 3D pressure fields
stay on disk and only the rows / columns asked for are read. StreamWriter writes a result file
entry by entry (3D sweeps), without holding the fields in memory.

A parametric sweep (kind "sweep", attrs parameter, values) holds one complete result per value in
/runs/<i>/ (same layout as the root of a single result).

Format 2 (the same without /model and /study) is still read; version 1 files (former ResultStore /
StudySaver) are read by loudpy.results.legacy.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

from loudpy.geometry.mesh import Mesh, MeshGroup

FORMAT = "3.0"
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
    """{name: {array name: array, "meta": {...}}} of the group f (the file, or a sweep run)."""
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


def _write_description(g, results) -> None:
    """/model and /study of a result (format 3), into the group g (the file, or a sweep run)."""
    desc = getattr(results, "model", None)
    if desc is not None:
        mg = g.create_group("model")
        mg.attrs["description"] = _json(desc.data)
        for i, (name, data) in enumerate(desc.files.items()):
            d = mg.require_group("files").create_dataset(f"{i:03d}", data=np.frombuffer(data, np.uint8))
            d.attrs["name"] = name
    study = dict(getattr(results, "study", None) or {})
    if study:
        arrays = study.pop("arrays", {})
        sg = g.create_group("study")
        sg.attrs["settings"] = _json(study)
        for name, arr in arrays.items():
            _dataset(sg.require_group("arrays"), name, np.asarray(arr))


def _read_description(g):
    from loudpy.results.description import ModelDescription
    model = None
    if "model" in g:
        files = {str(d.attrs["name"]): bytes(d[()]) for _, d in sorted(g["model"].get("files", {}).items())}
        model = ModelDescription(json.loads(g["model"].attrs.get("description", "{}")), files)
    study = {}
    if "study" in g:
        study = json.loads(g["study"].attrs.get("settings", "{}"))
        if "arrays" in g["study"]:
            study["arrays"] = {n: d[()] for n, d in g["study"]["arrays"].items()}
    return model, study


def _write_results(g, results) -> None:
    """One result into the group g (the file root, or /runs/<i> of a sweep)."""
    g.attrs["kind"] = results.kind
    g.attrs["meta"] = _json(results.meta)
    g.attrs["results_class"] = type(results).__name__
    for i, b in enumerate(results.blocks):
        bg = g.create_group(f"blocks/{i:04d}")
        bg.attrs["label"] = b.label
        bg.attrs["meta"] = _json(b.meta)
        _dataset(bg, "values", b.values)
        _write_meshes(bg, b.meshes)
        fields, extra = dict(b.fields), dict(b.extra)
        if b.basis:                                                  # the unit responses, not their sum
            for kind, out in (("fields", fields), ("extra", extra)):
                for name, arrays in b.basis.get(kind, {}).items():
                    out.pop(name, None)
                    for k, arr in enumerate(arrays):
                        out[name if k == 0 else f"{name}@{k}"] = arr
            bg.attrs["meta"] = _json({**b.meta, "basis": {**b.meta.get("basis", {}), **_basis_names(b.basis)}})
        bg.create_group("fields").attrs["order"] = list(fields)
        for name, arr in fields.items():
            _dataset(bg["fields"], name, arr)
        for name, arr in extra.items():
            _dataset(bg.require_group("extra"), name, np.asarray(arr))
        _write_boundaries(bg, b.boundaries)
    _write_boundaries(g, results.boundaries)
    _write_attachments(g, getattr(results, "attachments", None))
    _write_description(g, results)


def _basis_names(basis: dict) -> dict:
    """What a block keeps as unit responses: {"fields": [names], "extra": [names], "K": coils}."""
    K = max((len(a) for kind in basis.values() for a in kind.values()), default=0)
    return {"fields": list(basis.get("fields", {})), "extra": list(basis.get("extra", {})), "K": K}


def _with_basis(block):
    """A block read from a file with unit responses ("<name>", "<name>@k"): them kept, its fields their sum."""
    desc = block.meta.get("basis") or {}
    if not desc or "weights" not in block.extra:
        return block
    K = int(desc.get("K", 1))
    basis: dict = {"fields": {}, "extra": {}}
    for kind, store in (("fields", block.fields), ("extra", block.extra)):
        for name in desc.get(kind, []):
            if name in store:
                basis[kind][name] = [store.pop(name if k == 0 else f"{name}@{k}") for k in range(K)]
    block.basis = basis
    block.meta = {**block.meta, "basis": {"coils": desc.get("coils")}}       # (as in memory)
    return block.driven(block.extra["weights"])


def _write_meshes(bg, meshes) -> None:
    bg.create_group("meshes").attrs["order"] = list(meshes)
    for phys, mesh in meshes.items():
        mg = bg.create_group(f"meshes/{phys}")
        _dataset(mg, "node_tags", mesh.node_tags)
        _dataset(mg, "coords", mesh.coords)
        mg.create_group("groups").attrs["order"] = list(mesh.groups)
        for name, grp in mesh.groups.items():
            d = _dataset(mg["groups"], name, grp.cells)
            d.attrs["dim"] = grp.dim
            d.attrs["cell_type"] = grp.cell_type


def _write_boundaries(g, boundaries) -> None:
    for name, bd in (boundaries or {}).items():
        bg = g.create_group(f"boundaries/{name}")
        for k in ("node_tags", "edges"):
            if k in bd and len(bd[k]):
                _dataset(bg, k, np.asarray(bd[k]))


def _head(f, kind) -> None:
    f.attrs["loudpy_format"] = FORMAT
    f.attrs["kind"] = kind
    f.attrs["created"] = datetime.now().isoformat()


def write(results, path):
    """A result file (format 3); a SweepResults gives one file with a run per value."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        if getattr(results, "is_sweep", False):
            _head(f, "sweep")
            f.attrs["parameter"] = results.parameter                     # format 3.0 readers: the first one
            f.attrs["values"] = np.asarray([r[results.parameter] for r in results.runs], float)
            f.attrs["sweep"] = _json({"names": results.names, "combine": results.combine, "runs": results.runs})
            for i, r in enumerate(results.results):
                _write_results(f.create_group(f"runs/{i:04d}"), r)
            return
        _head(f, results.kind)
        _write_results(f, results)


def _ordered(group):
    """Members of a group in the order they were written (HDF5 lists them alphabetically)."""
    if group is None:
        return []
    names = [str(n) for n in group.attrs.get("order", [])] or list(group)
    return [(n, group[n]) for n in names]


def _read_results(g, path):
    from loudpy.results.lazy import LazyArray
    from loudpy.results.results import Block, result_class
    kind = str(g.attrs["kind"])
    blocks = []
    for key in sorted(g.get("blocks", {})):
        bg = g["blocks"][key]
        meshes = {}
        for phys, mg in _ordered(bg.get("meshes")):
            groups = {n: MeshGroup(n, int(d.attrs["dim"]), str(d.attrs["cell_type"]), d[()])
                      for n, d in _ordered(mg.get("groups"))}
            meshes[phys] = Mesh(mg["node_tags"][()], mg["coords"][()], groups, str(path))
        blocks.append(_with_basis(Block(meshes, bg["values"][()],
                            {n: (LazyArray(path, d.name) if d.size > LAZY_ELEMENTS else d[()])
                             for n, d in _ordered(bg.get("fields"))},
                            {n: d[()] for n, d in bg.get("extra", {}).items()},
                            json.loads(bg.attrs.get("meta", "{}")), str(bg.attrs.get("label", "")),
                            {n: {k: bd[k][()] for k in bd} for n, bd in bg.get("boundaries", {}).items()})))
    boundaries = {n: {k: bd[k][()] for k in bd} for n, bd in g.get("boundaries", {}).items()}
    meta = json.loads(g.attrs.get("meta", "{}"))
    cls = result_class(str(g.attrs.get("results_class", "")), kind)
    out = cls(blocks, meta, boundaries, source=str(path))
    out.attachments = _read_attachments(g, path)
    out.model, out.study = _read_description(g)
    return out


def read(path):
    """A result (or a sweep) from a format-2 or format-3 file."""
    with h5py.File(path, "r") as f:
        if str(f.attrs.get("kind", "")) == "sweep":
            from loudpy.results.sweep import SweepResults
            runs = [_read_results(f["runs"][k], path) for k in sorted(f.get("runs", {}))]
            if "sweep" in f.attrs:                                         # several parameters
                d = json.loads(f.attrs["sweep"])
                return SweepResults(d["names"], None, runs, source=str(path), combine=d["combine"], runs=d["runs"])
            return SweepResults(str(f.attrs["parameter"]), np.asarray(f.attrs["values"]), runs, source=str(path))
        return _read_results(f, path)


class StreamWriter:
    """
    Write a format-3 result file entry by entry (fields not held in memory):

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
        _head(self.f, kind)
        self.f.attrs["meta"] = _json(meta or {})
        if results_class:
            self.f.attrs["results_class"] = results_class
        self.n_blocks = 0

    def block(self, label, values, meshes, fields, dtype=np.complex64, meta=None) -> int:
        """Create a block (label, coordinate values, meshes, field shapes) whose rows are written later; returns
        its index."""
        i = self.n_blocks
        g = self.f.create_group(f"blocks/{i:04d}")
        g.attrs["label"] = label
        g.attrs["meta"] = _json(meta or {})
        _dataset(g, "values", np.asarray(values))
        _write_meshes(g, meshes)
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
        """Write entry i of a field of a block (and flush)."""
        self.f[f"blocks/{block:04d}/fields/{field}"][i] = data
        self.f.flush()

    def extra(self, block, name, data):
        """Write a per-entry array of a block (and flush)."""
        g = self.f[f"blocks/{block:04d}"].require_group("extra")
        if name in g:
            del g[name]
        _dataset(g, name, np.asarray(data))
        self.f.flush()

    def close(self, attachments=None, boundaries=None, meta=None, described=None):
        """Finish the file; `described`: an object with .model / .study (format 3 description)."""
        if meta is not None:
            self.f.attrs["meta"] = _json(meta)
        _write_boundaries(self.f, boundaries)
        _write_attachments(self.f, attachments)
        if described is not None:
            _write_description(self.f, described)
        self.f.close()
