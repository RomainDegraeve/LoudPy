"""
Whole results to files read outside loudpy (res.export(path)).

    .mat   MATLAB / Octave: one struct per block, block1, block2, … with
             coordinate   (n,) frequencies / times / mode frequencies
             <field>      (n, n_nodes[, n_components]) complex amplitudes or real values
             <field>_mesh struct: coords (n_nodes, 2|3), triangles (1-based, refined TRI6), node_tags
             <quantity>   (n,) per-entry quantities (zeta, I_coil, …)
           plus info (study, kind, source)
    .npz   NumPy: the same arrays under keys "block<i>/<name>" (np.load(path)["block0/p"])
    .pvd   ParaView series: one .vtu per entry and per mesh (solid, air) in <stem>_files/, listed in the
           .pvd with the entry's frequency / time / mode as its time step (part 0, 1 … per mesh); complex
           amplitudes as <name>_re, _im (vector components _r, _z …)

The HDF5 result file itself (format 3, res.save) is also readable outside Python (h5py, MATLAB
h5read, HDFView): see the file-format page of the documentation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from loudpy.results.data import mat_name, write_atomic


def block_arrays(res, b, fields):
    out = {res.coordinate: np.asarray(b.values)}
    for name, arr in b.fields.items():
        if fields is not None and name not in fields:
            continue
        mesh = b.mesh_of(name)
        out[name] = np.asarray(arr)
        tri = mesh.triangles(refine=True) if mesh.dim == 2 else np.zeros((0, 3), np.int64)
        out[f"{name}_mesh"] = {"coords": mesh.coords, "triangles": tri, "node_tags": np.asarray(mesh.node_tags)}
    for k, v in b.extra.items():
        if np.ndim(v) >= 1:
            out[mat_name(k)] = np.asarray(v)
    return out


def export_series(res, path, fields=None):
    """ParaView series (.pvd + one .vtu per entry and mesh): see the module docstring."""
    from xml.sax.saxutils import quoteattr
    path = Path(path)
    folder = path.with_name(path.stem + "_files")
    folder.mkdir(parents=True, exist_ok=True)
    entries = []
    k = 0
    for b in res.blocks:
        by_mesh = {}
        for name in b.fields:
            if fields is None or name in fields:
                by_mesh.setdefault(id(b.mesh_of(name)), (b.mesh_of(name), []))[1].append(name)
        for i in range(len(b)):
            for part, (mesh, names) in enumerate(by_mesh.values()):
                grid = None
                for name in names:
                    f = res.field(name, index=k + i)
                    g = f.to_pyvista()
                    if grid is None:
                        grid = g
                    else:
                        for a, v in g.point_data.items():
                            grid.point_data[a] = v
                fn = folder / f"{path.stem}_{k + i:05d}_{part}.vtu"
                grid.cast_to_unstructured_grid().save(fn)
                entries.append((float(b.values[i]), part, fn.relative_to(path.parent).as_posix()))
        k += len(b)
    lines = ['<?xml version="1.0"?>', '<VTKFile type="Collection" version="0.1">', "  <Collection>"]
    lines += [f'    <DataSet timestep="{t:.10g}" part="{p}" file={quoteattr(f)}/>' for t, p, f in entries]
    lines += ["  </Collection>", "</VTKFile>", ""]
    return write_atomic(path, lambda tmp: Path(tmp).write_text("\n".join(lines)))


def export_results(res, path, fields=None):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pvd":
        return export_series(res, path, fields)
    info = {"kind": res.kind, "study": str(res.meta.get("study", "")), "source": str(res.source or ""),
            "blocks": ",".join(b.label or str(i) for i, b in enumerate(res.blocks))}
    blocks = [block_arrays(res, b, fields) for b in res.blocks]
    if suffix == ".mat":
        from scipy.io import savemat
        d = {f"block{i + 1}": blk for i, blk in enumerate(blocks)}
        d["info"] = info
        return write_atomic(path, lambda tmp: savemat(tmp, d, do_compression=True))
    if suffix == ".npz":
        flat = {"info": np.array(json.dumps(info))}
        for i, blk in enumerate(blocks):
            for k, v in blk.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        flat[f"block{i}/{k}/{kk}"] = vv
                else:
                    flat[f"block{i}/{k}"] = v
        return write_atomic(path, lambda tmp: np.savez_compressed(tmp, **flat))
    raise ValueError(f"unknown export format {suffix!r}: use .mat, .npz or .pvd (single quantities: Data.export)")
