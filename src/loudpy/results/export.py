"""
Whole results to files read outside loudpy (res.export(path)).

    .mat   MATLAB / Octave: one struct per block, block1, block2, … with
             coordinate   (n,) frequencies / times / mode frequencies
             <field>      (n, n_nodes[, n_components]) complex amplitudes or real values
             <field>_mesh struct: coords (n_nodes, 2|3), triangles (1-based, refined TRI6), node_tags
             <quantity>   (n,) per-entry quantities (zeta, I_coil, …)
           plus info (study, kind, source)
    .npz   NumPy: the same arrays under keys "block<i>/<name>" (np.load(path)["block0/p"])

The HDF5 result file itself (format 2, res.save) is also readable outside Python (h5py, MATLAB
h5read, HDFView): see the file-format page of the documentation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from loudpy.results.data import _mat_name, _write_atomic


def _block_arrays(res, b, fields):
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
            out[_mat_name(k)] = np.asarray(v)
    return out


def export_results(res, path, fields=None):
    path = Path(path)
    suffix = path.suffix.lower()
    info = {"kind": res.kind, "study": str(res.meta.get("study", "")), "source": str(res.source or ""),
            "blocks": ",".join(b.label or str(i) for i, b in enumerate(res.blocks))}
    blocks = [_block_arrays(res, b, fields) for b in res.blocks]
    if suffix == ".mat":
        from scipy.io import savemat
        d = {f"block{i + 1}": blk for i, blk in enumerate(blocks)}
        d["info"] = info
        return _write_atomic(path, lambda tmp: savemat(tmp, d, do_compression=True))
    if suffix == ".npz":
        flat = {"info": np.array(json.dumps(info))}
        for i, blk in enumerate(blocks):
            for k, v in blk.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        flat[f"block{i}/{k}/{kk}"] = vv
                else:
                    flat[f"block{i}/{k}"] = v
        return _write_atomic(path, lambda tmp: np.savez_compressed(tmp, **flat))
    raise ValueError(f"unknown export format {suffix!r}: use .mat or .npz (single quantities: Data.export)")
