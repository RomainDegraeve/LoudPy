"""
lp.load: read results of any study, any format, one file or many.

    lp.load("sweep.h5")                        one file (format 3 or 2, or format 1 of the former studies);
                                               a parametric sweep file gives a SweepResults
    lp.load("Results/Files")                   every .h5 of a folder (e.g. one file per mesh block)
    lp.load("Results/Files/snap_*.h5")         a glob
    lp.load([path1, path2])                    an explicit list (kept in this order)
    lp.load(path, rename={"old": "new"})       group and item names renamed on read (files written before a
                                               geometry's groups were renamed); the file is not changed

Several files are merged into one Results of the same kind: their blocks are concatenated,
sorted by their first coordinate (frequency or time) unless an explicit list is given.
"""
from __future__ import annotations

import glob as _glob
from pathlib import Path

import h5py

from loudpy.results import legacy, schema
from loudpy.results.results import KINDS
from loudpy.errors import ResultsFormatError


def _read_one(path):
    with h5py.File(path, "r") as f:
        fmt = str(f.attrs.get("loudpy_format", "1.0"))
    if fmt.startswith(("2", "3")):
        return schema.read(path)
    return legacy.read(path)


def _paths(source) -> tuple[list[Path], bool]:
    if isinstance(source, (list, tuple)):
        return [Path(p) for p in source], False
    s = str(source)
    if any(c in s for c in "*?["):
        return [Path(p) for p in sorted(_glob.glob(s))], True
    p = Path(s)
    if p.is_dir():
        return sorted(p.glob("*.h5")), True
    return [p], False


def rename_groups(res, mapping: dict):
    """Rename, in place, mesh groups and item names (boundaries, regions, physical data of the groups) of results
    read from a file: {old name: new name}; names not in the mapping are kept. Returns the results."""
    from dataclasses import replace
    if not mapping:
        return res
    if getattr(res, "is_sweep", False):
        for r in res.results:
            rename_groups(r, mapping)
        return res

    def ren(d):
        return {mapping.get(k, k): v for k, v in d.items()}
    res.boundaries = ren(res.boundaries)
    seen = set()
    for b in res.blocks:
        b.boundaries = ren(b.boundaries)
        for m in b.meshes.values():
            if id(m) not in seen:
                seen.add(id(m))
                m.groups = {mapping.get(k, k): replace(g, name=mapping.get(g.name, g.name)) for k, g in m.groups.items()}
    if res.model is not None:
        data = res.model.data
        data["groups"] = ren(data.get("groups", {}))
        for key in ("regions", "boundaries"):
            for item in data.get(key, []) or []:
                if isinstance(item, dict) and item.get("name") in mapping:
                    item["name"] = mapping[item["name"]]
    return res


def load(source, rename: dict | None = None):
    """Results from a file, a folder, a glob or a list of files (see the module docstring); rename: {old: new}
    group and item names (see rename_groups())."""
    if rename:
        return rename_groups(load(source), rename)
    paths, sort = _paths(source)
    if not paths:
        raise FileNotFoundError(f"no result file matches {source!r}")
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"result file(s) not found: {[str(m) for m in missing]}")
    parts = [_read_one(p) for p in paths]
    if len(parts) == 1:
        return parts[0]
    if any(getattr(r, "is_sweep", False) for r in parts):
        raise ResultsFormatError("sweep files are read one at a time (lp.load(path))")
    kinds = {r.kind for r in parts}
    if len(kinds) != 1:
        raise ResultsFormatError(f"cannot merge results of different kinds: {sorted(kinds)}")
    blocks = [b for r in parts for b in r.blocks]
    if sort:
        blocks.sort(key=lambda b: float(b.values[0]) if len(b.values) else 0.0)
    boundaries = {k: v for r in parts for k, v in r.boundaries.items()}
    out = KINDS[kinds.pop()](blocks, parts[0].meta, boundaries, source=[str(p) for p in paths])
    out.model, out.study = parts[0].model, parts[0].study
    return out
