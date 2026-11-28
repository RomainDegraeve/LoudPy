"""
Results on a named part of the model (lp.On): a boundary ("coil1", "fluidstructure1" — the model's boundary items,
stored with the results) or a region ("cone", or a selector of domains), for every entry; optionally reduced to one
value per entry.

    res.probe("u", at=lp.On("fluidstructure1"))              Data over (entry, s): s = distance along the boundary [m],
                                                             nodes ordered along its edges (pieces one after another)
    res.displacement(at=lp.On("coil1", reduce="mean"), component="z")    the mean u_z of the coil over the entries
    res.pressure(at=lp.On("front", reduce="integral"))      ∫ p dA over the revolved surface [Pa·m²]
    res.probe("u", at=lp.On("cone"))                         Data over (entry, node) on a region
    res.nodes_on("fluidstructure1")                          {"node_tags", "r", "z", "s", "piece"} in the same order

reduce: "mean" and "integral" over the surface (a boundary) or the volume (a region) of revolution, weight 2πr dA
(quadratic edges by Simpson's rule, quadratic triangles by their mid-edge nodes: exact for quadratic fields up to the
r factor); "max": the largest modulus. Axisymmetric (2D) results; a 3D result is refused.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from loudpy.errors import ResultsError

REDUCE = (None, "mean", "integral", "max")


# ── the nodes of a place on one mesh ─────────────────────────────────────────
def _chain(edges) -> list[list[int]]:
    """Node tags of the edges ordered along the curves they form: one list per piece, from an end (a closed curve:
    from any node); each edge's inner nodes between its corners (gmsh line order: corners, then inner nodes)."""
    edges = [tuple(int(t) for t in e) for e in np.asarray(edges)]
    at: dict[int, list[int]] = {}
    for k, e in enumerate(edges):
        at.setdefault(e[0], []).append(k)
        at.setdefault(e[1], []).append(k)
    used, pieces = set(), []
    starts = [n for n, ks in at.items() if len(ks) == 1] + list(at)        # ends first, then loops
    for start in starts:
        if all(k in used for k in at[start]):
            continue
        piece, node = [start], start
        while True:
            nxt = next((k for k in at[node] if k not in used), None)
            if nxt is None:
                break
            used.add(nxt)
            e = edges[nxt]
            inner = list(e[2:])
            if e[0] == node:
                piece += inner + [e[1]]
                node = e[1]
            else:
                piece += inner[::-1] + [e[0]]
                node = e[0]
        pieces.append(piece)
    return pieces


def _boundary_nodes(bd, mesh):
    """(tags in order, rows on `mesh`, s, weights 2πr ds, piece of each node) of a stored boundary {"node_tags",
    "edges"} on the mesh of a field; the edges whose nodes are not all on that mesh left out."""
    edges = np.asarray(bd["edges"])
    if edges.ndim != 2 or not len(edges):
        raise ResultsError("this boundary was stored without its edges (results written before loudpy 0.2)")
    rows = mesh.rows_of(edges.ravel(), missing="ignore").reshape(edges.shape)
    edges = edges[(rows >= 0).all(axis=1)]
    if not len(edges):
        raise ResultsError("the boundary is not on the mesh of this field")
    tags, s, seen, which = [], [], 0.0, []
    for k, piece in enumerate(_chain(edges)):
        xy = np.asarray(mesh.coords)[mesh.rows_of(np.asarray(piece)), :2]
        d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
        tags += piece
        which += [k] * len(piece)
        s += list(seen + d)
        seen += d[-1]
    tags = np.asarray(tags)
    r = mesh.rows_of(tags)
    w = np.zeros(len(tags))
    pos = {int(t): i for i, t in enumerate(tags)}
    xy = np.asarray(mesh.coords)[:, :2]
    for e in edges:
        a, b = xy[mesh.rows_of(np.asarray(e[:2]))]
        length = float(np.linalg.norm(b - a))
        if len(e) == 3:                                              # Simpson: ends 1/6, middle 4/6
            parts = ((e[0], 1 / 6), (e[1], 1 / 6), (e[2], 4 / 6))
        else:
            parts = [(t, 1.0 / len(e)) for t in e]
        for t, f in parts:
            i = pos[int(t)]
            w[i] += f * length * 2 * np.pi * xy[r[i], 0]
    return tags, r, np.asarray(s), w, np.asarray(which)


def _cross(a, b):
    """The z component of a × b for 2D vectors (np.cross refuses 2D vectors since NumPy 2)."""
    return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]


_MID = {6: (3, 4, 5)}                     # quadratic triangles: the mid-edge rule (exact for quadratics)


def _region_nodes(names, mesh):
    """(tags, rows, weights 2πr dA) of the 2D mesh groups `names`."""
    xy = np.asarray(mesh.coords)[:, :2]
    weights: dict[int, float] = {}
    for n in names:
        cells = np.asarray(mesh.groups[n].cells)
        k = cells.shape[1]
        corners = xy[mesh.rows_of(cells[:, :3].ravel())].reshape(-1, 3, 2)
        area = 0.5 * np.abs(_cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]))
        if k == 4 or k in (8, 9):                                    # quadrilaterals: two triangles
            c4 = xy[mesh.rows_of(cells[:, :4].ravel())].reshape(-1, 4, 2)
            area = area + 0.5 * np.abs(_cross(c4[:, 2] - c4[:, 0], c4[:, 3] - c4[:, 0]))
        cols = _MID.get(k, tuple(range(3 if k in (3, 6, 10) else 4)))
        for j in cols:
            for t, a in zip(cells[:, j], area):
                weights[int(t)] = weights.get(int(t), 0.0) + a / len(cols)
    tags = np.unique(np.concatenate([np.asarray(mesh.groups[n].cells).ravel() for n in names]))
    rows = mesh.rows_of(tags)
    w = np.array([weights.get(int(t), 0.0) for t in tags]) * 2 * np.pi * xy[rows, 0]
    return tags, rows, w


def _place(res, block, on, mesh):
    """("boundary" | "region", label, tags, rows, s or None, weights, piece or None) of `on` on the mesh of a field
    in a block."""
    from loudpy.geometry.naming import matches
    if np.asarray(mesh.coords).shape[1] != 2:
        raise ResultsError("lp.On: results on a named part are for axisymmetric (2D) results")
    if isinstance(on, str):
        bds = res.boundaries_of(block) or res.boundaries
        hit = [n for n in bds if n == on] or [n for n in bds if matches(on, n)]
        if hit:
            if len(hit) > 1:
                bd = {"edges": np.concatenate([np.asarray(bds[n]["edges"]) for n in hit])}
            else:
                bd = bds[hit[0]]
            tags, rows, s, w, piece = _boundary_nodes(bd, mesh)
            return "boundary", on, tags, rows, s, w, piece
        names = [n for n in mesh.names(2) if n == on] or [n for n in mesh.names(2) if matches(on, n)]
        if not names:
            raise ResultsError(f"lp.On: no boundary or region {on!r} (boundaries: {list(bds)}; regions: "
                               f"{mesh.names(2)})")
    else:
        from loudpy.geometry.selectors import Domains, as_selector
        sel = as_selector(on).domains(Domains.from_mesh(mesh))
        names = [n for n in mesh.names(2) if n in sel]
        if not names:
            raise ResultsError(f"lp.On: {on!r} selects no region of this mesh ({mesh.names(2)})")
        on = repr(as_selector(on))
    tags, rows, w = _region_nodes(names, mesh)
    return "region", on, tags, rows, None, w, None


# ── evaluation ───────────────────────────────────────────────────────────────
def _read(arr, rows):
    """arr[:, rows] (arrays or lazy HDF5 datasets: read at sorted rows, then put in order)."""
    order = np.unique(rows)
    sub = np.asarray(arr[:, order])
    return sub[:, np.searchsorted(order, rows)]


def evaluate(res, name: str, on, reduce: str | None = None, component: str | None = None):
    """A field on a named part for every entry (module docstring): lp.Data."""
    from loudpy.results.data import Data
    from loudpy.results.quantities import COMPONENTS, quantity
    if reduce not in REDUCE:
        raise ResultsError(f"lp.On: reduce is one of {REDUCE}, got {reduce!r}")
    parts, kind, label, first = [], None, "", None
    for b in res.blocks:
        if name not in b.fields:
            raise ResultsError(f"no field {name!r} in block {b.label!r} (fields: {list(b.fields)})")
        mesh = b.mesh_of(name)
        kind, label, tags, rows, s, w, _ = _place(res, b, on, mesh)
        v = _read(b.fields[name], rows)                               # (n, nodes[, dpn])
        if reduce is None:
            if first is not None and (len(tags) != len(first[0]) or not np.array_equal(tags, first[0])):
                raise ResultsError(f"lp.On({label!r}): the blocks lie on different meshes (a sweep remeshed per "
                                   f"band): give reduce= (one value per entry), or evaluate one block at a time")
            first = first or (tags, s)
        else:
            if v.ndim == 3:
                w3 = w[None, :, None]
            else:
                w3 = w[None, :]
            if reduce == "max":
                v = np.abs(v).max(axis=1)
            elif reduce == "integral":
                v = (v * w3).sum(axis=1)
            else:
                v = (v * w3).sum(axis=1) / max(float(w.sum()), 1e-300)
        parts.append(v)
    width = max(p.shape[-1] for p in parts) if parts[0].ndim == (3 if reduce is None else 2) else 0
    if width:                                                         # vector fields: pad the missing u_theta
        parts = [np.concatenate([p, np.zeros(p.shape[:-1] + (width - p.shape[-1],), p.dtype)], axis=-1)
                 if p.shape[-1] < width else p for p in parts]
    v = np.concatenate(parts, axis=0)
    if res.real:
        v = np.real(v)
    q = quantity(name)
    vector = v.ndim == (3 if reduce is None else 2)
    if vector and component is not None:
        v, q = v[..., COMPONENTS[component]], quantity(f"{name}_{component}")
        vector = False
    coords: dict = {res.dim: res._coord()}
    if reduce is None and first is not None:
        tags, s = first
        coords["s" if kind == "boundary" else "node"] = s if kind == "boundary" else tags
    if vector:
        coords["component"] = np.array(["r", "z", "theta"][:v.shape[-1]])
    if reduce == "max":
        q = q.part("abs") if q.complex else q
        q = dataclasses.replace(q, label=f"max |{q.label}|")
    elif reduce == "integral":
        unit = "m²" if kind == "boundary" else "m³"
        q = dataclasses.replace(q, name=f"∫{q.name}", label=f"∫ {q.label} d{'A' if kind == 'boundary' else 'V'}",
                                unit=f"{q.unit}·{unit}" if q.unit else unit)
    elif reduce == "mean":
        q = dataclasses.replace(q, label=f"mean {q.label}")
    meta = res._meta(on=label, part=kind, **({"reduce": reduce} if reduce else {}))
    return Data(v, coords, q, label=f"{label}" + (f" ({reduce})" if reduce else ""), meta=meta)


def nodes_on(res, on, field: str | None = None, block: int = 0) -> dict:
    """The nodes of a named part as lp.On evaluates them (same order): node_tags, r, z, s and piece (a boundary: the
    distance along it, the index of each separate piece; None for a region), on the mesh of `field` (default: the
    first field) in `block`."""
    b = res.blocks[block]
    name = field or next(iter(b.fields))
    mesh = b.mesh_of(name)
    kind, label, tags, rows, s, w, piece = _place(res, b, on, mesh)
    xy = np.asarray(mesh.coords)[rows, :2]
    return {"node_tags": tags, "r": xy[:, 0], "z": xy[:, 1], "s": s, "piece": piece, "part": kind, "on": label}
