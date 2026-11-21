"""
The boundaries of a result on the meshes of each block.

A study stores the meshed boundaries (node tags and edges, as gmsh node tags) with each block; node tags only
mean something on the mesh they come from, and a sweep remeshed per band has one mesh per block. Files written
before blocks carried their own boundaries hold one set, valid on the last block's meshes: on another mesh
the boundary is found again from the geometry (the cell edges of that mesh lying on the stored boundary's
curve).
"""
from __future__ import annotations

import numpy as np

from loudpy.errors import ResultsError

# nodes of each cell edge, gmsh line order (corners, then the inner nodes from the first corner)
_EDGES: dict[int, list[tuple[int, ...]]] = {
    3: [(0, 1), (1, 2), (2, 0)],
    6: [(0, 1, 3), (1, 2, 4), (2, 0, 5)],
    10: [(0, 1, 3, 4), (1, 2, 5, 6), (2, 0, 7, 8)],
    4: [(0, 1), (1, 2), (2, 3), (3, 0)],
    8: [(0, 1, 4), (1, 2, 5), (2, 3, 6), (3, 0, 7)],
    9: [(0, 1, 4), (1, 2, 5), (2, 3, 6), (3, 0, 7)],
}


def boundaries_of(res, block) -> dict:
    """{name: {"node_tags", "edges"}} on the meshes of `block` (stored with it, or found again: see above)."""
    if block.boundaries:
        return block.boundaries
    if not res.boundaries:
        return {}
    ref = _reference_block(res)
    if ref is None or ref is block or _same_meshes(ref, block) or not block.meshes:
        return res.boundaries
    block.boundaries = {n: _recover(bd, ref, block) for n, bd in res.boundaries.items()}
    return block.boundaries


def _reference_block(res):
    """The block the result-level boundaries belong to: the last one with meshes."""
    with_meshes = [b for b in res.blocks if b.meshes]
    return with_meshes[-1] if with_meshes else None


def _same_meshes(a, b) -> bool:
    if set(a.meshes) != set(b.meshes):
        return False
    for k, ma in a.meshes.items():
        mb = b.meshes[k]
        if ma is mb:
            continue
        if ma.coords.shape != mb.coords.shape or not np.array_equal(ma.node_tags, mb.node_tags) \
                or not np.allclose(ma.coords, mb.coords, rtol=0, atol=1e-12 * _size(ma)):
            return False
    return True


def _size(mesh) -> float:
    c = np.asarray(mesh.coords)
    return float(np.linalg.norm(c.max(axis=0) - c.min(axis=0))) or 1.0


def _coords(block, tags) -> np.ndarray:
    """Coordinates of node tags looked up over the meshes of a block (NaN where no mesh has the tag)."""
    tags = np.asarray(tags).ravel()
    out = np.full((len(tags), 2), np.nan)
    for mesh in block.meshes.values():
        nt = np.asarray(mesh.node_tags)
        have = np.isin(tags, nt) & np.isnan(out[:, 0])
        if have.any():
            out[have] = np.asarray(mesh.coords)[mesh._rows(tags[have]), :2]
    return out


def _cell_edges(block) -> tuple[np.ndarray, np.ndarray]:
    """Every edge of the 2D cells of the block's meshes (rows of node tags, gmsh line order; one per edge) and
    the coordinates of their nodes (n_e, k, 2)."""
    edges: dict[tuple, tuple] = {}
    coords: dict[tuple, np.ndarray] = {}
    for mesh in block.meshes.values():
        for g in mesh.groups.values():
            cells = np.asarray(g.cells)
            if g.dim != 2 or cells.shape[1] not in _EDGES:
                continue
            X = np.asarray(mesh.coords)[mesh._rows(cells), :2]
            for loc in _EDGES[cells.shape[1]]:
                e = cells[:, loc]
                key = np.sort(e[:, :2], axis=1)
                for i, (a, b) in enumerate(key.tolist()):
                    if (a, b) not in edges:
                        edges[(a, b)] = tuple(e[i].tolist())
                        coords[(a, b)] = X[i, list(loc)]
    if not edges:
        return np.zeros((0, 2), np.int64), np.zeros((0, 2, 2))
    keys = list(edges)
    k = max(len(edges[c]) for c in keys)
    keys = [c for c in keys if len(edges[c]) == k]                   # one element order
    return np.array([edges[c] for c in keys], np.int64), np.array([coords[c] for c in keys])


def _segments(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Polyline segments (a, b) through the nodes of each reference edge, in order along it."""
    order = [0] + list(range(2, points.shape[1])) + [1]               # corner, inner nodes, corner
    p = points[:, order]
    return p[:, :-1].reshape(-1, 2), p[:, 1:].reshape(-1, 2)


def _distance(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance of points x (n, 2) to the nearest segment (a, b) (m, 2)."""
    d = b - a
    L2 = np.maximum(np.einsum("md,md->m", d, d), 1e-300)
    out = np.full(len(x), np.inf)
    for s in range(0, len(x), 4096):
        xs = x[s:s + 4096]
        t = np.clip(np.einsum("nmd,md->nm", xs[:, None] - a[None], d) / L2, 0.0, 1.0)
        q = a[None] + t[..., None] * d[None]
        out[s:s + 4096] = np.linalg.norm(xs[:, None] - q, axis=2).min(axis=1)
    return out


def _recover(bd: dict, ref, block) -> dict:
    """A boundary stored on the meshes of `ref`, found again on the meshes of `block`."""
    edges = np.asarray(bd.get("edges", np.zeros((0, 2), np.int64)))
    tags = np.asarray(bd.get("node_tags", np.zeros(0, np.int64)))
    if len(edges):
        P = _coords(ref, edges).reshape(edges.shape + (2,))
        if np.isnan(P).any():
            raise ResultsError("a stored boundary refers to nodes that are not in the result's meshes")
        a, b = _segments(P)
        h = float(np.median(np.linalg.norm(P[:, 1] - P[:, 0], axis=1)))
        tol = 0.05 * h
        cand, X = _cell_edges(block)
        if len(cand):
            lo, hi = P.reshape(-1, 2).min(axis=0) - tol, P.reshape(-1, 2).max(axis=0) + tol
            near = ((X >= lo) & (X <= hi)).all(axis=(1, 2))
            cand, X = cand[near], X[near]
        if len(cand):
            d = _distance(X.reshape(-1, 2), a, b).reshape(X.shape[:2])
            keep = (d <= tol).all(axis=1)
            cand = cand[keep]
        if not len(cand):
            raise ResultsError("a stored boundary could not be found again on the mesh of another block")
        return {"node_tags": np.unique(cand), "edges": cand}
    if len(tags):                                                       # nodes only: nearest nodes
        P = _coords(ref, tags)
        out = []
        for mesh in block.meshes.values():
            tol = 1e-6 * _size(mesh)
            C = np.asarray(mesh.coords)[:, :2]
            for p in P[~np.isnan(P[:, 0])]:
                i = int(np.argmin(np.linalg.norm(C - p, axis=1)))
                if np.linalg.norm(C[i] - p) <= tol:
                    out.append(int(mesh.node_tags[i]))
        return {"node_tags": np.unique(np.asarray(out, np.int64)), "edges": edges}
    return dict(bd)
