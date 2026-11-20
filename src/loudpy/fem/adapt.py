"""
Mesh adaptation from error indicators: new element sizes, and a gmsh background size field.

    h_new = new_sizes(h, eta2, target, order, dim)      per cell (equidistribution, see below)
    hn = nodal_sizes(n_nodes, corner_rows, h_new)        per node: the smallest size of its cells
    write_pos(path, coords, corner_rows, hn)             a gmsh view (list format: scalar triangles / quads)
    background_field(gmsh, path)                         merges the view; returns a PostView field tag

Equidistribution: a cell's squared indicator scales as h^(2p + dim) (energy norm, order p, dimension dim).
With N cells and the target estimate T, every cell should carry T² / N: h_new = h (T² / (N η²))^(1/(2p+dim)),
the factor kept within [min_factor, max_factor] per step (no oscillation, a few steps to converge).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["new_sizes", "nodal_sizes", "write_pos", "background_field"]


def new_sizes(h, eta2, target: float, order: int, dim: int = 2, min_factor: float = 0.3,
              max_factor: float = 2.0) -> np.ndarray:
    """New cell sizes from the squared indicators eta2 for the target estimate √Σ η² = target."""
    h = np.asarray(h, float)
    eta2 = np.maximum(np.asarray(eta2, float), 1e-300)
    per_cell = target ** 2 / len(eta2)
    factor = (per_cell / eta2) ** (1.0 / (2 * order + dim))
    return h * np.clip(factor, min_factor, max_factor)


def nodal_sizes(n_nodes: int, corner_rows, h_cell) -> np.ndarray:
    """Per node the smallest size of the cells around it (corner_rows: (n_cells, n_corners) node rows)."""
    hn = np.full(n_nodes, np.inf)
    rows = np.asarray(corner_rows)
    np.minimum.at(hn, rows.ravel(), np.repeat(np.asarray(h_cell, float), rows.shape[1]))
    hn[~np.isfinite(hn)] = np.nanmax(hn[np.isfinite(hn)]) if np.isfinite(hn).any() else 1.0
    return hn


def write_pos(path, coords, corner_rows, hn) -> Path:
    """A gmsh list-format view of nodal sizes on triangles (ST) or quadrangles (SQ); 2D coords (r, z)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = np.asarray(corner_rows)
    kind = {3: "ST", 4: "SQ"}[rows.shape[1]]
    X = np.asarray(coords, float)
    lines = ['View "size" {']
    for row in rows:
        xyz = ",".join(f"{X[i, 0]:.17g},{X[i, 1]:.17g},0" for i in row)
        vals = ",".join(f"{hn[i]:.17g}" for i in row)
        lines.append(f"{kind}({xyz}){{{vals}}};")
    lines.append("};")
    path.write_text("\n".join(lines) + "\n")
    return path


def background_field(gmsh, path) -> int:
    """Merge the size view `path` into the current gmsh model; the tag of a PostView field reading it."""
    before = set(gmsh.view.getTags())
    gmsh.merge(str(path))
    new = [t for t in gmsh.view.getTags() if t not in before]
    if not new:
        raise ValueError(f"no view read from {path}")
    f = gmsh.model.mesh.field.add("PostView")
    gmsh.model.mesh.field.setNumber(f, "ViewTag", new[-1])
    return f
