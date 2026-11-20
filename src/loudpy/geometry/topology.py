"""
Topology of a 2D mesh, from its cells only (no names): which region lies on which side of an edge.

    sides = EdgeSides(region_parts)               triangles of the regions, indexed by their edges
    sides.outward(edges, node_xy)                 per edge: (region index, sign) with sign = +1 when the
                                                  edge normal (dz, −dr) from its node order points out
                                                  of that region, −1 otherwise

An edge is identified by its two corner nodes (the mid-side node of a LINE3 / TRI6 edge is not
needed). Used to orient the fluid–structure coupling from the geometry: the normal of every
coupling edge is the outward normal of the air next to it.
"""
from __future__ import annotations

import numpy as np

from loudpy.errors import MeshError


def edge_key(a, b) -> tuple[int, int]:
    a, b = int(a), int(b)
    return (a, b) if a < b else (b, a)


class EdgeSides:
    """The triangles of a list of region parts (objects with node_tags, node_coords, tri), by edge."""

    def __init__(self, regions):
        self.regions = list(regions)
        self._sides: dict[tuple[int, int], list[tuple[int, np.ndarray]]] = {}
        for k, part in enumerate(self.regions):
            tri = np.asarray(part.tri)
            if not len(tri):
                continue
            rows = {int(t): i for i, t in enumerate(np.asarray(part.node_tags))}
            xy = np.asarray(part.node_coords)[:, :2]
            corners = tri[:, :3]
            for i in range(3):
                a, b, o = corners[:, i], corners[:, (i + 1) % 3], corners[:, (i + 2) % 3]
                for aa, bb, oo in zip(a, b, o):
                    self._sides.setdefault(edge_key(aa, bb), []).append((k, xy[rows[int(oo)]]))

    def regions_of(self, a, b) -> list[int]:
        """Indices of the regions with a triangle on the edge (a, b)."""
        return [k for k, _ in self._sides.get(edge_key(a, b), [])]

    def outward(self, edges, node_xy, where=None) -> tuple[np.ndarray, np.ndarray]:
        """
        For each edge (rows of node tags, corners first) with coordinates node_xy[tag] → (r, z):
        the index of the region on it and the sign of (edge normal · outward normal of that region).
        `where(k)` restricts the regions considered (e.g. the air regions); exactly one must touch
        each edge, otherwise MeshError.
        """
        edges = np.asarray(edges)
        region = np.empty(len(edges), np.int64)
        sign = np.empty(len(edges), np.float64)
        for i, e in enumerate(edges):
            a, b = int(e[0]), int(e[1])
            found = [(k, third) for k, third in self._sides.get(edge_key(a, b), []) if where is None or where(k)]
            if len(found) != 1:
                what = "no" if not found else f"{len(found)}"
                raise MeshError(f"edge ({a}, {b}) at {np.round(node_xy(a), 6)}: {what} adjacent region(s) of the "
                                f"required kind (expected exactly one)")
            k, third = found[0]
            x0, x1 = node_xy(a), node_xy(b)
            n_edge = np.array([x1[1] - x0[1], -(x1[0] - x0[0])])
            sign[i] = 1.0 if n_edge @ (0.5 * (x0 + x1) - third) > 0 else -1.0
            region[i] = k
        return region, sign
