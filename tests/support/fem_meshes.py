"""
Small in-memory gmsh meshes for the FE-core tests (nothing written to disk, single thread).

    m = unit_square(n=8, order=2, quads=False)     # structured (transfinite) [0, 1]²
    m = unit_cube(h=0.25, order=3)                 # unstructured [0, 1]³ tetrahedra
    m = quarter_annulus(h=0.25, order=2, curved=True)   # 1 ≤ r ≤ 2, curved arcs (or straight-sided)

Each returns a dict: node_tags, coords (n, gdim), cells (n_e, nodes) of the top-dimension cells, gmsh_type,
boundary_tags (nodes on the boundary), boundary_cells (cells of the boundary entities), boundary_type.
"""
from __future__ import annotations

import numpy as np


def _start(gmsh):
    gmsh.initialize(readConfigFiles=False)
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.NumThreads", 1)
    gmsh.option.setNumber("Mesh.MaxNumThreads1D", 1)
    gmsh.option.setNumber("Mesh.MaxNumThreads2D", 1)
    gmsh.option.setNumber("Mesh.MaxNumThreads3D", 1)
    gmsh.option.setNumber("Mesh.RandomFactor", 1e-9)
    gmsh.model.add("fem_test")


def _collect(gmsh, dim, order, curved=True):
    gmsh.option.setNumber("Mesh.SecondOrderLinear", 0 if curved else 1)
    gmsh.model.mesh.generate(dim)
    gmsh.model.mesh.setOrder(order)
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    coords = np.reshape(xyz, (-1, 3))[:, :dim]
    types = [t for t in gmsh.model.mesh.getElementTypes(dim)]
    assert len(types) == 1, f"mixed cell types {types}"
    etype = int(types[0])
    _, nodes = gmsh.model.mesh.getElementsByType(etype)
    props = gmsh.model.mesh.getElementProperties(etype)
    cells = np.reshape(nodes, (-1, props[3])).astype(np.int64)
    btags, bcells, btype = [], [], None
    for _, ent in gmsh.model.getEntities(dim - 1):
        t, _, _ = gmsh.model.mesh.getNodes(dim - 1, ent, includeBoundary=True)
        btags.append(t)
        et, _, en = gmsh.model.mesh.getElements(dim - 1, ent)
        if len(et):
            btype = int(et[0])
            nb = gmsh.model.mesh.getElementProperties(btype)[3]
            bcells.append(np.reshape(en[0], (-1, nb)).astype(np.int64))
    return dict(node_tags=np.asarray(tags, np.int64), coords=np.ascontiguousarray(coords), cells=cells,
                gmsh_type=etype, boundary_tags=np.unique(np.concatenate(btags)).astype(np.int64),
                boundary_cells=np.vstack(bcells), boundary_type=btype)


def unit_square(n: int, order: int, quads: bool = False):
    import gmsh
    _start(gmsh)
    try:
        s = gmsh.model.occ.addRectangle(0, 0, 0, 1, 1)
        gmsh.model.occ.synchronize()
        for _, c in gmsh.model.getEntities(1):
            gmsh.model.mesh.setTransfiniteCurve(c, n + 1)
        gmsh.model.mesh.setTransfiniteSurface(s)
        if quads:
            gmsh.model.mesh.setRecombine(2, s)
        return _collect(gmsh, 2, order)
    finally:
        gmsh.finalize()


def unit_cube(h: float, order: int):
    import gmsh
    _start(gmsh)
    try:
        gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        gmsh.option.setNumber("Mesh.MeshSizeMin", h)
        return _collect(gmsh, 3, order)
    finally:
        gmsh.finalize()


def quarter_annulus(h: float, order: int, curved: bool = True):
    import gmsh
    _start(gmsh)
    try:
        g = gmsh.model.geo
        c = g.addPoint(0, 0, 0)
        p = [g.addPoint(1, 0, 0), g.addPoint(2, 0, 0), g.addPoint(0, 2, 0), g.addPoint(0, 1, 0)]
        lines = [g.addLine(p[0], p[1]), g.addCircleArc(p[1], c, p[2]), g.addLine(p[2], p[3]),
                 g.addCircleArc(p[3], c, p[0])]
        g.addPlaneSurface([g.addCurveLoop(lines)])
        g.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        gmsh.option.setNumber("Mesh.MeshSizeMin", h)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        return _collect(gmsh, 2, order, curved)
    finally:
        gmsh.finalize()


def ball(h: float, order: int = 2):
    """Unit ball (OCC sphere), unstructured; returns nodes (n, 3), tets and boundary triangles as row
    indices (the layout of the 3D toolkit, loudpy.fem.tet10)."""
    import gmsh
    _start(gmsh)
    try:
        gmsh.model.occ.addSphere(0, 0, 0, 1)
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        gmsh.option.setNumber("Mesh.MeshSizeMin", h)
        m = _collect(gmsh, 3, order)
    finally:
        gmsh.finalize()
    rows = {int(t): i for i, t in enumerate(m["node_tags"])}
    to_rows = np.vectorize(rows.get)
    return m["coords"], to_rows(m["cells"]), to_rows(m["boundary_cells"])
