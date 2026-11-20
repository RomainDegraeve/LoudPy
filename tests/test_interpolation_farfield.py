"""
Phase G2: interpolation exact for the mesh's own elements (2D, quadrilaterals, tetrahedra, curved cells), plain
arrays, and the far field from a result file alone (2D axisymmetric Kirchhoff–Helmholtz vs analytical monopoles,
and vs the FE field of the speaker inside the mesh). The 3D scene far field from the file alone: test_scene.
"""
import gmsh
import numpy as np
import pytest

import loudpy as lp
from loudpy.fem.reference import cell_type_name
from loudpy.geometry.mesh import Mesh, MeshGroup
from loudpy.results.results import Block, FrequencyResults
from support import current_api as api
from support import fem_meshes as fm


def as_mesh(d):
    dim, k = d["coords"].shape[1], d["cells"].shape[1]
    return Mesh(d["node_tags"], d["coords"], {"g": MeshGroup("g", dim, cell_type_name(dim, k), d["cells"])})


@pytest.mark.parametrize("case, p", [("tri", 1), ("tri", 2), ("tri", 3), ("quad", 1), ("quad", 2),
                                     ("tet", 1), ("tet", 2), ("tet", 3)])
def test_interpolation_is_exact_for_the_elements(case, p):
    """A polynomial of the element's space is interpolated exactly anywhere in the cells (values and gradients)."""
    d = {"tri": lambda: fm.unit_square(3, p), "quad": lambda: fm.unit_square(3, p, quads=True),
         "tet": lambda: fm.unit_cube(0.5, p)}[case]()
    mesh = as_mesh(d)
    dim = mesh.dim
    rng = np.random.default_rng(p)
    pts = rng.uniform(0.01, 0.99, (60, dim))
    e = np.array([p] + [0] * (dim - 1)) if case != "quad" else np.array([p, p])

    def f(x):                                                   # x^p (tri, tet) or x^p y^p (quad) + a linear part
        return np.prod(x ** e, axis=1) + 0.5 * x[:, -1] - 2.0

    def df(x):
        g = np.zeros_like(x)
        for j in range(dim):
            ej = e.copy()
            if ej[j]:
                ej[j] -= 1
                g[:, j] = e[j] * np.prod(x ** ej, axis=1)
        g[:, -1] += 0.5
        return g
    W, G = mesh.interpolation(pts, gradient=True)
    v = f(mesh.coords)
    np.testing.assert_allclose(W @ v, f(pts), atol=1e-12)
    np.testing.assert_allclose(np.column_stack([g @ v for g in G]), df(pts), atol=1e-10)


def test_interpolation_on_curved_cells_nodes_and_outside():
    mesh = as_mesh(fm.quarter_annulus(0.2, 2))                   # P2 cells curved on the arcs
    rng = np.random.default_rng(0)
    ang, rad = rng.uniform(0.0, np.pi / 2, 100), rng.uniform(1.0, 2.0, 100)
    pts = np.column_stack([rad * np.cos(ang), rad * np.sin(ang)])
    W, G = mesh.interpolation(pts, gradient=True)
    x, y = mesh.coords.T
    np.testing.assert_allclose(W @ (3 * x - 2 * y), 3 * pts[:, 0] - 2 * pts[:, 1], atol=1e-12)   # isoparametric
    np.testing.assert_allclose(G[0] @ (3 * x - 2 * y), 3.0, atol=1e-10)
    k = [5, 17]
    W = mesh.interpolation(mesh.coords[k])                       # on nodes: exactly the node values
    assert W.nnz == 2 and np.array_equal(W.indices, k) and np.all(W.data == 1.0)
    W = mesh.interpolation([[0.1, 0.1]])                         # inside the hole: the nearest node
    assert W.nnz == 1 and W.indices[0] == mesh.nearest([0.1, 0.1])


def annulus(r_in, r_out, h, baffled):
    """Air around a source at the origin: half annulus (r ≥ 0) or, above a baffle at z = 0, a quarter annulus."""
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)
        g = gmsh.model.geo
        c = g.addPoint(0, 0, 0)
        if baffled:
            a, b, t, u = g.addPoint(r_in, 0, 0), g.addPoint(r_out, 0, 0), g.addPoint(0, r_out, 0), g.addPoint(0, r_in, 0)
            loop = [g.addLine(a, b), g.addCircleArc(b, c, t), g.addLine(t, u), g.addCircleArc(u, c, a)]
        else:
            t1, t2, e1, e2 = (g.addPoint(0, r_out, 0), g.addPoint(0, r_in, 0), g.addPoint(0, -r_in, 0),
                              g.addPoint(0, -r_out, 0))
            m1, m2 = g.addPoint(r_out, 0, 0), g.addPoint(r_in, 0, 0)
            loop = [g.addCircleArc(t1, c, m1), g.addCircleArc(m1, c, e2), g.addLine(e2, e1),
                    g.addCircleArc(e1, c, m2), g.addCircleArc(m2, c, t2), g.addLine(t2, t1)]
        g.addPlaneSurface([g.addCurveLoop(loop)])
        g.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        gmsh.option.setNumber("Mesh.MeshSizeMin", h)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(2)
        tags, xyz, _ = gmsh.model.mesh.getNodes(2, -1, includeBoundary=True)    # not the arcs' centre
        _, nodes = gmsh.model.mesh.getElementsByType(9)
    finally:
        gmsh.finalize()
    return Mesh(np.asarray(tags), np.reshape(xyz, (-1, 3))[:, :2],
                {"air": MeshGroup("air", 2, "tri6", np.reshape(nodes, (-1, 6)))})


@pytest.mark.parametrize("baffled", [False, True])
def test_far_field_of_monopoles_from_the_file(tmp_path, baffled):
    """Analytical monopole (above a rigid baffle: plus its image) on the nodes of an air mesh, saved and read
    back: the Kirchhoff–Helmholtz far field at 3 m equals the analytical field."""
    c, f = 343.0, np.array([200.0, 1000.0])
    k = 2 * np.pi * f / c
    sources = [(0.0, 0.02), (0.0, -0.02)] if baffled else [(0.0, 0.0)]

    def p(x):
        d = [np.linalg.norm(x - np.array(s), axis=1) for s in sources]
        return sum(np.exp(-1j * k[:, None] * r) / r for r in d)
    mesh = annulus(0.05, 0.5, 0.01, baffled)
    path = FrequencyResults([Block({"air": mesh}, f, {"p": p(mesh.coords)})]).save(tmp_path / "monopole.h5")
    res = lp.load(path)
    arc = lp.Arc(3.0, np.arange(0.0, 91.0 if baffled else 181.0, 15.0))
    ff = np.asarray(res.far_field(arc, radius=0.3, baffle=0.0 if baffled else None, c=c))
    exact = p(arc.points(2))
    err = np.abs(ff - exact).max(axis=1) / np.abs(exact).max(axis=1)
    assert err[0] < 1e-4 and err[1] < 1e-3, err
    with pytest.raises(lp.errors.SelectionError, match="outside the integration sphere"):
        res.far_field([(0.0, 0.2)], radius=0.3, c=c)
    with pytest.raises(lp.errors.SelectionError, match="leaves the mesh"):
        res.far_field([(0.0, 3.0)], radius=0.7, c=c)


def test_far_field_of_the_speaker(mesh_dir):
    """The coupled speaker (front air above the baffle z = 0.032): the far field from spheres of two radii
    reproduces the FE pressure at points still inside the mesh, and does not depend on the radius."""
    model = api.speaker_problem(mesh_dir, acoustic=True)
    res = lp.FrequencyResponse(model, drive=lp.Force(1.0)).solve([500.0, 1500.0])
    arc = lp.Arc(0.26, np.arange(0.0, 81.0, 10.0), center=(0.0, 0.032))
    fe = np.asarray(res.pressure(arc))
    ff = [np.asarray(res.far_field(arc, radius=r, center=(0.0, 0.032), baffle=0.032)) for r in (0.12, 0.2)]
    for x in ff:
        assert (np.abs(x - fe).max(axis=1) / np.abs(fe).max(axis=1)).max() < 3e-3
    assert (np.abs(ff[0] - ff[1]).max(axis=1) / np.abs(ff[1]).max(axis=1)).max() < 3e-3
    assert res.far_field(lp.Arc(1.0, [0, 45], center=(0, 0.032)), radius=0.2, center=(0, 0.032),
                         baffle=0.032).dims == ("f", "angle")


def test_plain_arrays(mesh_dir):
    res = lp.FrequencyResponse(api.speaker_problem(mesh_dir), drive=lp.Force(1.0)).solve([100.0, 200.0])
    a = res.arrays()
    assert set(a) >= {"frequency", "u", "u_mesh"} and a["u"].shape == (2,) + res.blocks[0].fields["u"].shape[1:]
    np.testing.assert_array_equal(a["frequency"], [100.0, 200.0])
    assert a["u_mesh"]["coords"].shape[1] == 2
