"""
Phase H2: element sizes from error indicators and remeshing with a gmsh background size field. Gate H: the
adaptive loop beats uniform refinement (L-shaped domain, corner singularity r^(2/3) sin(2θ/3), P2, energy-norm
error against the number of DOFs).
"""
import gmsh
import numpy as np

import loudpy as lp
from loudpy.fem import adapt
from loudpy.fem import estimate as est
from loudpy.fem.quadrature import quadrature
from support import current_api as api
from test_fem_assembly import make_space, solve_poisson


def test_sizes_equidistribute():
    h = np.full(4, 0.1)
    eta2 = np.array([1e-2, 1e-4, 1e-4, 1e-6])
    hn = adapt.new_sizes(h, eta2, target=np.sqrt(eta2.sum()), order=2)
    assert hn[0] < 0.1 < hn[3] and hn[1] == hn[2]                         # refine the worst, coarsen the best
    assert np.all(hn >= 0.03 - 1e-15) and np.all(hn <= 0.2 + 1e-15)       # factor within [0.3, 2] per step
    assert np.allclose(adapt.nodal_sizes(4, np.array([[0, 1, 2], [1, 2, 3]]), [0.1, 0.3]), [0.1, 0.1, 0.1, 0.3])


def exact(x):
    r, th = np.hypot(x[:, 0], x[:, 1]), np.mod(np.arctan2(x[:, 1], x[:, 0]), 2 * np.pi)
    return r ** (2 / 3) * np.sin(2 * th / 3)


def grad(x):
    r, th = np.maximum(np.hypot(x[:, 0], x[:, 1]), 1e-300), np.mod(np.arctan2(x[:, 1], x[:, 0]), 2 * np.pi)
    ur, ut = (2 / 3) * r ** (-1 / 3) * np.sin(2 * th / 3), (2 / 3) * r ** (-1 / 3) * np.cos(2 * th / 3)
    return np.column_stack([ur * np.cos(th) - ut * np.sin(th), ur * np.sin(th) + ut * np.cos(th)])


def l_shape(h, pos=None):
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)
        g = gmsh.model.geo
        pts = [g.addPoint(*c, 0, h) for c in [(0, 0), (1, 0), (1, 1), (-1, 1), (-1, -1), (0, -1)]]
        g.addPlaneSurface([g.addCurveLoop([g.addLine(pts[i], pts[(i + 1) % 6]) for i in range(6)])])
        g.synchronize()
        if pos is not None:
            gmsh.model.mesh.field.setAsBackgroundMesh(adapt.background_field(gmsh, pos))
            for o in ("Mesh.MeshSizeExtendFromBoundary", "Mesh.MeshSizeFromPoints", "Mesh.MeshSizeFromCurvature"):
                gmsh.option.setNumber(o, 0)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(2)
        tags, xyz, _ = gmsh.model.mesh.getNodes(2, -1, includeBoundary=True)
        _, nodes = gmsh.model.mesh.getElementsByType(9)
        bt = set()
        for _, e in gmsh.model.getEntities(1):
            bt |= set(gmsh.model.mesh.getNodes(1, e, includeBoundary=True)[0].tolist())
    finally:
        gmsh.finalize()
    return dict(node_tags=np.asarray(tags), coords=np.reshape(xyz, (-1, 3))[:, :2],
                cells=np.reshape(nodes, (-1, 6)).astype(np.int64), gmsh_type=9, boundary_tags=np.array(sorted(bt)))


def energy_error(sp, U):
    rule = quadrature(sp.element.cell.name, 2 * sp.element.degree + 6)
    total = 0.0
    for nodes, N, G, dx, x in est._cell_data(sp, rule):
        gh = np.einsum("eqai,ea->eqi", G, U[nodes])
        total += np.sum(dx * np.sum((grad(x.reshape(-1, 2)).reshape(gh.shape) - gh) ** 2, -1))
    return np.sqrt(total)


def solve(m):
    sp = make_space(m)
    return sp, solve_poisson(sp, lambda x: 0 * x[:, 0], exact, m["boundary_tags"])


def test_adaptive_beats_uniform(tmp_path):
    uniform = []
    for h in (0.2, 0.1, 0.05):
        sp, U = solve(l_shape(h))
        uniform.append((sp.n_dofs, energy_error(sp, U)))
    m, adaptive = l_shape(0.2), []
    for _ in range(6):
        sp, U = solve(m)
        eta2 = est.recovery_indicator(sp, U)
        adaptive.append((sp.n_dofs, energy_error(sp, U)))
        corners = np.vstack(sp.cell_nodes)[:, :3]
        hn = adapt.new_sizes(est.cell_sizes(sp), eta2, 0.5 * np.sqrt(eta2.sum()), order=2)
        adapt.write_pos(tmp_path / "size.pos", sp.node_coords, corners, adapt.nodal_sizes(sp.n_nodes, corners, hn))
        m = l_shape(0.2, tmp_path / "size.pos")
    n_u, e_u = np.array(uniform).T
    n_a, e_a = adaptive[-1]
    assert n_a < n_u[-1] and e_a < 0.2 * e_u[-1], (adaptive, uniform)    # fewer DOFs, 5× smaller error
    e_u_same = np.exp(np.interp(np.log(n_a), np.log(n_u), np.log(e_u)))
    assert e_a < 0.15 * e_u_same, (adaptive, uniform)                       # at the same number of DOFs
    rate = -np.log(adaptive[-1][1] / adaptive[-3][1]) / np.log(adaptive[-1][0] / adaptive[-3][0])
    assert rate > 0.8                                                       # → N^(-p/2) = N^(-1); uniform N^(-1/3)


def test_model_mesh_with_a_size_field(tmp_path):
    """A plate meshed with a size view that refines near the rim: more cells there; the view is part of the mesh
    signature (reused while unchanged, rebuilt when it changes)."""
    model = api.plate_model()
    base = model.mesh(tmp_path / "plate.msh")
    pos = adapt.write_pos(tmp_path / "s.pos", np.array([[0.0, -0.01], [0.12, -0.01], [0.12, 0.01], [0.0, 0.01]]),
                          np.array([[0, 1, 2], [0, 2, 3]]), np.array([2e-3, 2e-4, 2e-4, 2e-3]))
    fine = lp.Model(api.PLATE_GEO).add(*model.regions, *model.boundaries)
    mesh = fine.mesh(tmp_path / "plate_fine.msh", size_field=pos)
    assert mesh.n_nodes > base.n_nodes
    near_rim = lambda m: np.sum(m.coords[:, 0] > 0.09)                   # noqa: E731
    assert near_rim(mesh) > 3 * near_rim(base)
    stamp = (tmp_path / "plate_fine.msh").stat().st_mtime_ns
    fine.mesh(tmp_path / "plate_fine.msh", size_field=pos)
    assert (tmp_path / "plate_fine.msh").stat().st_mtime_ns == stamp           # reused
    adapt.write_pos(pos, np.array([[0.0, -0.01], [0.12, -0.01], [0.12, 0.01]]), np.array([[0, 1, 2]]),
                    np.array([2e-3, 1e-3, 1e-3]))
    fine.mesh(tmp_path / "plate_fine.msh", size_field=pos)
    assert (tmp_path / "plate_fine.msh").stat().st_mtime_ns != stamp           # rebuilt
