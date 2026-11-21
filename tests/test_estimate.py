"""
Phase H1: a-posteriori error estimators (loudpy.fem.estimate). Gate H: the recovery estimator's effectivity
(estimate / true energy-norm error) lies in [0.8, 1.5] on manufactured solutions (P1–P3 triangles, Q1–Q2,
curved unstructured cells, axisymmetric); the residual indicator converges at the error's rate; the elastic
energy indicator is exact on fields its recovery reproduces and asymptotically exact on smooth ones.
"""
import numpy as np
import pytest

from loudpy.fem import estimate as est
from loudpy.fem.quadrature import quadrature
from loudpy.results.derived import elasticity
from support import fem_meshes as fm
from test_fem_assembly import f2, f_axi, make_space, solve_poisson, u2, u_axi


def du2(x):
    return np.column_stack([np.pi * np.cos(np.pi * x[:, 0]) * np.exp(x[:, 1]), np.sin(np.pi * x[:, 0]) * np.exp(x[:, 1])])


def du_axi(x):
    return np.column_stack([-np.pi * np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1]),
                            np.pi * np.cos(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])])


def energy_error(space, U, grad, axisym=False):
    rule = quadrature(space.element.cell.name, 2 * space.element.degree + 4)
    total = 0.0
    for nodes, N, G, dx, x in est._cell_data(space, rule):
        gh = np.einsum("eqai,ea->eqi", G, U[nodes])
        ge = grad(x.reshape(-1, 2)).reshape(gh.shape)
        total += np.sum(dx * (2 * np.pi * x[..., 0] if axisym else 1.0) * np.sum((ge - gh) ** 2, -1))
    return np.sqrt(total)


MESHES = {"tri": lambda n, p: fm.unit_square(n, p), "quad": lambda n, p: fm.unit_square(n, p, quads=True),
          "curved": lambda n, p: fm.quarter_annulus(1.0 / n, p)}


@pytest.mark.parametrize("mesh, p", [("tri", 1), ("tri", 2), ("tri", 3), ("quad", 1), ("quad", 2),
                                     ("curved", 1), ("curved", 2), ("curved", 3)])
def test_recovery_effectivity(mesh, p, verify):
    eff, ratio = [], []
    for n in (8, 16):
        m = MESHES[mesh](n, p)
        sp = make_space(m)
        U = solve_poisson(sp, f2, u2, m["boundary_tags"])
        e = energy_error(sp, U, du2)
        eff.append(np.sqrt(est.recovery_indicator(sp, U).sum()) / e)
        ratio.append(np.sqrt(est.residual_indicator(sp, U, f2).sum()) / e)
    verify("Error estimation", "Recovery estimator effectivity (estimate / true energy error)",
           f"{mesh} P{p}" if mesh != "quad" else f"quad Q{p}", eff[-1], bound=(0.8, 1.5),
           reference="manufactured Poisson solution", note="n = 16 cells per side (curved: quarter annulus).")
    assert 0.8 <= eff[-1] <= 1.5, eff
    assert abs(eff[-1] - 1) <= abs(eff[0] - 1) + 0.02, eff                  # towards 1 as h → 0
    if mesh != "curved" or p < 3:                                          # residual: the error's rate
        assert ratio[1] == pytest.approx(ratio[0], rel=0.1), ratio


@pytest.mark.parametrize("p", [1, 2, 3])
def test_recovery_effectivity_axisymmetric(p):
    m = fm.unit_square(16, p)
    sp = make_space(m)
    bt = m["boundary_tags"]
    off_axis = bt[sp.node_coords[np.searchsorted(sp.tags, bt)][:, 0] > 1e-12]
    U = solve_poisson(sp, f_axi, u_axi, off_axis, axisym=True)
    e = energy_error(sp, U, du_axi, True)
    assert 0.8 <= np.sqrt(est.recovery_indicator(sp, U, axisymmetric=True).sum()) / e <= 1.5


def test_elastic_indicator():
    """Axisymmetric displacement: a quadratic field in P2 is recovered exactly (indicator 0); the interpolant
    of a smooth field gets an energy-error estimate within the gate."""
    D = elasticity(2e11, 0.3)

    def energy(sp, U, eps_exact):
        rule = quadrature(sp.element.cell.name, 2 * sp.element.degree + 4)
        U = U.reshape(sp.n_nodes, 2)
        tot = 0.0
        for nodes, N, G, dx, x in est._cell_data(sp, rule):
            grad = np.einsum("eqai,eac->eqci", G, U[nodes])
            ur = np.einsum("qa,ea->eq", N, U[nodes][..., 0])
            de = eps_exact(x.reshape(-1, 2)).reshape(x.shape[:2] + (4,)) - est._strain(grad, ur, x[..., 0], grad[..., 0, 0])
            tot += np.sum(dx * 2 * np.pi * x[..., 0] * np.einsum("eqk,kl,eql->eq", de, D, de))
        return np.sqrt(tot)

    m = fm.unit_square(4, 2)
    sp = make_space(m, components=2)
    r, z = sp.node_coords.T
    quad = np.column_stack([r ** 2 + r * z, z ** 2 - r]).ravel()
    assert est.elastic_indicator(sp, quad, D).sum() < 1e-20 * np.abs(D).max()

    def u(x):
        r, z = x[:, 0], x[:, 1]
        return np.column_stack([1e-3 * r * np.cos(z), 1e-3 * np.sin(r + z)])

    def eps(x):
        r, z = x[:, 0], x[:, 1]
        return 1e-3 * np.column_stack([np.cos(z), np.cos(r + z), -r * np.sin(z) + np.cos(r + z), np.cos(z)])
    for p in (1, 2):
        m = fm.unit_square(16, p)
        sp = make_space(m, components=2)
        U = u(sp.node_coords).ravel()
        eff = np.sqrt(est.elastic_indicator(sp, U, D).sum()) / energy(sp, U, eps)
        assert 0.8 <= eff <= 1.5, (p, eff)


def test_recovery_one_cell_across_a_thin_layer():
    """A P2 strip one cell thick (3 levels of nodes across): the cubic fit cannot be determined; the recovery
    falls back to the quadratic fit and is exact on a quadratic field (the minimum-norm cubic fit was not)."""
    import gmsh
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        g = gmsh.model.geo
        p = [g.addPoint(x, y, 0) for x, y in [(0, 0), (1, 0), (1, 0.02), (0, 0.02)]]
        lines = [g.addLine(p[i], p[(i + 1) % 4]) for i in range(4)]
        s = g.addPlaneSurface([g.addCurveLoop(lines)])
        for c, n in zip(lines, (41, 2, 41, 2)):
            g.mesh.setTransfiniteCurve(c, n)
        g.mesh.setTransfiniteSurface(s)
        g.synchronize()
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(2)
        tags, xyz, _ = gmsh.model.mesh.getNodes(2, -1, includeBoundary=True)
        _, nodes = gmsh.model.mesh.getElementsByType(9)
    finally:
        gmsh.finalize()
    m = dict(node_tags=np.asarray(tags), coords=np.reshape(xyz, (-1, 3))[:, :2],
             cells=np.reshape(nodes, (-1, 6)).astype(np.int64), gmsh_type=9)
    sp = make_space(m)
    x, y = sp.node_coords.T
    G = est.recover_gradient(sp, x ** 2 + 3 * x * y - 40 * y ** 2)
    assert np.allclose(G, np.column_stack([2 * x + 3 * y, 3 * x - 80 * y]), atol=1e-9)
