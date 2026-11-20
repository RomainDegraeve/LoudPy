"""
Phase D1: reference cells, Lagrange bases on gmsh's node layouts, quadrature exact to a degree.
"""
from itertools import product
from math import factorial

import numpy as np
import pytest

from loudpy.fem.quadrature import legacy_line, legacy_triangle7, quadrature
from loudpy.fem.reference import CELLS, GMSH_TYPES, element_for_gmsh_type, lagrange

ELEMENTS = sorted(GMSH_TYPES.values())
rng = np.random.default_rng(0)


def random_points(cell, n=20):
    """Points inside the reference cell."""
    c = CELLS[cell]
    if c.simplex and c.dim > 1:
        lam = rng.dirichlet(np.ones(c.dim + 1), size=n)
        return lam @ c.vertices
    return rng.uniform(-1, 1, size=(n, c.dim))


def exact_integral(cell, e):
    """∫ x^e over the reference cell."""
    if cell in ("line", "quadrilateral"):
        return np.prod([(1 - (-1) ** (a + 1)) / (a + 1) for a in e])
    return np.prod([factorial(a) for a in e]) / factorial(sum(e) + len(e))


# --------------------------------------------------------------------------------------------- layouts

@pytest.mark.parametrize("gmsh_type", sorted(GMSH_TYPES))
def test_layout_matches_gmsh(gmsh_type):
    gmsh = pytest.importorskip("gmsh")
    gmsh.initialize(readConfigFiles=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        name, dim, order, n, coords, _ = gmsh.model.mesh.getElementProperties(gmsh_type)
    finally:
        gmsh.finalize()
    elem = element_for_gmsh_type(gmsh_type)
    assert (elem.dim, elem.order, elem.n_nodes) == (dim, order, n)
    np.testing.assert_allclose(elem.nodes, np.reshape(coords, (n, dim)), atol=1e-15)


def test_unknown_type():
    with pytest.raises(ValueError, match="not supported"):
        element_for_gmsh_type(5)            # hexahedron
    with pytest.raises(ValueError, match="no Lagrange element"):
        lagrange("quadrilateral", 3)


# ------------------------------------------------------------------------------------------- the bases

@pytest.mark.parametrize("cell,order", ELEMENTS)
def test_nodal_basis(cell, order):
    """N_i(x_j) = δ_ij; Σ N = 1 and Σ ∇N = 0 everywhere."""
    elem = lagrange(cell, order)
    N, _ = elem.tabulate(elem.nodes)
    np.testing.assert_allclose(N, np.eye(elem.n_nodes), atol=1e-12)
    N, dN = elem.tabulate(random_points(cell))
    np.testing.assert_allclose(N.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(dN.sum(axis=1), 0.0, atol=1e-11)


@pytest.mark.parametrize("cell,order", ELEMENTS)
def test_reproduces_its_polynomials(cell, order):
    """Nodal interpolation is exact (values and gradients) for every monomial of the element's space."""
    elem = lagrange(cell, order)
    pts = random_points(cell)
    simplex = CELLS[cell].simplex
    for e in product(range(order + 1), repeat=elem.dim):
        if simplex and sum(e) > order:
            continue
        e = np.array(e)
        f = lambda x: np.prod(x ** e, axis=1)                                   # noqa: E731
        N, dN = elem.tabulate(pts)
        u = elem.interpolate(f)
        np.testing.assert_allclose(N @ u, f(pts), atol=1e-12)
        grad = np.stack([e[d] * pts[:, d] ** max(e[d] - 1, 0)
                         * np.prod(np.delete(pts ** e, d, axis=1), axis=1) for d in range(elem.dim)],
                        axis=1)
        np.testing.assert_allclose(np.einsum("qnd,n->qd", dN, u), grad, atol=1e-11)


def test_closed_form_tri6_line3():
    """The generic TRI6 and LINE3 bases equal their closed forms (the hand-written functions of the former
    kernels, removed in phase D5)."""
    tri, line = lagrange("triangle", 2), lagrange("line", 2)
    for x, y in random_points("triangle", 10):
        N, dN = tri.tabulate([[x, y]])
        N0 = [1 - 3 * x - 3 * y + 2 * x ** 2 + 4 * x * y + 2 * y ** 2, 2 * x ** 2 - x, 2 * y ** 2 - y,
              4 * x - 4 * x ** 2 - 4 * x * y, 4 * x * y, 4 * y - 4 * x * y - 4 * y ** 2]
        dx = [-3 + 4 * x + 4 * y, 4 * x - 1, 0, 4 - 8 * x - 4 * y, 4 * y, -4 * y]
        dy = [-3 + 4 * x + 4 * y, 0, 4 * y - 1, -4 * x, 4 * x, 4 - 4 * x - 8 * y]
        np.testing.assert_allclose(N[0], N0, atol=1e-14)
        np.testing.assert_allclose(dN[0, :, 0], dx, atol=1e-13)
        np.testing.assert_allclose(dN[0, :, 1], dy, atol=1e-13)
    for (x,) in random_points("line", 10):
        N, dN = line.tabulate([[x]])
        np.testing.assert_allclose(N[0], [-0.5 * x * (1 - x), 0.5 * x * (1 + x), 1 - x * x], atol=1e-14)
        np.testing.assert_allclose(dN[0, :, 0], [x - 0.5, x + 0.5, -2 * x], atol=1e-14)


@pytest.mark.parametrize("cell,order", [e for e in ELEMENTS if e[0] != "line"])
def test_facet_nodes(cell, order):
    """On a facet only its nodes' functions live, and they equal the facet element's basis."""
    elem = lagrange(cell, order)
    sub = lagrange(CELLS[cell].facet_cell, order)
    for facet, nodes in zip(elem.cell.facets, elem.facet_nodes):
        X = elem.cell.vertices[list(facet)]
        A = np.column_stack([sub.cell.vertices, np.ones(len(sub.cell.vertices))])
        M, *_ = np.linalg.lstsq(A[: sub.dim + 1], X[: sub.dim + 1], rcond=None)
        s = random_points(sub.cell.name, 8)
        x = np.column_stack([s, np.ones(len(s))]) @ M
        N, _ = elem.tabulate(x)
        Ns, _ = sub.tabulate(s)
        np.testing.assert_allclose(N[:, nodes], Ns, atol=1e-12)
        others = np.setdiff1d(np.arange(elem.n_nodes), nodes)
        np.testing.assert_allclose(N[:, others], 0.0, atol=1e-12)


# ------------------------------------------------------------------------------------------ quadrature

@pytest.mark.parametrize("cell", sorted(CELLS))
@pytest.mark.parametrize("degree", range(0, 9))
def test_quadrature_exact(cell, degree):
    rule = quadrature(cell, degree)
    c = CELLS[cell]
    assert np.all(rule.weights > 0)
    np.testing.assert_allclose(rule.weights.sum(), c.measure, rtol=1e-14)
    lam = rule.points @ np.linalg.pinv(c.vertices[1:] - c.vertices[0]) if c.simplex else None
    if c.simplex and c.dim > 1:                                 # points strictly inside
        b = np.column_stack([1 - lam.sum(1), lam])
        assert np.all(b > 0)
    for e in product(range(degree + 1), repeat=c.dim):
        if sum(e) > degree:
            continue
        val = rule.weights @ np.prod(rule.points ** np.array(e), axis=1)
        np.testing.assert_allclose(val, exact_integral(cell, e), rtol=1e-13, atol=1e-15)


def test_quadrature_not_exact_above_its_degree():
    rule = quadrature("triangle", 3)
    e = (4, 0)
    val = rule.weights @ rule.points[:, 0] ** 4
    assert abs(val - exact_integral("triangle", e)) > 1e-6


def test_legacy_rules():
    """The former rules: 7-point triangle exact to degree 5 at its 8 printed digits, Gauss lines exact."""
    tri = legacy_triangle7()
    np.testing.assert_allclose(tri.weights.sum(), 0.5, rtol=1e-8)
    for e in product(range(6), repeat=2):
        if sum(e) <= 5:
            val = tri.weights @ np.prod(tri.points ** np.array(e), axis=1)
            np.testing.assert_allclose(val, exact_integral("triangle", e), rtol=2e-7)
    for n in (3, 4):
        rule = legacy_line(n)
        for a in range(2 * n):
            np.testing.assert_allclose(rule.weights @ rule.points[:, 0] ** a,
                                       exact_integral("line", (a,)), atol=1e-14)


@pytest.mark.parametrize("cell,order", ELEMENTS)
def test_mass_of_reference_element(cell, order):
    """∫ N_i N_j with a rule of degree 2·degree is exact: Σ_ij M_ij = measure, M symmetric positive."""
    elem = lagrange(cell, order)
    rule = quadrature(cell, 2 * elem.degree)
    N, _ = elem.tabulate(rule.points)
    M = N.T @ (rule.weights[:, None] * N)
    np.testing.assert_allclose(M.sum(), CELLS[cell].measure, rtol=1e-13)
    np.testing.assert_allclose(M, M.T, atol=1e-15)
    assert np.linalg.eigvalsh(M).min() > 0
