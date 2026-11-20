"""
Phase D2: function spaces, generic kernels and assembly. Patch tests, manufactured solutions
(convergence orders ≈ p + 1 in L2 for P1–P3, Q1–Q2, axisymmetric, 3D), curved isoparametric cells.
"""
import numpy as np
import pytest
import scipy.sparse.linalg as spla

from loudpy.fem import kernels as fk
from loudpy.fem.quadrature import quadrature
from loudpy.fem.reference import element_for_gmsh_type, lagrange, tri10_interior_nodes
from loudpy.fem.space import Space
from support import fem_meshes

pytest.importorskip("gmsh")


def make_space(m, components=1):
    return Space(m["node_tags"], m["coords"], m["cells"], element_for_gmsh_type(m["gmsh_type"]), components)


def solve_poisson(space, f, exact, dirichlet_tags, k2=0.0, axisym=False, degree=None):
    """−Δu − k² u = f with u = exact on the Dirichlet nodes; returns nodal U."""
    elem = space.element
    rule = quadrature(elem.cell.name, degree or 2 * elem.degree + 2)
    asm = fk.Assembler(space)
    A = asm.matrix(fk.laplace(space, rule, axisymmetric=axisym))
    if k2:
        A = A - k2 * asm.matrix(fk.mass(space, rule, axisymmetric=axisym))
    b = asm.vector(fk.load(space, rule, f, axisymmetric=axisym))
    U = np.zeros(space.n_dofs)
    bd = space.dofs_of(dirichlet_tags)
    U[bd] = exact(space.node_coords[bd])
    free = np.setdiff1d(np.arange(space.n_dofs), bd)
    A = A.tocsr()
    U[free] = spla.spsolve(A[free][:, free].tocsc(), b[free] - A[free][:, bd] @ U[bd])
    return U


def l2_error(space, U, exact, axisym=False):
    elem = space.element
    rule = quadrature(elem.cell.name, 2 * elem.degree + 4)
    N, dN = elem.tabulate(rule.points)
    err = 0.0
    for ec, nodes in zip(space.cell_coords, space.cell_nodes):
        x = fk.quadrature_points(ec, N)
        uh = np.einsum("qa,ea->eq", N, U[nodes])
        J = np.einsum("eai,qaj->eqij", ec, dN)
        dx = np.abs(np.linalg.det(J)) * rule.weights
        if axisym:
            dx = dx * 2 * np.pi * x[..., 0]
        err += np.sum(dx * (uh - exact(x.reshape(-1, x.shape[-1])).reshape(uh.shape)) ** 2)
    return np.sqrt(err)


def order(errors, hs):
    return np.log(errors[-2] / errors[-1]) / np.log(hs[-2] / hs[-1])


# --------------------------------------------------------------------------------------- basic checks

def test_space_numbering():
    """DOFs: sorted node tags, components interleaved; cells map back to their node coordinates."""
    m = fem_meshes.unit_square(2, 2)
    sp1, sp2 = make_space(m), make_space(m, components=2)
    assert sp1.n_dofs == len(np.unique(m["cells"])) and sp2.n_dofs == 2 * sp1.n_dofs
    np.testing.assert_array_equal(sp1.tags, np.sort(np.unique(m["cells"])))
    np.testing.assert_array_equal(sp2.cell_dofs[0][:, 0::2] // 2, sp1.cell_dofs[0])
    np.testing.assert_array_equal(sp2.cell_dofs[0][:, 1::2], sp2.cell_dofs[0][:, 0::2] + 1)
    pos = {int(t): i for i, t in enumerate(m["node_tags"])}
    rows = np.vectorize(pos.get)(m["cells"])
    np.testing.assert_array_equal(sp1.cell_coords[0], m["coords"][rows])
    np.testing.assert_array_equal(sp2.dofs_of(m["cells"][0, :2], component=1), sp2.cell_dofs[0][0, [1, 3]])


def test_p1_reference_triangle():
    """The textbook P1 matrices on the unit right triangle."""
    elem = lagrange("triangle", 1)
    sp = Space([1, 2, 3], elem.nodes, np.array([[1, 2, 3]]), elem)
    rule = quadrature("triangle", 2)
    K = fk.laplace(sp, rule).reshape(3, 3)
    M = fk.mass(sp, rule).reshape(3, 3)
    np.testing.assert_allclose(K, [[1, -0.5, -0.5], [-0.5, 0.5, 0], [-0.5, 0, 0.5]], atol=1e-15)
    np.testing.assert_allclose(M, (np.ones((3, 3)) + np.eye(3)) / 24, atol=1e-16)


@pytest.mark.parametrize("quads,p", [(False, 1), (False, 2), (False, 3), (True, 1), (True, 2)])
def test_measures(quads, p):
    """Σ M = area; axisymmetric Σ M = 2π ∫ r dA (= π for the unit square); Laplace rows sum to 0."""
    sp = make_space(fem_meshes.unit_square(3, p, quads))
    rule = quadrature(sp.element.cell.name, 2 * sp.element.degree + 1)
    asm = fk.Assembler(sp)
    np.testing.assert_allclose(asm.matrix(fk.mass(sp, rule)).sum(), 1.0, rtol=1e-13)
    np.testing.assert_allclose(asm.matrix(fk.mass(sp, rule, axisymmetric=True)).sum(), np.pi, rtol=1e-13)
    np.testing.assert_allclose(asm.matrix(fk.laplace(sp, rule)) @ np.ones(sp.n_dofs), 0, atol=1e-12)
    np.testing.assert_allclose(asm.vector(fk.load(sp, rule, lambda x: x[:, 0])).sum(), 0.5, rtol=1e-13)
    c = asm.matrix(fk.mass(sp, rule, coef=1 + 2j))
    np.testing.assert_allclose(c.sum(), 1 + 2j, rtol=1e-13)


def test_boundary_kernels():
    """Perimeter of the unit square and of a curved quarter annulus (P2 arcs) by boundary_mass."""
    m = fem_meshes.unit_square(4, 2)
    elem = element_for_gmsh_type(m["boundary_type"])
    sp = Space(m["node_tags"], m["coords"], m["boundary_cells"], elem)
    rule = quadrature("line", 4)
    asm = fk.Assembler(sp)
    np.testing.assert_allclose(asm.matrix(fk.boundary_mass(sp, rule)).sum(), 4.0, rtol=1e-13)
    np.testing.assert_allclose(asm.vector(fk.boundary_load(sp, rule, 1.0)).sum(), 4.0, rtol=1e-13)
    # axisymmetric: lateral area of the solid [0,1]×[0,1] revolved: 2π·1 (r=1) + 2·π (discs) = 4π
    np.testing.assert_allclose(asm.matrix(fk.boundary_mass(sp, rule, axisymmetric=True)).sum(), 4 * np.pi,
                               rtol=1e-13)
    a = fem_meshes.quarter_annulus(0.3, 2)
    sp = Space(a["node_tags"], a["coords"], a["boundary_cells"], element_for_gmsh_type(a["boundary_type"]))
    L = fk.Assembler(sp).vector(fk.boundary_load(sp, rule, 1.0)).sum()
    np.testing.assert_allclose(L, 2 + 1.5 * np.pi, rtol=1e-4)
    with pytest.raises(ValueError, match="boundary kernel"):
        fk.boundary_mass(make_space(m), rule)
    with pytest.raises(ValueError, match="volume kernel"):
        fk.mass(sp, rule)


# ---------------------------------------------------------------------------------------- patch tests

def distorted(m, amount=0.15, seed=1):
    """Move every interior node at random by up to `amount` × node spacing (cells become curved
    isoparametric cells); checks that no cell is inverted (det J > 0 at the quadrature points)."""
    m = dict(m)
    rng = np.random.default_rng(seed)
    elem = element_for_gmsh_type(m["gmsh_type"])
    inner = ~np.isin(m["node_tags"], m["boundary_tags"])
    h = len(m["cells"]) ** (-1.0 / elem.dim) / elem.order
    m["coords"] = m["coords"] + inner[:, None] * rng.uniform(-amount, amount, m["coords"].shape) * h
    sp = make_space(m)
    _, dN = elem.tabulate(quadrature(elem.cell.name, 2 * elem.degree + 2).points)
    det = np.linalg.det(np.einsum("eai,qaj->eqij", sp.cell_coords[0], dN))
    assert det.min() > 0, "distortion inverted a cell"
    return m


@pytest.mark.parametrize("quads,p", [(False, 1), (False, 2), (False, 3), (True, 1), (True, 2)])
def test_patch_2d(quads, p):
    """A linear field is reproduced exactly on distorted (curved) cells; also axisymmetric u = 1 + 2z."""
    m = distorted(fem_meshes.unit_square(4, p, quads))
    sp = make_space(m)
    exact = lambda x: 1 + 2 * x[:, 0] - 3 * x[:, 1]                                    # noqa: E731
    U = solve_poisson(sp, 0.0, exact, m["boundary_tags"])
    np.testing.assert_allclose(U, exact(sp.node_coords), atol=1e-11)
    exact = lambda x: 1 + 2 * x[:, 1]                                                  # noqa: E731
    U = solve_poisson(sp, 0.0, exact, m["boundary_tags"], axisym=True)
    np.testing.assert_allclose(U, exact(sp.node_coords), atol=1e-11)


@pytest.mark.parametrize("p", [1, 2, 3])
def test_patch_3d(p):
    m = distorted(fem_meshes.unit_cube(0.5, p), amount=0.1)
    sp = make_space(m)
    exact = lambda x: 1 + x[:, 0] - 2 * x[:, 1] + 3 * x[:, 2]                         # noqa: E731
    U = solve_poisson(sp, 0.0, exact, m["boundary_tags"])
    np.testing.assert_allclose(U, exact(sp.node_coords), atol=1e-10)


# ------------------------------------------------------------------------------ manufactured solutions

def u2(x):
    return np.sin(np.pi * x[:, 0]) * np.exp(x[:, 1])


def f2(x, k2=0.0):
    return (np.pi ** 2 - 1 - k2) * u2(x)


@pytest.mark.parametrize("quads,p", [(False, 1), (False, 2), (False, 3), (True, 1), (True, 2)])
def test_convergence_2d(quads, p):
    """Poisson and Helmholtz (k² = 4): L2 error order p + 1 on structured meshes."""
    hs, e_poisson, e_helm = [], [], []
    for n in (4, 8, 16):
        m = fem_meshes.unit_square(n, p, quads)
        sp = make_space(m)
        hs.append(1.0 / n)
        e_poisson.append(l2_error(sp, solve_poisson(sp, f2, u2, m["boundary_tags"]), u2))
        U = solve_poisson(sp, lambda x: f2(x, 4.0), u2, m["boundary_tags"], k2=4.0)
        e_helm.append(l2_error(sp, U, u2))
    for e in (e_poisson, e_helm):
        assert order(e, hs) > p + 1 - 0.15, (p, quads, e)


def u_axi(x):
    return np.cos(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])


def f_axi(x):
    r, z = x[:, 0], x[:, 1]
    return np.pi * np.sin(np.pi * z) * (np.sin(np.pi * r) / r + 2 * np.pi * np.cos(np.pi * r))


@pytest.mark.parametrize("quads,p", [(False, 1), (False, 2), (False, 3), (True, 1), (True, 2)])
def test_convergence_axisymmetric(quads, p):
    """−(1/r)(r u_r)_r − u_zz = f on [0,1]²: natural condition on the axis, order p + 1."""
    hs, errs = [], []
    for n in (4, 8, 16):
        m = fem_meshes.unit_square(n, p, quads)
        sp = make_space(m)
        x = sp.node_coords
        bd_rows = np.isin(sp.tags, m["boundary_tags"]) & ~((x[:, 0] < 1e-12) & (x[:, 1] > 1e-12)
                                                           & (x[:, 1] < 1 - 1e-12))
        U = solve_poisson(sp, f_axi, u_axi, sp.tags[bd_rows], axisym=True)
        hs.append(1.0 / n)
        errs.append(l2_error(sp, U, u_axi, axisym=True))
    assert order(errs, hs) > p + 1 - 0.15, (p, quads, errs)


def u3(x):
    return np.exp(x[:, 0] + 0.5 * x[:, 1] - 0.5 * x[:, 2])


@pytest.mark.parametrize("p", [1, 2, 3])
def test_convergence_3d(p):
    """Tetrahedra P1–P3: −Δu = −1.5 u on unstructured meshes; h_eff = n_cells^(−1/3) (orders within 0.3)."""
    errs, hs = [], []
    for h in (0.25, 0.18):
        m = fem_meshes.unit_cube(h, p)
        sp = make_space(m)
        hs.append(len(m["cells"]) ** (-1 / 3))
        errs.append(l2_error(sp, solve_poisson(sp, lambda x: -1.5 * u3(x), u3, m["boundary_tags"]), u3))
    assert order(errs, hs) > p + 1 - 0.3, (p, errs)


# --------------------------------------------------------------------------------------- curved cells

@pytest.mark.parametrize("p", [2, 3])
def test_curved_cells(p):
    """Quarter annulus 1 ≤ r ≤ 2: ∫ r² dA = 15π/8 converges only as h² with straight-sided cells and
    is 10³–10⁸ times more accurate with curved isoparametric cells; u = ln r (harmonic) converges at
    order p + 1 on the curved cells. (The plain area is a bad probe: the chord errors of the two arcs
    cancel when both have the same segment length.)"""
    r2 = lambda x: x[:, 0] ** 2 + x[:, 1] ** 2                                           # noqa: E731
    fixed = []
    ln_r = lambda x: 0.5 * np.log(r2(x))                                                 # noqa: E731
    hs, curved, straight, errs = [], [], [], []
    for h in (0.5, 0.25, 0.125):
        for flag, out in ((True, curved), (False, straight)):
            m = fem_meshes.quarter_annulus(h, p, curved=flag)
            sp = make_space(m)
            rule = quadrature("triangle", 2 * p + 4)
            out.append(abs(fk.Assembler(sp).vector(fk.load(sp, rule, r2)).sum() - 15 * np.pi / 8))
            if flag:
                errs.append(l2_error(sp, solve_poisson(sp, 0.0, ln_r, m["boundary_tags"]), ln_r))
                if p == 3:                                   # gmsh's TRI10 interior node re-placed
                    m = dict(m, coords=m["coords"].copy())
                    rows = np.searchsorted(m["node_tags"], m["cells"][:, 9])
                    assert np.array_equal(m["node_tags"][rows], m["cells"][:, 9])
                    m["coords"][rows] = tri10_interior_nodes(sp.cell_coords[0])
                    sp = make_space(m)
                    fixed.append(l2_error(sp, solve_poisson(sp, 0.0, ln_r, m["boundary_tags"]), ln_r))
        hs.append(len(m["cells"]) ** -0.5)
    assert 1.7 < order(straight, hs) < 2.5
    assert all(c < 1e-3 * s for c, s in zip(curved, straight))
    assert np.log(curved[0] / curved[-1]) / np.log(hs[0] / hs[-1]) > p + 1
    if p == 3:
        assert order(errs, hs) > 3.2, errs                   # gmsh's interior nodes: ≈ 3.5
        errs = fixed
    assert order(errs, hs) > p + 1 - 0.2, errs


@pytest.mark.parametrize("dim,p", [(2, 1), (2, 2), (2, 3), (3, 1), (3, 2)])
def test_boundary_normals(dim, p):
    """Divergence theorem with the normal kernel: ∮ x_i n_i ds = |Ω| for each i, ∮ n ds = 0
    (unit square boundary lines, unit cube boundary triangles)."""
    m = fem_meshes.unit_square(3, p) if dim == 2 else fem_meshes.unit_cube(0.5, p)
    elem = element_for_gmsh_type(m["boundary_type"])
    sp = Space(m["node_tags"], m["coords"], m["boundary_cells"], elem)
    rule = quadrature(elem.cell.name, 2 * elem.degree + 2)
    blocks = fk.boundary_normal_mass(sp, rule)                       # (n_e, dim, n²)
    asm = fk.Assembler(sp)
    one = np.ones(sp.n_dofs)
    for i in range(dim):
        Mi = asm.matrix(np.ascontiguousarray(blocks[:, i]))
        np.testing.assert_allclose(one @ Mi @ one, 0.0, atol=1e-13)
        np.testing.assert_allclose(abs(one @ Mi @ sp.node_coords[:, i]), 1.0, rtol=1e-12)


def test_tet10_toolkit_curved():
    """3D toolkit (fem.tet10): straight-sided cells see a polyhedron (volume, area errors ~ h²);
    isoparametric cells follow gmsh's mid nodes on the sphere (errors ~ h⁴); Laplace rows sum to 0."""
    from loudpy.fem import tet10
    X, tets, tris = fem_meshes.ball(0.4)
    K, M, V_straight = tet10.assemble_p2(X, tets, curved=False)
    _, Mc, V_curved = tet10.assemble_p2(X, tets, curved=True)
    a_straight = tet10.surface_quadrature(X, tris, curved=False)[1].sum()
    a_curved = tet10.surface_quadrature(X, tris, curved=True)[1].sum()
    V, A = 4 * np.pi / 3, 4 * np.pi
    # the gmsh mesh differs between platforms (Linux CI: volume −6.1 % straight, −0.051 % curved; macOS:
    # −3.5 % and −0.015 %): compare the two geometries rather than fixed numbers
    for straight, curved, exact in ((V_straight, V_curved, V), (a_straight, a_curved, A)):
        assert abs(curved - exact) < abs(straight - exact) / 50 and abs(curved - exact) / exact < 2e-3
    np.testing.assert_allclose(M.sum(), V_straight, rtol=1e-13)
    np.testing.assert_allclose(Mc.sum(), V_curved, rtol=1e-13)
    np.testing.assert_allclose(K @ np.ones(len(X)), 0.0, atol=1e-12)
