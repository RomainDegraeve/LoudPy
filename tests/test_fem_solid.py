"""
Phase D3: generic axisymmetric elasticity (any element order), Fourier orders, total-Lagrangian
element and PML coefficients. Checks by physics, not by comparison with the former kernels
(those are compared in a git worktree, see docs/ROADMAP.md, phase D3).
"""
import numpy as np
import pytest

from loudpy.fem import kernels as fk
from loudpy.fem.quadrature import quadrature
from loudpy.fem.reference import element_for_gmsh_type
from loudpy.fem.solid import (axisymmetric_kernel, fourier_kernel, material_matrix_D, material_matrix_D6,
                              total_lagrangian_kernel)
from loudpy.fem.space import Space
from support import fem_meshes

pytest.importorskip("gmsh")
ELEMENTS = [(False, 1), (False, 2), (False, 3), (True, 1), (True, 2)]
E, NU, RHO = 2.0e9, 0.3, 1200.0


def annular_block(quads, p, n=3):
    """[0.5, 1.5] × [0, 1] in (r, z): away from the axis, so u_r / r is smooth."""
    m = dict(fem_meshes.unit_square(n, p, quads))
    m["coords"] = m["coords"] + np.array([0.5, 0.0])
    sp = Space(m["node_tags"], m["coords"], m["cells"], element_for_gmsh_type(m["gmsh_type"]), 2)
    return m, sp


def stiffness(sp, D, kernel=axisymmetric_kernel, extra=()):
    elem = sp.element
    N, dN, w = fk.tabulate(elem, quadrature(elem.cell.name, 2 * elem.degree + 2))
    c = 3 if kernel is fourier_kernel else 2
    out = np.empty((sp.n_cells, (c * elem.n_nodes) ** 2))
    kernel(sp.cell_coords[0], N, dN, w, D, *extra, out)
    return out


@pytest.mark.parametrize("quads,p", ELEMENTS)
def test_axisymmetric_patch(quads, p):
    """Rigid axial translation has no energy; the uniform-strain field u_r = a r, u_z = b z has
    K u = 0 at interior nodes and strain energy ½ uᵀKu = ½ εᵀDε · volume."""
    m, sp = annular_block(quads, p)
    K = fk.Assembler(sp).matrix(stiffness(sp, material_matrix_D(E, NU)))
    x = sp.node_coords
    uz = np.zeros(sp.n_dofs)
    uz[1::2] = 1.0
    np.testing.assert_allclose(K @ uz, 0.0, atol=1e-6 * abs(K).max())
    a, b = 1e-3, -2e-3
    u = np.zeros(sp.n_dofs)
    u[0::2], u[1::2] = a * x[:, 0], b * x[:, 1]
    f = K @ u
    inner = sp.dofs_of(np.setdiff1d(sp.tags, m["boundary_tags"]))
    np.testing.assert_allclose(f[inner], 0.0, atol=1e-9 * abs(f).max())
    eps = np.array([a, b, 0.0, a])                                     # ε_r, ε_z, γ_rz, ε_θ
    volume = np.pi * (1.5 ** 2 - 0.5 ** 2)
    np.testing.assert_allclose(0.5 * u @ f, 0.5 * eps @ material_matrix_D(E, NU) @ eps * volume, rtol=1e-11)


@pytest.mark.parametrize("quads,p", ELEMENTS)
def test_fourier_rigid_modes(quads, p):
    """Order 1: lateral translation (U_r = 1, U_θ = −1) and tilt about a horizontal axis
    (U_r = z, U_z = −r, U_θ = −z) have no strain energy; order 0: rotation about the axis (U_θ = r).
    Order 0 without U_θ equals the axisymmetric stiffness."""
    m, sp2 = annular_block(quads, p)
    sp = Space(m["node_tags"], m["coords"], m["cells"], sp2.element, 3)
    D6 = material_matrix_D6(E, NU)
    x = sp.node_coords
    K1 = fk.Assembler(sp).matrix(stiffness(sp, D6, fourier_kernel, (1,)))
    scale = abs(K1).max()
    for mode in (np.column_stack([np.ones(len(x)), np.zeros(len(x)), -np.ones(len(x))]),
                 np.column_stack([x[:, 1], -x[:, 0], -x[:, 1]])):
        np.testing.assert_allclose(K1 @ mode.ravel(), 0.0, atol=1e-9 * scale)
    K0 = fk.Assembler(sp).matrix(stiffness(sp, D6, fourier_kernel, (0,)))
    rot = np.column_stack([np.zeros(len(x)), np.zeros(len(x)), x[:, 0]]).ravel()
    np.testing.assert_allclose(K0 @ rot, 0.0, atol=1e-9 * abs(K0).max())
    keep = np.sort(np.concatenate([np.arange(0, sp.n_dofs, 3), np.arange(1, sp.n_dofs, 3)]))
    K2 = fk.Assembler(sp2).matrix(stiffness(sp2, material_matrix_D(E, NU)))
    np.testing.assert_allclose(K0.toarray()[np.ix_(keep, keep)], K2.toarray(), atol=1e-12 * abs(K2).max())


@pytest.mark.parametrize("quads,p", [(False, 2), (True, 2), (False, 3)])
def test_total_lagrangian(quads, p):
    """At u = 0 the TL element is the linear one (K_T = K, F_int = 0, consistent mass, C = αM + βK);
    at u ≠ 0, K_T is the derivative of F_int (central differences)."""
    _, sp = annular_block(quads, p, n=2)
    elem = sp.element
    N, dN, w = fk.tabulate(elem, quadrature(elem.cell.name, 2 * elem.degree + 2))
    ec = sp.cell_coords[0]
    D = material_matrix_D(E, NU)
    n_e, m = len(ec), 2 * elem.n_nodes
    alpha, beta = 3.0, 2e-5

    def tl(u):
        outs = [np.empty((n_e, m * m)) for _ in range(3)] + [np.empty((n_e, m))]
        ue = np.ascontiguousarray(u[sp.cell_dofs[0]])
        total_lagrangian_kernel(ec, ue, N, dN, w, D, RHO, alpha, beta, *outs)
        return outs

    KT, M, C, F = tl(np.zeros(sp.n_dofs))
    K = stiffness(sp, D)
    Ms = np.empty((n_e, elem.n_nodes ** 2))
    fk._mass_kernel(ec, N, dN, w, fk.coefficient(RHO, ec, N), True, Ms)
    np.testing.assert_allclose(KT, K, atol=1e-12 * abs(K).max())
    np.testing.assert_allclose(M, fk.expand_components(Ms, 2), atol=1e-14 * abs(M).max())
    np.testing.assert_allclose(C, alpha * M + beta * K, atol=1e-12 * abs(C).max())
    np.testing.assert_allclose(F, 0.0, atol=0)

    rng = np.random.default_rng(3)
    u = 1e-2 * rng.standard_normal(sp.n_dofs)
    du = rng.standard_normal(sp.n_dofs)
    asm = fk.Assembler(sp)
    h = 1e-6
    fp, fm = (asm.vector(tl(u + s * h * du)[3]) for s in (1, -1))
    KTu = asm.matrix(tl(u)[0])
    np.testing.assert_allclose((fp - fm) / (2 * h), KTu @ du, rtol=1e-6, atol=1e-6 * abs(KTu @ du).max())


def test_pml_coefficients():
    """Outside the layer nothing is stretched; inside Im s < 0, Im r̃ < 0 growing with depth, as
    s = 1 − iαλ (n/t) ξ^(n−1), r̃ = r − iαλ ξⁿ (sign +)."""
    from types import SimpleNamespace
    from loudpy.physics.acoustic import pml_coefficients
    layer = SimpleNamespace(use_r=True, r0=1.0, tr=0.5, sgn_r=1.0, use_z=False, z0=0.0, tz=1.0, sgn_z=1.0)
    x = np.array([[0.5, 0.2], [1.0, 0.0], [1.25, 0.1], [1.5, 0.3], [1.7, 0.3]])
    c, omega, alpha, n = 343.0, 2 * np.pi * 1000.0, 3.0, 2
    sr, sz, rt = pml_coefficients(layer, x, alpha, n, omega, c)
    lam = 2 * np.pi * c / omega
    xi = np.clip((x[:, 0] - 1.0) / 0.5, 0, 1)
    np.testing.assert_allclose(sr, np.where(xi > 0, 1 - 1j * alpha * lam * (n / 0.5) * xi, 1))
    np.testing.assert_allclose(rt, x[:, 0] - 1j * alpha * lam * xi ** 2)
    np.testing.assert_array_equal(sz, 1.0)
    assert np.all(np.diff(rt.imag[1:4]) < 0)


def test_plate_at_every_order(tmp_path):
    """End to end through lp.Model(mesh_order=p): P2 and P3 meshes of the clamped plate give the first
    two axisymmetric bending modes within 0.1 % of each other, P3 below Kirchhoff by < 0.5 % (3D
    elasticity); linear triangles lock in bending (> 10 % too stiff)."""
    import loudpy as lp
    from support import current_api as api
    E_, NU_, RHO_, A_, H_ = 2.1e11, 0.3, 7850.0, 0.1, 0.002
    scale = np.sqrt(E_ * H_ ** 3 / (12 * (1 - NU_ ** 2)) / (RHO_ * H_)) / (2 * np.pi * A_ ** 2)
    kirchhoff = np.array([10.2158, 39.771]) * scale
    steel = lp.Material("Steel", rho=RHO_, E=E_, nu=NU_)
    f = {}
    for p in (1, 2, 3):
        model = lp.Model(api.PLATE_GEO, mesh_order=p).add(
            lp.Solid("plate", steel, damping=lp.Undamped(), mesh_size=H_),
            lp.Clamped("interface_constrained_rz"), lp.Clamped("interface_constrained_r", dofs="r"))
        model.mesh(tmp_path / f"plate_p{p}.msh", deterministic=True)
        cell = {g.cell_type for g in model.meshed.groups.values() if g.dim == 2}
        assert cell == {{1: "tri3", 2: "tri6", 3: "tri10"}[p]}
        f[p] = np.sort(lp.Modes(model).solve(f_min=10.0, f_max=2500.0, orders=[0],
                                             n_per_shift=10).order(0).frequencies)[:2]
    np.testing.assert_allclose(f[2], f[3], rtol=1e-3)
    assert np.all((f[3] < kirchhoff) & (f[3] > 0.995 * kirchhoff)), (f[3], kirchhoff)
    assert np.all(f[1] > 1.1 * kirchhoff), f[1]


@pytest.mark.parametrize("n_nodes", [3, 6, 10, 4, 9])
def test_mesh_cells_of_every_order(n_nodes):
    """Edges of 2D cells (selectors) and the linear sub-triangles (plots, interpolation) of every order."""
    from loudpy.fem.reference import element_for_cells
    from loudpy.geometry.mesh import Mesh, MeshGroup
    from loudpy.geometry.selectors import _tri_edges
    elem = element_for_cells(2, n_nodes)
    cells = np.arange(1, n_nodes + 1)[None]
    rows, owner = _tri_edges(cells)
    assert rows.shape == (len(elem.cell.edges), elem.order + 1) and np.all(owner == 0)
    for row, (a, b) in zip(rows, elem.cell.edges):
        assert tuple(row[:2] - 1) == (a, b)
    mesh = Mesh(np.arange(1, n_nodes + 1), elem.nodes.copy(),
                {"g": MeshGroup("g", 2, "x", cells)})
    for refine in (False, True):
        tri = mesh.triangles(refine=refine)
        P = mesh.coords[tri]
        area = 0.5 * np.abs(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])).sum()
        np.testing.assert_allclose(area, elem.cell.measure, rtol=1e-12)
    assert len(np.unique(mesh.triangles(refine=True))) == n_nodes
