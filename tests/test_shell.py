"""MITC3 / MITC4 Reissner–Mindlin shells (loudpy.fem.shell): rigid motions, patch tests, plate modes against
the exact Mindlin and Kirchhoff frequencies, convergence, no shear locking (MITC4)."""
import numpy as np
import pytest
import scipy.linalg as sla

from loudpy.fem.shell import cell_normals, feature_nodes, fixed_dofs, modes, shell_matrices
from support.fem_meshes import shell_surface

E, NU, RHO = 70e9, 0.3, 2700.0


def grid(n, a=1.0, b=1.0, tri=False, distort=0.0, seed=0):
    x, y = np.meshgrid(np.linspace(0, a, n + 1), np.linspace(0, b, n + 1), indexing="ij")
    inner = (x > 0) & (x < a) & (y > 0) & (y < b)
    rng = np.random.default_rng(seed)
    x = x + distort * a / n * rng.uniform(-1, 1, x.shape) * inner
    y = y + distort * b / n * rng.uniform(-1, 1, y.shape) * inner
    nodes = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]
    k = lambda i, j: i * (n + 1) + j
    q = np.array([[k(i, j), k(i + 1, j), k(i + 1, j + 1), k(i, j + 1)] for i in range(n) for j in range(n)])
    return nodes, (np.vstack([q[:, [0, 1, 2]], q[:, [0, 2, 3]]]) if tri else q)


def mindlin_ss(a, b, h, m, n, k=5 / 6):
    """Exact lowest frequency [Hz] of mode (m, n) of a hard simply supported Mindlin plate."""
    D, s = E * h ** 3 / (12 * (1 - NU ** 2)), k * E / (2 * (1 + NU)) * h
    al, be = m * np.pi / a, n * np.pi / b
    K = np.array([[s * (al ** 2 + be ** 2), s * al, s * be],
                  [s * al, D / 2 * ((1 - NU) * (al ** 2 + be ** 2) + (1 + NU) * al ** 2) + s, D / 2 * (1 + NU) * al * be],
                  [s * be, D / 2 * (1 + NU) * al * be, D / 2 * ((1 - NU) * (al ** 2 + be ** 2) + (1 + NU) * be ** 2) + s]])
    M = np.diag([RHO * h, RHO * h ** 3 / 12, RHO * h ** 3 / 12])
    return np.sqrt(sla.eigh(K, M, eigvals_only=True)[0]) / (2 * np.pi)


def lowest(K, M, fixed, n=6):
    return modes(K, M, n, fixed)[0]


def ss_plate(nodes, cells, h, a=1.0, b=1.0, n=4):
    K, M = shell_matrices(nodes, cells, E, NU, RHO, h)
    x, y = nodes[:, 0], nodes[:, 1]
    ex = np.where(np.isclose(x, 0) | np.isclose(x, a))[0]
    ey = np.where(np.isclose(y, 0) | np.isclose(y, b))[0]
    fixed = np.concatenate([fixed_dofs(np.union1d(ex, ey), "xyz"), fixed_dofs(ex, "a"), fixed_dofs(ey, "b")])
    return lowest(K, M, fixed, n + 2)[:n]


def rigid_fields(nodes):
    """The six rigid motions as nodal DOF vectors (u, θ per node)."""
    out = []
    for i in range(3):
        v = np.zeros((len(nodes), 6))
        v[:, i] = 1.0
        out.append(v.ravel())
    for i in range(3):
        w = np.zeros(3)
        w[i] = 1.0
        v = np.zeros((len(nodes), 6))
        v[:, :3] = np.cross(w, nodes - nodes.mean(axis=0))
        v[:, 3:] = w
        out.append(v.ravel())
    return np.column_stack(out)


@pytest.mark.parametrize("kind,quads", [("box", False), ("cylinder", False), ("cylinder", True), ("disk", True)])
def test_rigid_motions_cost_no_energy(kind, quads):
    nodes, cells = shell_surface(kind, 0.3, quads=quads, R=0.5)
    K, M = shell_matrices(nodes, cells, E, NU, RHO, 0.01)
    U = rigid_fields(nodes)
    scale = abs(K).max() * np.abs(U).max()
    assert np.abs(K @ U).max() < 1e-9 * scale
    if len(nodes) < 120:                                     # exactly six zero eigenvalues, no mechanism
        lam = sla.eigh(K.toarray(), M.toarray(), eigvals_only=True)
        assert np.sum(lam < 1e-8 * lam.max()) == 6


@pytest.mark.parametrize("tri", [False, True])
def test_patch_tests(tri):
    """Uniform membrane strain and uniform curvature (zero shear) on distorted cells: no interior force."""
    nodes, cells = grid(4, tri=tri, distort=0.3)
    K, _ = shell_matrices(nodes, cells, E, NU, RHO, 0.02)
    x, y = nodes[:, 0], nodes[:, 1]
    inner = np.where((x > 1e-9) & (x < 1 - 1e-9) & (y > 1e-9) & (y < 1 - 1e-9))[0]
    u = np.zeros((len(nodes), 6))
    u[:, 0], u[:, 1] = 1e-3 * (x + 0.4 * y), 1e-3 * (-0.7 * x + 0.2 * y)              # membrane
    u[:, 2] = 1e-3 * (0.5 * x ** 2 + 0.3 * x * y - 0.25 * y ** 2)                     # bending, γ = 0
    u[:, 4] = -1e-3 * (x + 0.3 * y)                                                   # θ2 = −∂w/∂x
    u[:, 3] = 1e-3 * (0.3 * x - 0.5 * y)                                              # θ1 = ∂w/∂y
    u[:, 5] = 0.5 * 1e-3 * (-0.7 - 0.4)                                               # drilling = ω
    r = (K @ u.ravel()).reshape(-1, 6)
    scale = abs(K).max() * 1e-3
    assert np.abs(r[inner]).max() < 1e-10 * scale


def test_simply_supported_plate_modes(verify):
    """Gate I: hard simply supported square plate vs the exact Mindlin frequencies (h/a = 0.01 and 0.1):
    MITC4 within 0.5 % (32 × 32), order 2; MITC3 on an unstructured mesh within 0.5 % for the first mode."""
    for h in (0.01, 0.1):
        exact = np.array(sorted(mindlin_ss(1, 1, h, m, n) for m in range(1, 4) for n in range(1, 4))[:4])
        err = [np.abs(ss_plate(*grid(n), h) / exact - 1) for n in (8, 16, 32)]
        case = "Simply supported square plate, MITC4 shells vs exact Mindlin"
        verify("Shells", case, f"max error of the first 4 modes, 32 × 32, h/a = {h:g}", err[-1].max(), bound=0.005,
               reference="exact Mindlin frequencies (3 × 3 eigenproblem per (m, n))")
        order = np.log2(err[1][0] / err[2][0])
        verify("Shells", case, f"observed order (16 → 32), h/a = {h:g}", order, bound=(1.9, 2.1))
        assert err[-1].max() < 0.005, (h, err[-1])
        assert 1.9 < order < 2.1, order
    exact = np.array(sorted(mindlin_ss(1, 1, 0.01, m, n) for m in range(1, 4) for n in range(1, 4))[:4])
    f3 = ss_plate(*shell_surface("rectangle", 1 / 32), 0.01)
    e3 = np.abs(f3 / exact - 1)
    verify("Shells", "Simply supported square plate, MITC3 unstructured (a/32) vs exact Mindlin",
           "error of the first mode, h/a = 0.01", e3[0], bound=0.005, reference="exact Mindlin frequencies")
    verify("Shells", "Simply supported square plate, MITC3 unstructured (a/32) vs exact Mindlin",
           "max error of the first 4 modes", e3.max(), bound=0.01)
    assert e3[0] < 0.005 and e3.max() < 0.01, e3


def test_no_shear_locking_mitc4():
    """MITC4 at h/a = 1e-3 has the errors of h/a = 1e-2 (no locking); MITC3 converges, more slowly."""
    for n in (8, 16):
        e = [ss_plate(*grid(n), h)[0] / mindlin_ss(1, 1, h, 1, 1) - 1 for h in (1e-2, 1e-3)]
        assert abs(e[1] - e[0]) < 2e-4, e


def test_clamped_circular_plate(verify):
    """Clamped disk (MITC4 and MITC3, unstructured) vs Kirchhoff λ² = 10.2158, 39.771 (h/R = 0.005): 0.02 % / 0.14 % at R/50."""
    R, h = 0.5, 0.0025
    D = E * h ** 3 / (12 * (1 - NU ** 2))
    exact = np.array([10.2158, 39.771]) / R ** 2 * np.sqrt(D / (RHO * h)) / (2 * np.pi)
    for quads, size in ((True, 0.01), (False, 0.01)):
        nodes, cells = shell_surface("disk", size, quads=quads, R=R)
        K, M = shell_matrices(nodes, cells, E, NU, RHO, h)
        rim = np.where(np.isclose(np.hypot(nodes[:, 0], nodes[:, 1]), R, atol=1e-9))[0]
        f = lowest(K, M, fixed_dofs(rim), 8)
        f = np.array([f[0], f[5]])                         # (0, 1) and (0, 2) axisymmetric modes
        for k, mode in enumerate(("(0,1)", "(0,2)")):
            verify("Shells", "Clamped disk (h/R = 0.005) vs Kirchhoff", f"{'MITC4' if quads else 'MITC3'} {mode} error",
                   abs(f[k] / exact[k] - 1), bound=0.005, reference="Leissa, λ² = 10.2158, 39.771")
        assert np.all(np.abs(f / exact - 1) < 0.005), (quads, f / exact - 1)


def test_feature_nodes_and_normals():
    nodes, cells = shell_surface("box", 0.25)
    feat = feature_nodes(nodes, cells)
    on_edge = np.sum(np.isclose(nodes, 0) | np.isclose(nodes, 1), axis=1) >= 2
    assert set(feat) == set(np.where(on_edge)[0])
    n, a = cell_normals(nodes, cells["tri3"])
    assert np.isclose(a.sum(), 6.0)
    assert np.allclose(np.abs(n).max(axis=1), 1.0)
