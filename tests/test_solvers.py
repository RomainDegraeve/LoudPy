"""
Solvers layer on its own: small problems with known answers, no mesh and no physics.
"""
import numpy as np
import pytest
import scipy.sparse as sp

from loudpy.solvers import eigen as eig
from loudpy.solvers.linear import SweepSolver

OMEGA = 2 * np.pi * np.array([50.0, 120.0, 300.0, 700.0, 1500.0, 2600.0, 4000.0])


def chain(omega=OMEGA, alpha=None, beta=None):
    """Uncoupled oscillators: K = diag(ω²), M = I, C = α M + β K (or None)."""
    n = len(omega)
    K, M = eig.to_csc_complex(sp.diags(omega ** 2), sp.identity(n))
    C = None if alpha is None else eig.to_csc_complex(alpha * M + beta * K)[0]
    return K, M, C


def test_gep_undamped_frequencies():
    K, M, _ = chain()
    s, phi = eig.multishift(K, M, None, eig.default_shifts(20.0, 5000.0), 4, hysteretic=False)
    f, zeta, eta, s, phi = eig.filter_modes(s, phi, 20.0, 5000.0, filter_halfplane=False)
    s, phi = eig.mac_deduplicate(s, phi)
    f, idx = np.unique(np.round(np.abs(s) / 2 / np.pi, 6), return_index=True)
    np.testing.assert_allclose(f, OMEGA / 2 / np.pi, rtol=1e-10)


def test_qep_rayleigh_damping():
    """s = −ζω ± iω√(1 − ζ²), ζ = (α/ω + β ω)/2 for C = α M + β K."""
    alpha, beta = 20.0, 2e-6
    K, M, C = chain(alpha=alpha, beta=beta)
    s, phi = eig.multishift(K, M, C, eig.default_shifts(20.0, 5000.0), 4, hysteretic=False)
    s, phi = eig.mac_deduplicate(s, phi)
    f, zeta, _, s, phi = eig.filter_modes(s, phi, 20.0, 5000.0, filter_halfplane=True)
    zeta_exact = 0.5 * (alpha / OMEGA + beta * OMEGA)
    np.testing.assert_allclose(zeta, zeta_exact, rtol=1e-6)        # ARPACK tolerance 1e-7
    np.testing.assert_allclose(np.abs(s), OMEGA, rtol=1e-6)        # |s| = ω
    phi_n = eig.normalise_modes(phi, s, M, C, hysteretic=False, viscous=True)
    for k in range(phi_n.shape[1]):               # φᵀ(2sM + C)φ = 1
        v = phi_n[:, k]
        assert abs(v @ (2 * s[k] * (M @ v) + C @ v) - 1) < 1e-10


def test_covering_finds_every_mode_in_the_band():
    K, M, C = chain(alpha=20.0, beta=2e-6)
    s, phi, shifts = eig.covering(K, M, C, 20.0, 5000.0, 3)
    f = np.sort(np.abs(s[s.imag > 0]) / 2 / np.pi)
    np.testing.assert_allclose(f, OMEGA / 2 / np.pi, rtol=1e-8)


def test_arpack_is_reproducible():
    K, M, _ = chain()
    op = eig.shift_invert_operator(K - 1e5 * M, lambda x: M @ x, K.shape[0])
    a, b = eig.arpack_eigs(op, 3), eig.arpack_eigs(op, 3)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_mac_deduplicate_keeps_degenerate_pairs():
    s = np.array([10.0 + 100j, 10.0 + 100.2j, 10.0 + 100.1j])
    phi = np.array([[1.0, 1.0, 0.0], [0.0, 1e-9, 1.0]], dtype=complex)  # 0 and 1 same, 2 orthogonal
    s2, phi2 = eig.mac_deduplicate(s, phi, freq_tol_rad=1.0)
    assert len(s2) == 2 and np.allclose(np.abs(phi2[:, 1]), [0, 1])


def test_filter_modes_keeps_upper_half_plane():
    s = np.array([-1 + 2 * np.pi * 100j, -1 - 2 * np.pi * 100j, -5000.0 + 0j, 3 + 2 * np.pi * 200j])
    phi = np.eye(4, dtype=complex)
    with pytest.warns(UserWarning):                  # the growing one (Re s > 0) is reported
        f, zeta, _, s_kept, _ = eig.filter_modes(s, phi, 10.0, 1000.0, filter_halfplane=True)
    assert len(s_kept) == 1 and s_kept[0].imag > 0


def test_sweep_solver_matches_direct_solve():
    rng = np.random.default_rng(1)
    n = 200
    A0 = sp.diags(2 + rng.random(n)) + sp.diags(-np.ones(n - 1), 1) + sp.diags(-np.ones(n - 1), -1)
    s = SweepSolver()
    for w in (1.0, 1.01, 1.02, 2.0):
        A = (A0 * w).tocsc().astype(complex)
        b = rng.standard_normal(n) + 0j
        x = s.solve(A, b)
        assert np.linalg.norm(A @ x - b) <= 1e-8 * np.linalg.norm(b)


def test_generalized_alpha_second_order():
    """
    Linear undamped oscillator from rest under F(t) = F0 (1 − cos ω_f t) (starts with zero force and
    zero slope), exact solution known; generalised-α (ρ∞ = 1) must be second-order accurate.
    """
    from loudpy.solvers.time import free_dof_reducer, generalized_alpha_newton
    k, m, F0 = 4.0e4, 1.0, 1.0
    w = np.sqrt(k / m)
    wf = w / 3
    K, M, zero = sp.csr_matrix([[k]]), sp.csc_matrix([[m]]), sp.csr_matrix([[0.0]])
    tangent = lambda u: (K, zero, K @ u)
    T = 20 * 2 * np.pi / w

    def run(n):
        dt = T / n
        t = np.arange(n + 1) * dt
        out = generalized_alpha_newton(tangent, M, free_dof_reducer(K, np.array([0])), np.array([0]), 1,
                                       np.array([F0]), 1 - np.cos(wf * t), dt, n, rho_inf=1.0,
                                       nr_tol=1e-12, log=lambda *a: None)
        exact = F0 / k * (1 - np.cos(w * t)) - F0 / (m * (w ** 2 - wf ** 2)) * (np.cos(wf * t) - np.cos(w * t))
        return np.abs(out[1][:, 0] - exact).max(), out[4]

    e1, n_fact = run(2000)
    e2, _ = run(4000)
    assert e1 < 1e-2 * F0 / k                          # 100 steps per period: accurate
    assert 3.5 < e1 / e2 < 4.5                         # second order
    assert n_fact == 1                                 # linear: the tangent is factorised once
