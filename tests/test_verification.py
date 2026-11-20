"""
Verification against analytical solutions: the numbers are right, not only unchanged.

  clamped circular plate     axisymmetric + Fourier-order structure vs Kirchhoff plate theory
  rectangular rigid cavity   3D P2 acoustics vs the exact modes, and the observed convergence order
  pulsating hemisphere       3D exterior FE + half-space DtN vs the exact radiated pressure

Each case states its tolerance and why. They run in a few seconds (fast suite, every CI run).
"""
import itertools

import numpy as np
import pytest

import loudpy as lp
from support import current_api as api


# ── clamped circular plate (Kirchhoff) ───────────────────────────────────────
E, NU, RHO, A, H = 2.1e11, 0.3, 7850.0, 0.1, 0.002
# λ²_mn of the clamped circular plate (Leissa, Vibration of Plates, NASA SP-160, table 2.1)
LAMBDA2 = {(0, 1): 10.2158, (0, 2): 39.771, (1, 1): 21.260, (2, 1): 34.877}


def _bending(modes_of_order, m):
    """Frequencies of the bending modes (U_z dominant) of one azimuthal order, sorted."""
    U = modes_of_order.blocks[0].fields["u"]
    bend = np.linalg.norm(U[..., 1], axis=1) > 0.5 * np.linalg.norm(U.reshape(len(U), -1), axis=1)
    return np.sort(modes_of_order.frequencies if m == 0 else modes_of_order.frequencies[bend])


def test_clamped_plate_matches_kirchhoff(tmp_path):
    """
    3D elasticity (no plate assumption) on an h/a = 0.02 plate: within 0.5 % of Kirchhoff and below it
    (shear and rotary inertia lower the frequencies; the mesh is converged to 0.02 % here).
    """
    steel = lp.Material("Steel", rho=RHO, E=E, nu=NU)
    model = lp.Model(api.PLATE_GEO).add(lp.Solid("plate", steel, damping=lp.Undamped(), mesh_size=H / 2),
                                        lp.Clamped("interface_constrained"))
    model.mesh(tmp_path / "plate.msh", deterministic=True)
    modes = lp.Modes(model).solve(f_min=10.0, f_max=2500.0, orders=[0, 1, 2], n_per_shift=10)
    scale = np.sqrt(E * H ** 3 / (12 * (1 - NU ** 2)) / (RHO * H)) / (2 * np.pi * A ** 2)
    for (m, n), lam in LAMBDA2.items():
        f_fe, f_ex = _bending(modes.order(m), m)[n - 1], lam * scale
        assert -0.005 < (f_fe - f_ex) / f_ex < 0.0, (m, n, f_fe, f_ex)


# ── rectangular rigid cavity ─────────────────────────────────────────────────
def _cavity_error(tmp_path, h):
    lx, ly, lz = 0.30, 0.20, 0.14
    d = tmp_path / f"h{h}"
    d.mkdir()
    freqs, cav = api.rect_cavity_modes(d, h=h, f_max=1500.0)
    exact = np.sort([cav.c / 2 * np.sqrt((i / lx) ** 2 + (j / ly) ** 2 + (k / lz) ** 2)
                     for i, j, k in itertools.product(range(8), repeat=3)])[:len(freqs)]
    assert abs(freqs[0]) < 1e-6                                       # the constant mode
    return np.max(np.abs(freqs[1:] - exact[1:]) / exact[1:])


def test_rect_cavity_modes_converge_at_order_four(tmp_path):
    """
    P2 tetrahedra: eigenvalue errors O(h^{2p}) = O(h^4). Observed order > 3.3 between h = 3 cm and 2 cm,
    and every mode below 1.5 kHz within 2e-4 at h = 2 cm.
    """
    e1, e2 = _cavity_error(tmp_path, 0.03), _cavity_error(tmp_path, 0.02)
    order = np.log(e1 / e2) / np.log(0.03 / 0.02)
    assert e2 < 2e-4 and order > 3.3, (e1, e2, order)


# ── pulsating hemisphere on a rigid ground ───────────────────────────────────
def test_pulsating_hemisphere_radiation(tmp_path):
    """
    Exterior FE + exact half-space DtN against the analytical monopole field at 1 m. The meshes are
    straight-sided P2 (faceted sphere): the error is dominated by the geometry and decreases slowly
    with h. Bound: 2.2 % at h = 3.5 cm, and smaller than at h = 5 cm.
    """
    errs = []
    for h in (0.05, 0.035):
        d = tmp_path / f"h{h}"
        d.mkdir()
        fe, exact = api.hemisphere_radiation(d, [200.0, 1000.0], h=h)
        errs.append(np.linalg.norm(fe - exact, axis=1) / np.linalg.norm(exact, axis=1))
    coarse, fine = errs
    assert np.all(fine < coarse) and fine.max() < 0.022, (coarse, fine)


# ── units and plausibility checks ────────────────────────────────────────────
def test_invalid_and_implausible_inputs():
    with pytest.raises(lp.errors.ModelError, match="valid range"):
        lp.Solid("cone", "Paper", nu=0.5)
    with pytest.raises(lp.errors.ModelError):
        lp.Material("x", rho=-1.0)
    with pytest.warns(lp.errors.PlausibilityWarning, match="metres"):
        lp.Solid("cone", "Paper", mesh_size=2.0)                       # 2 m: probably meant 2 mm
    with pytest.warns(lp.errors.PlausibilityWarning):
        lp.Material("soft", rho=1000.0, E=2.0, nu=0.3)                 # 2 Pa: probably meant GPa


def test_exceptions_keep_their_builtin_bases():
    """loudpy errors are also the built-in exceptions they replace (former except clauses still work)."""
    e = lp.errors
    assert issubclass(e.SelectionError, KeyError) and issubclass(e.SelectionError, ValueError)
    assert issubclass(e.ModelError, ValueError) and issubclass(e.StateError, RuntimeError)
    assert str(e.ResultsError("no field 'p'")) == "no field 'p'"       # no KeyError quoting
