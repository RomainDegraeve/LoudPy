"""
Verification against analytical solutions: the numbers are right, not only unchanged.

  clamped circular plate     axisymmetric + Fourier-order structure vs Kirchhoff plate theory
  rectangular rigid cavity   3D P2 acoustics vs the exact modes, and the observed convergence order
  pulsating hemisphere       3D exterior FE + half-space DtN vs the exact radiated pressure (curved P2)
  air                        lp.air (ideal-gas mixture) vs Cramer's speed-of-sound correlation
  structure–air coupling     swept volume, energy balance, radiated power, piston antisymmetry

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
                                        lp.Clamped("interface_constrained_rz"), lp.Clamped("interface_constrained_r", dofs="r"))
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
    Exterior FE + exact half-space DtN against the analytical monopole field at 1 m, on isoparametric
    P2 tetrahedra (gmsh's mid-edge nodes on the spheres). The former straight-sided cells (faceted
    spheres) gave 1.8 % / 1.4 % at h = 3.5 cm; curved cells 0.040 % / 0.081 % (200 Hz / 1 kHz; one cell of
    that mesh is inverted by the curved sphere and repaired when read, see fem.tet10.untangle_p2).
    Bound: 0.1 % at h = 3.5 cm, and the worst error smaller than at h = 5 cm.
    """
    errs = []
    for h in (0.05, 0.035):
        d = tmp_path / f"h{h}"
        d.mkdir()
        fe, exact = api.hemisphere_radiation(d, [200.0, 1000.0], h=h, curved=True)
        errs.append(np.linalg.norm(fe - exact, axis=1) / np.linalg.norm(exact, axis=1))
    coarse, fine = errs
    assert fine.max() < coarse.max() and fine.max() < 1e-3, (coarse, fine)


# ── structure–air coupling ───────────────────────────────────────────────────
@pytest.fixture(scope="module")
def fsi_energy(mesh_dir):
    return api.speaker_fsi_energy(mesh_dir, [50.0, 300.0, 2000.0])


def test_coupling_sweeps_the_right_volume(fsi_energy):
    """
    A unit rigid axial translation of the structure sweeps π r_max² through each interface (both
    span from the axis to r_max): front +, rear − (n_s points into the front / rear air). Checks the
    coupling normal of every edge, whatever the direction of the curves in the geometry.
    """
    for name, V in fsi_energy["swept"].items():
        exact = np.pi * fsi_energy["r_max"][name] ** 2
        assert abs(abs(V) - exact) < 1e-9 * exact, (name, V, exact)
        assert np.sign(V) == (1 if "front" in name else -1), name


def test_coupled_energy_balance(fsi_energy):
    """
    Power given by the coil = power dissipated in the structure + power radiated into each air
    region, each radiated power positive (the PML absorbs it). Exact for the discrete system when
    the coupling is consistent: 1e-8 (measured ~1e-10). A sign error on one side breaks it at the
    percent level.
    """
    e = fsi_energy
    for k, f in enumerate(e["freqs"]):
        radiated = [P[k] for P in e["P_air"].values()]
        assert all(P > 0 for P in radiated), (f, radiated)
        residual = e["P_in"][k] - e["P_diss"][k] - sum(radiated)
        assert abs(residual) < 1e-8 * e["P_in"][k], (f, residual / e["P_in"][k])


def test_piston_front_and_rear_in_antiphase(fsi_energy):
    """At 50 Hz the cone moves as a piston: the pressure on the axis is opposite in front and behind it."""
    pf, pr = fsi_energy["p_front"][0], fsi_energy["p_rear"][0]
    assert np.real(pf * np.conj(pr)) < 0, (pf, pr)


# ── air ──────────────────────────────────────────────────────────────────────
# Cramer, "The variation of the specific heat ratio and the speed of sound in air with temperature,
# pressure, humidity, and CO2 concentration", JASA 93 (1993) 2510, eq. (15) and table III; the
# saturation pressure and enhancement factor of Davis (1992), as used by Cramer. CO2: 400 ppm.
_CRAMER = (331.5024, 0.603055, -0.000528, 51.471935, 0.1495874, -0.000782, -1.82e-7, 3.73e-8, -2.93e-10,
           -85.20931, -0.228525, 5.91e-5, -2.835149, -2.15e-13, 29.179762, 0.000486)


def _cramer_c(t, p, rh, xc=4e-4):
    a, T = _CRAMER, t + 273.15
    psv = np.exp(1.2378847e-5 * T ** 2 - 1.9121316e-2 * T + 33.93711047 - 6.3431645e3 / T)
    xw = rh * (1.00062 + 3.14e-8 * p + 5.6e-7 * t ** 2) * psv / p
    return (a[0] + a[1] * t + a[2] * t ** 2 + (a[3] + a[4] * t + a[5] * t ** 2) * xw
            + (a[6] + a[7] * t + a[8] * t ** 2) * p + (a[9] + a[10] * t + a[11] * t ** 2) * xc
            + a[12] * xw ** 2 + a[13] * p ** 2 + a[14] * xc ** 2 + a[15] * xw * p * xc)


def test_air_matches_cramer():
    """
    Speed of sound of lp.air within 0.06 % of Cramer over 0–40 °C, 80–101 kPa, 0–100 % RH (measured
    0.048 %; the ideal-gas model ignores the small dispersion of γ with temperature and the CO2).
    """
    worst = 0.0
    for t, p, rh in itertools.product(range(0, 41, 5), (80e3, 90e3, 101325.0), (0.0, 0.25, 0.5, 0.75, 1.0)):
        ours, ref = lp.air(t + 273.15, p, rh).c, _cramer_c(t, p, rh)
        worst = max(worst, abs(ours - ref) / ref)
    assert worst < 6e-4, worst


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
