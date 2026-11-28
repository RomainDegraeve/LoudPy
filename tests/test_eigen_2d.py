"""Eigen studies (2D axisymmetric): clamped plate (analytical) and the coarse loudspeaker."""
import numpy as np
import pytest

from support import current_api as api

# λ²_mn of the clamped circular plate (Leissa), first bending mode of each order m
LAMBDA2_FIRST = {0: 10.2158, 1: 21.260, 2: 34.877}


@pytest.fixture(scope="module")
def plate_modes(mesh_dir):
    return api.plate_fourier_modes(mesh_dir, orders=[0, 1, 2])


def test_plate_matches_kirchhoff(plate_modes):
    """First bending mode of m = 0, 1, 2 within 1 % of thin-plate theory (physics sanity)."""
    E, nu, rho, a, h = (api.PLATE[k] for k in ("E", "nu", "rho", "a", "h"))
    scale = np.sqrt(E * h ** 3 / (12 * (1 - nu ** 2)) / (rho * h)) / (2 * np.pi * a ** 2)
    for m, lam2 in LAMBDA2_FIRST.items():
        d = plate_modes[m]
        uz_share = np.linalg.norm(d["shapes"][..., 1], axis=1) / np.linalg.norm(
            d["shapes"].reshape(len(d["freqs"]), -1), axis=1)
        f_bend = np.sort(d["freqs"][uz_share > 0.5])[0]
        assert abs(f_bend - lam2 * scale) / (lam2 * scale) < 0.01, (m, f_bend, lam2 * scale)


@pytest.mark.parametrize("m", [0, 1, 2])
def test_plate_fourier_modes(plate_modes, reference, m):
    d = plate_modes[m]
    reference.check_modes(f"m{m}", d["freqs"], d["shapes"], d["coords"])


@pytest.fixture(scope="module")
def speaker(mesh_dir):
    return api.speaker_modes(mesh_dir, f_max=3000.0, orders=(0, 1))[0]


@pytest.mark.parametrize("m", [0, 1])
def test_speaker_modes(speaker, reference, m):
    d = speaker[m]
    reference.check_modes(f"m{m}", d["freqs"], d["shapes"], d["coords"])
    reference.check(**{f"zeta_m{m}": np.sort(d["zeta"])}, rtol=1e-6)


def test_multishift_takes_its_number_of_shifts(mesh_dir):
    """Modes(method="multishift") with n_shifts: that many shifts over the band (stored with the results), the same
    modes as the default placement within the band; refusals: arpack with orders (multishift computes them),
    n_shifts with arpack or below 1. The schema shows each method's settings only (Param.when)."""
    import loudpy as lp
    model = api.plate_problem(mesh_dir)
    auto = lp.Modes(model).solve(f_max=3000.0, orders=[0])
    three = lp.Modes(model).solve(f_max=3000.0, orders=[0], n_shifts=3, n_per_shift=20)
    assert three.blocks[0].meta["n_shifts"] == 3 and auto.blocks[0].meta["n_shifts"] == 8    # (20 Hz … 3 kHz: ⌈log2 150⌉)
    f_auto, f_three = np.sort(np.asarray(auto.values)), np.sort(np.asarray(three.values))
    assert len(f_three) and np.allclose(f_three[:3], f_auto[:3], rtol=1e-6)
    with pytest.raises(lp.errors.ModelError, match="multishift"):
        lp.Modes(model).solve(f_max=3000.0, orders=[0, 1], method="arpack")
    with pytest.raises(lp.errors.ModelError, match="n_shifts"):
        lp.Modes(model).solve(f_max=3000.0, method="arpack", n_shifts=3)
    with pytest.raises(lp.errors.ModelError, match="n_shifts"):
        lp.Modes(model).solve(f_max=3000.0, n_shifts=0)
    solver = [p for p in lp.schema.block("Modes").params if p.group == "Solver"]
    shown = lambda **v: [p.name for p in solver if p.applies(v)]                       # noqa: E731
    assert shown(physics="structure", method="multishift") == ["method", "n_shifts", "n_per_shift"]
    assert shown(physics="structure", method="arpack") == ["method", "n_modes", "f_target"]
    assert shown(physics="coupled", method="multishift") == ["n_per_shift"]
    assert shown(physics="acoustic", method="multishift") == ["n_modes", "f_target"]


def test_modes_progress_counts_finished_shifts_over_all_orders(mesh_dir):
    """A modes search over several orders reports one count: the shifts finished so far, over every order (orders 0
    and 1 with 3 shifts each: 0/6 … 6/6), so a long first shift shows 0 % and an interface's estimate of the time left
    starts after one shift is done."""
    import loudpy as lp
    from loudpy import progress
    seen = []
    progress.set_callback(lambda what, i, n: seen.append((what, i, n)))
    try:
        lp.Modes(api.plate_problem(mesh_dir)).solve(f_max=3000.0, orders=[0, 1], n_shifts=3, n_per_shift=10)
    finally:
        progress.set_callback(None)
    counts = [(i, n) for _, i, n in seen]
    assert counts[0] == (0, 6) and counts[-1] == (6, 6) and {n for _, n in counts} == {6}
    assert [i for i, _ in counts] == sorted(i for i, _ in counts)
    assert seen[0][0].startswith("order 0: ") and seen[-1][0].startswith("order 1: ")


def test_a_probe_over_mode_sets_with_and_without_u_theta(mesh_dir):
    """Mode sets of orders 0 (u_r, u_z) and m ≥ 1 (u_r, u_z, u_theta) probed together: u_theta 0 for m = 0, one value
    per mode at a point."""
    import loudpy as lp
    meca = api._meshed(api.speaker_model(), mesh_dir, api._speaker_tag(1e-3, False, False))
    modes = lp.Modes(meca).solve(f_max=2500.0, f_min=20.0, orders=[0, 1], n_per_shift=10)
    pt = tuple(np.asarray(modes.blocks[0].mesh_of("u").coords)[0, :2])
    u = modes.displacement(at=pt)
    assert u.dims == ("mode", "component") and u.shape == (len(modes), 3)
    n0 = len(modes.blocks[0])
    np.testing.assert_array_equal(u.values[:n0, 2], 0.0)
    assert modes.displacement(at=pt, component="z").shape == (len(modes),)
    assert isinstance(modes, lp.ModeResults)
