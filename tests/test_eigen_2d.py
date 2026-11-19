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
