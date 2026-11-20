"""
3D coupling (slow): cavity modes, half-space DtN radiation, a driver in a box, several drivers.

Cavity mode shapes have arbitrary signs (and can mix inside degenerate pairs), so the
references use quantities that do not depend on them: frequencies, S Sᵀ, R, the driver
response q and pressures at probe points.
"""
import itertools

import numpy as np
import pytest

import loudpy as lp

from support import current_api as api

pytestmark = pytest.mark.slow
FREQS = np.array([60.0, 120.0, 300.0, 900.0, 1500.0])
# CavityModes.solve starts ARPACK from a fixed vector: identical runs give identical cavity modes on one
# machine. Across machines the forced responses near resonances spread by ~3e-8 (support.reference).
from support.reference import RTOL_COUPLED_MODES, RTOL_RESPONSE as RTOL_3D  # noqa: E402


def test_rect_cavity(mesh_dir, reference):
    freqs, cav = api.rect_cavity_modes(mesh_dir)
    lx, ly, lz = 0.30, 0.20, 0.14
    exact = np.sort([cav.c / 2 * np.sqrt((a / lx) ** 2 + (b / ly) ** 2 + (c / lz) ** 2)
                     for a, b, c in itertools.product(range(10), repeat=3)])[:12]
    assert np.abs(freqs[1:12] - exact[1:12]).max() / exact[11] < 0.01      # physics sanity
    reference.check(freqs=freqs, rtol=1e-7)
    reference.check(volume=cav.volume, rtol=1e-12)


def test_hemisphere_dtn(mesh_dir, reference):
    fe, exact = api.hemisphere_radiation(mesh_dir, [100.0, 1000.0])
    err = np.linalg.norm(fe - exact, axis=1) / np.linalg.norm(exact, axis=1)
    assert err.max() < 0.03, err                                            # physics sanity
    reference.check(p_far=fe, rtol=1e-7)


@pytest.fixture(scope="module")
def drv(mesh_dir, cache_dir):
    return api.driver_modes(mesh_dir, cache_dir)


@pytest.fixture(scope="module")
def coil(drv):
    return api.coil_nodes(drv)


def test_driver_in_box(mesh_dir, drv, coil, reference):
    center = (api.BOX[0] / 2, api.BOX[1] / 2)
    cav = api.box_cavity(mesh_dir, [center], "box_one_driver")
    cpl = api.coupled_box(cav, drv, center=center)
    F = cpl.coil_force(coil)
    q, P = cpl.forced(FREQS, F)
    probe = cav.nearest_node((0.03, 0.03, api.Z_B - 0.12))
    s, _, _ = cpl.coupled_modes(f_max=1500.0)
    reference.check(q=q, p_probe=P @ cav.psi[probe], R=cpl.R, SSt=cpl.S @ cpl.S.T,
                    wet_area=cpl.wet_area, cav_freqs=cav.freqs, rtol=RTOL_3D)
    reference.check(coupled=s, rtol=RTOL_COUPLED_MODES)
    # physics sanity: the box air spring raises the pistonic resonance
    k0 = int(np.argmax(np.abs(cpl.volume_disp) * (drv.m == 0)))
    f_c = np.sqrt(drv.freq[k0] ** 2 + lp.materials["Air"].rho * cav.c ** 2 * cpl.volume_disp[k0] ** 2 / cav.volume / (2 * np.pi) ** 2)
    f_coupled = s.imag[np.argmin(np.abs(s.imag / 2 / np.pi - f_c))] / 2 / np.pi
    assert drv.freq[k0] < f_coupled < 1.05 * f_c


@pytest.fixture(scope="module")
def two_drivers(mesh_dir, drv):
    from loudpy.physics.modal import MultiDriver
    W, H, _ = api.BOX
    centers = [(0.07, H / 2), (W - 0.07, H / 2)]                        # mirror pair about x = W/2
    cav = api.box_cavity(mesh_dir, centers, "box_two_drivers")
    drivers = MultiDriver(drv, centers, names=["left", "right"])
    return cav, drivers, api.coupled_box(cav, drivers)


def test_two_drivers_transfer(two_drivers, coil, reference):
    cav, drivers, cpl = two_drivers
    Fs = cpl.coil_forces(coil)
    q, P = cpl.forced(FREQS, Fs)                                        # (n_f, n, 2)
    u = np.einsum("ni,fnj->fij", Fs, q)                                 # coil displacement i ← force j
    rel = lambda a, b: np.abs(a - b).max() / np.abs(b).max()
    assert rel(u, u.transpose(0, 2, 1)) < 1e-10                         # reciprocity (exact)
    assert rel(u[:, 0, 0], u[:, 1, 1]) < 5e-3                           # mirror symmetry (mesh)
    reference.check(u=u, R=cpl.R, SSt=cpl.S @ cpl.S.T, wet_areas=np.array(cpl.wet_areas), rtol=RTOL_3D)


def test_frame_rotation_invariance(two_drivers, coil):
    """Rotating + translating the whole model (and the frames) leaves the coupling unchanged."""
    from scipy.spatial.transform import Rotation
    from loudpy.geometry.frame import Frame as DriverFrame
    from loudpy.physics.modal import MultiDriver, CavityModes
    cav, drivers, cpl = two_drivers
    Rg = Rotation.from_euler("zyx", [37, -61, 23], degrees=True).as_matrix()
    t = np.array([0.3, -1.2, 0.7])
    rot = CavityModes(cav.nodes @ Rg.T + t, cav.tets, cav.patch_tris, cav.patches)
    rot.kappa2, rot.psi, rot.c, rot.volume = cav.kappa2, cav.psi, cav.c, cav.volume
    frames = [DriverFrame(Rg @ f.origin + t, Rg) for f in drivers.frames]
    cpl_r = api.coupled_box(rot, MultiDriver(drivers.modes, frames))
    assert np.abs(cpl_r.Bc - cpl.Bc).max() < 1e-10 * np.abs(cpl.Bc).max()
    assert np.abs(cpl_r.R - cpl.R).max() < 1e-8 * np.abs(cpl.R).max()
