"""Frequency studies (2D axisymmetric): in-vacuo forced response and the coupled FSI system."""
import numpy as np
import pytest

from support import current_api as api
from support.reference import RTOL_RESPONSE

FREQS_MECA = [50.0, 116.0, 500.0, 2000.0, 8000.0]
VOLTAGE = dict(Re=8.0, Le=4e-3, n=0.8, Bl=7.0, u=np.sqrt(8.0))


@pytest.fixture(scope="module")
def meca(mesh_dir):
    return api.speaker_freq_meca(mesh_dir, FREQS_MECA)[0]


def test_meca_response(meca, reference):
    for f, u in zip(FREQS_MECA, meca["u"]):
        reference.check_nodal(f"u_{f:.0f}Hz", meca["coords"], u, rtol=RTOL_RESPONSE)


def test_meca_static_limit(meca):
    """Far below the first mode the response is quasi-static: real and in phase with the force."""
    u = meca["u"][0]
    i = np.argmax(np.abs(u[:, 1]))
    assert abs(u[i, 1].imag) < 0.05 * abs(u[i, 1].real)


@pytest.mark.parametrize("freq", [200.0, 1500.0])
def test_fsi_force(mesh_dir, reference, freq):
    r = api.speaker_fsi(mesh_dir, freq)
    reference.check_nodal("u", r["coords_u"], r["u"], rtol=RTOL_RESPONSE)
    reference.check_nodal("p", r["coords_p"], r["p"], rtol=RTOL_RESPONSE)


def test_fsi_voltage(mesh_dir, reference):
    r = api.speaker_fsi(mesh_dir, 300.0, voltage=VOLTAGE)
    reference.check_nodal("u", r["coords_u"], r["u"], rtol=RTOL_RESPONSE)
    reference.check_nodal("p", r["coords_p"], r["p"], rtol=RTOL_RESPONSE)
    reference.check(I_coil=r["I_coil"])
    assert abs(r["I_coil"]) < VOLTAGE["u"] / VOLTAGE["Re"] * 1.01      # back-EMF lowers the current
