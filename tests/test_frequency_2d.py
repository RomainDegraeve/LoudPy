"""Frequency studies (2D axisymmetric): in-vacuo forced response and the coupled FSI system."""
import numpy as np

import loudpy as lp
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


def test_fsi_circuit(mesh_dir, tmp_path):
    """A 2D model driven by a circuit (loudpy.circuit, its one coil on the model's lp.Coil): an amplifier and the coil
    is exactly the voltage drive; with an amplifier resistance, a filter block and a series inductor: the coil's
    impedance (motional part included) is the voltage drive's, its voltage what the network leaves it; the result
    keeps the circuit (saved and read back) and its coil scalars (Impedance, Thiele–Small)."""
    from loudpy.circuit import Coil, Filter, Inductor, Semi, Source, butterworth
    model = api.speaker_problem(mesh_dir, acoustic=True)
    freqs = [150.0, 700.0]
    v = VOLTAGE
    volt = lp.FrequencyResponse(model, drive=lp.Voltage(v["u"], Re=v["Re"], Bl=v["Bl"], Le=v["Le"], n=v["n"])
                                ).solve(freqs)
    coil = Coil("vc", "c", "0", Semi(Re=v["Re"], Bl=v["Bl"], Le=v["Le"], n=v["n"]))
    circ = lp.FrequencyResponse(model, drive=lp.Circuit(Source("amp", "c", U=v["u"]), coil)).solve(freqs)
    np.testing.assert_allclose(circ.blocks[0].fields["u"], volt.blocks[0].fields["u"], rtol=1e-9,
                               atol=1e-12 * np.abs(volt.blocks[0].fields["u"]).max())
    np.testing.assert_allclose(circ.impedance().values, volt.impedance().values, rtol=1e-9)
    np.testing.assert_allclose(circ.input_impedance().values, volt.impedance().values, rtol=1e-9)
    H = butterworth(2, 400.0)
    net = lp.Circuit(Source("amp", "a", U=v["u"], R_out=0.2), Filter("lp", "a", "f", H), Inductor("L", "f", "c", 5e-4),
                     coil)
    res = lp.FrequencyResponse(model, drive=net).solve(freqs)
    Zc = res.impedance().values
    np.testing.assert_allclose(Zc, volt.impedance().values, rtol=1e-8)
    s = 2j * np.pi * np.asarray(freqs)
    np.testing.assert_allclose(res.voltage("c").values, H.response(freqs) * v["u"] * Zc / (Zc + s * 5e-4), rtol=1e-8)
    res.save(tmp_path / "circuit2d.h5")
    back = lp.load(tmp_path / "circuit2d.h5")
    np.testing.assert_allclose(back.voltage("c").values, res.voltage("c").values, rtol=1e-12)
    assert "I_coil" in back.scalars and [p.name for p in back.circuit.circuit.parts] == ["amp", "lp", "L", "vc"]
