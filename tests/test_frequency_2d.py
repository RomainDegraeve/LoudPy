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


def test_redrive_2d(mesh_dir, tmp_path):
    """The structure and the air are solved for a unit force on the coil, the drive weights it: a result changes its
    drive afterwards (Results.redrive) exactly as solving again — a circuit for another, a force for a voltage —
    saved and read back, and through an lp.Output keeping every other frequency."""
    from loudpy.circuit import Capacitor, Coil, Filter, Inductor, Resistor, Semi, Source, butterworth, peq
    model = api.speaker_problem(mesh_dir, acoustic=True)
    freqs = [150.0, 400.0, 700.0, 1500.0]
    v = VOLTAGE
    coil = Coil("vc", "c", "0", Semi(Re=v["Re"], Bl=v["Bl"], Le=v["Le"], n=v["n"]))
    a = lp.Circuit(Source("amp", "a", U=v["u"]), Filter("lp", "a", "f", butterworth(2, 400.0)), Inductor("L", "f", "c", 5e-4),
                   coil)
    b = lp.Circuit(Source("amp", "a", U=2.0, R_out=0.1), Filter("eq", "a", "f", [peq(500.0, 6.0, 2.0)]),
                   Resistor("R", "f", "c", 1.0), Capacitor("C", "c", "0", 2e-5), coil)
    res_a = lp.FrequencyResponse(model, drive=a).solve(freqs)
    direct = lp.FrequencyResponse(model, drive=b).solve(freqs)
    assert res_a.redrivable
    res_a.save(tmp_path / "a.h5")
    for src in (res_a, lp.load(tmp_path / "a.h5")):
        again = src.redrive(b)
        for name in ("u", "p"):
            ref = np.asarray(direct.blocks[0].fields[name])
            np.testing.assert_allclose(np.asarray(again.blocks[0].fields[name]), ref, rtol=1e-7, atol=1e-9 * np.abs(ref).max())
        np.testing.assert_allclose(again.voltage("c").values, direct.voltage("c").values, rtol=1e-9)
        np.testing.assert_allclose(again.impedance().values, direct.impedance().values, rtol=1e-9)
        assert again.circuit.circuit.to_dict() == b.to_dict() and again.study["drive"]["type"] == "Circuit"
    force = lp.FrequencyResponse(model, drive=lp.Force(2.5)).solve(freqs)
    volt = lp.FrequencyResponse(model, drive=lp.Voltage(v["u"], Re=v["Re"], Bl=v["Bl"], Le=v["Le"], n=v["n"])
                                ).solve(freqs)
    from_force = force.redrive(lp.Voltage(v["u"], Re=v["Re"], Bl=v["Bl"], Le=v["Le"], n=v["n"]))
    np.testing.assert_allclose(from_force.impedance().values, volt.impedance().values, rtol=1e-9)
    np.testing.assert_allclose(np.asarray(from_force.blocks[0].fields["u"]), np.asarray(volt.blocks[0].fields["u"]),
                               rtol=1e-7, atol=1e-9 * np.abs(np.asarray(volt.blocks[0].fields["u"])).max())
    back = volt.redrive(lp.Force(2.5))
    np.testing.assert_allclose(np.asarray(back.blocks[0].fields["u"]), np.asarray(force.blocks[0].fields["u"]),
                               rtol=1e-7, atol=1e-9 * np.abs(np.asarray(force.blocks[0].fields["u"])).max())
    thin = lp.FrequencyResponse(model, drive=a).solve(freqs, output=lp.Output(every=2))
    np.testing.assert_allclose(thin.redrive(b).impedance().values, direct.impedance().values[::2], rtol=1e-9)
    meca = api.speaker_problem(mesh_dir)                                         # the structure alone, no air
    z = lp.FrequencyResponse(meca, drive=lp.Voltage(v["u"], Re=v["Re"], Bl=v["Bl"])).solve(freqs).impedance().values
    assert np.all(z.real >= v["Re"] - 1e-9) and np.all(np.abs(z - v["Re"]) > 1e-6)
    with pytest.raises(lp.errors.ModelError, match="both set the voltage"):
        res_a.redrive(lp.Circuit(Source("amp", "a"), Filter("F1", "a", "c", butterworth(2, 400.0)),
                                 Filter("F2", "a", "c", butterworth(2, 900.0, "high")), coil))


def test_motor_on_the_coil(mesh_dir, tmp_path):
    """The motor sits on the model's lp.Coil item (motor=): a Voltage with its amplitude only and a circuit's coil
    without a model drive it, as the same values given in the drive do; values given in the drive override it; the
    result keeps it (redrive, saved and read back) and so does the model's description; no motor: refused."""
    from loudpy.circuit import Coil, Inductor, Semi, Source
    v = VOLTAGE
    motor = Semi(Re=v["Re"], Bl=v["Bl"], Le=v["Le"], n=v["n"])
    model = api.speaker_problem(mesh_dir, acoustic=True)
    coil_item = next(b for b in model.boundaries if isinstance(b, lp.Coil))
    freqs = [150.0, 700.0]
    with pytest.raises(lp.errors.ModelError, match="no motor"):
        lp.FrequencyResponse(model, drive=lp.Voltage(v["u"])).solve(freqs)
    given = lp.FrequencyResponse(model, drive=lp.Voltage(v["u"], Re=v["Re"], Bl=v["Bl"], Le=v["Le"], n=v["n"])
                                 ).solve(freqs)
    coil_item.motor = motor
    own = lp.FrequencyResponse(model, drive=lp.Voltage(v["u"])).solve(freqs)
    np.testing.assert_allclose(own.impedance().values, given.impedance().values, rtol=1e-12)
    net = lp.Circuit(Source("amp", "a", U=v["u"]), Inductor("L", "a", "c", 5e-4), Coil("vc", "c", "0"))
    res = lp.FrequencyResponse(model, drive=net).solve(freqs)
    explicit = lp.FrequencyResponse(model, drive=lp.Circuit(Source("amp", "a", U=v["u"]), Inductor("L", "a", "c", 5e-4),
                                                            Coil("vc", "c", "0", motor))).solve(freqs)
    np.testing.assert_allclose(res.impedance().values, explicit.impedance().values, rtol=1e-12)
    other = lp.FrequencyResponse(model, drive=lp.Voltage(v["u"], Re=2 * v["Re"], Bl=v["Bl"])).solve(freqs)
    assert np.all(np.abs(other.impedance().values - own.impedance().values) > 1.0)            # the drive's own coil
    described = next(b for b in res.model["boundaries"] if b["type"] == "Coil")             # (a scene's driver
    assert res.motors == {None: motor} and described["motor"] == {"type": "Semi", "Re": v["Re"], "Bl": v["Bl"],
                                                                  "Le": v["Le"], "n": v["n"]}  # reads it there)
    back = lp.load(res.save(tmp_path / "motor.h5"))
    assert back.motors == {None: motor}
    np.testing.assert_allclose(back.redrive(lp.Voltage(v["u"])).impedance().values, own.impedance().values, rtol=1e-9)
    with pytest.raises(lp.errors.ModelError, match="Re and Bl together"):
        lp.Voltage(2.83, Re=8.0)
    with pytest.raises(lp.errors.ModelError, match="motor"):
        lp.Coil("x", motor=6.0)


def test_named_circuits_and_parameters(mesh_dir, tmp_path):
    """A simulation's circuits are named once in its document, a study's drive names one (CircuitRef), the script
    writes each once as a variable that the studies use; parameters (lp.P) in a circuit's values — a filter stage's
    too — take the model's values when it is solved, and are swept like any parameter."""
    from loudpy.circuit import Coil, Filter, Resistor, Semi, Source, linkwitz_riley
    v = VOLTAGE
    model = api.speaker_problem(mesh_dir, acoustic=True)
    model.parameters = lp.Parameters(fc=400.0, r=1.0)
    next(b for b in model.boundaries if isinstance(b, lp.Coil)).motor = Semi(Re=v["Re"], Bl=v["Bl"], Le=v["Le"],
                                                                                n=v["n"])

    def xo(fc, r):
        return lp.Circuit(Source("amp", "a", U=v["u"]), Filter("lo", "a", "f", [linkwitz_riley(4, fc, "low")]),
                          Resistor("R", "f", "c", r), Coil("vc", "c", "0"))
    freqs = [150.0, 700.0]
    direct = lp.FrequencyResponse(model, drive=xo(400.0, 1.0)).solve(freqs)
    res = lp.FrequencyResponse(model, drive=xo(lp.P("fc"), 2 * lp.P("r") - 1.0)).solve(freqs)
    np.testing.assert_allclose(res.voltage("c").values, direct.voltage("c").values, rtol=1e-12)
    assert res.circuit.circuit.parameters == []                                  # (the result: its values)
    with pytest.raises(lp.errors.ModelError, match=r"parameter\(s\) \['fc', 'r'\]"):
        xo(lp.P("fc"), lp.P("r")).check()

    ref = {"type": "CircuitRef", "params": {"name": "xo"}}
    doc = lp.Document.of(model, studies=[{"name": "fr", "type": "FrequencyResponse",
                                          "params": {"drive": ref, "freqs": freqs}}],
                         circuits={"xo": xo(lp.P("fc"), lp.P("r"))})
    assert doc.check() == []
    doc.save(tmp_path / "spk.loudpy.json")
    back = lp.Document.load(tmp_path / "spk.loudpy.json")
    assert back.to_dict() == doc.to_dict() and list(back.circuits) == ["xo"]
    study, kw = back.study("fr", model)
    np.testing.assert_allclose(study.solve(**kw).voltage("c").values, direct.voltage("c").values, rtol=1e-12)
    text = back.to_script(tmp_path)
    assert "\nxo = lp.Circuit(" in text and "fc=lp.P('fc')" in text and "drive=xo" in text
    compile(text, "spk_ui.py", "exec")
    bad = back.copy()
    bad.studies[0]["params"]["drive"] = {"type": "CircuitRef", "params": {"name": "nope"}}
    bad.circuits["model"] = bad.circuits["xo"]
    problems = bad.check()
    assert any("no circuit 'nope'" in p for p in problems) and any("circuit 'model'" in p for p in problems)

    sweep = lp.Sweep(lp.FrequencyResponse(model, drive=xo(lp.P("fc"), lp.P("r"))), "fc", [400.0, 800.0])
    assert sweep.electrical_only() == (True, "")                                 # one solve, the runs re-driven
    runs = sweep.solve(freqs).results
    np.testing.assert_allclose(runs[1].voltage("c").values,
                               lp.FrequencyResponse(model, drive=xo(800.0, 1.0)).solve(freqs).voltage("c").values,
                               rtol=1e-9)
    assert runs[1].model.parameters == {"fc": 800.0, "r": 1.0}
    every = lp.Sweep(lp.FrequencyResponse(model, drive=xo(lp.P("fc"), lp.P("r"))), "fc", [400.0, 800.0],
                     redrive=False).solve(freqs).results                          # solved again: the same
    np.testing.assert_allclose(runs[1].voltage("c").values, every[1].voltage("c").values, rtol=1e-9)


def test_sweep_by_redrive_and_its_limits(mesh_dir):
    """A sweep of a motor's value runs by redrive (the coil item's motor= with a parameter); a parameter reaching the
    structure, or a study other than a frequency response, is solved again — redrive=True says why."""
    from loudpy.circuit import Semi
    v = VOLTAGE
    model = api.speaker_problem(mesh_dir, acoustic=True)
    model.parameters = lp.Parameters(bl=v["Bl"], eta=0.01)
    coil = next(b for b in model.boundaries if isinstance(b, lp.Coil))
    coil.motor = Semi(Re=v["Re"], Bl=lp.P("bl"), Le=v["Le"], n=v["n"])
    freqs = [150.0, 700.0]
    sweep = lp.Sweep(lp.FrequencyResponse(model, drive=lp.Voltage(v["u"])), "bl", [5.0, 9.0])
    assert sweep.electrical_only()[0]
    runs = sweep.solve(freqs).results
    direct = lp.FrequencyResponse(model, drive=lp.Voltage(v["u"], Re=v["Re"], Bl=9.0, Le=v["Le"], n=v["n"]))
    np.testing.assert_allclose(runs[1].impedance().values, direct.solve(freqs).impedance().values, rtol=1e-9)
    ok, why = lp.Sweep(lp.FrequencyResponse(model, drive=lp.Voltage(v["u"])), "eta", [0.01, 0.02]).electrical_only()
    assert not ok and "not values of the circuit" in why
    model.regions[0].mesh_size = lp.P("bl") * 1e-4                                  # (the structure now uses it too)
    with pytest.raises(lp.errors.ModelError, match="also reach the structure"):
        lp.Sweep(lp.FrequencyResponse(model, drive=lp.Voltage(v["u"])), "bl", [5.0, 9.0], redrive=True).solve(freqs)
    with pytest.raises(lp.errors.ModelError, match="Modes is solved again"):
        lp.Sweep(lp.Modes(model), "bl", [5.0, 9.0], redrive=True).solve(f_max=500.0)
