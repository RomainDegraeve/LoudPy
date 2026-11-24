"""
loudpy.circuit alone (its coils blocked): the parts against their analytic answers — an RC low pass, an LC
network, the amplifier's output resistance, the ideal filter blocks (Butterworth, Linkwitz–Riley sums, EQ, shelving,
delay, gain), the coil models (Semi, Leach, LR-2, Wright) — the checks of a circuit, and its data round trip.
"""
import json

import numpy as np
import pytest

import loudpy as lp
from loudpy.circuit import (Capacitor, Coil, Filter, Inductor, Leach, LR2, Resistor, Semi, Source, Wright,
                            butterworth, delay, gain, high_shelf, linkwitz_riley, low_shelf, peq)

F = np.array([50.0, 400.0, 1000.0, 3000.0, 12000.0])
S = 2j * np.pi * F


def test_passive_parts_and_the_amplifier():
    rc = lp.Circuit(Source("amp", "in", U=2.0), Resistor("R", "in", "o", 100.0), Capacitor("C", "o", "0", 1e-6))
    r = rc.solve(F)
    np.testing.assert_allclose(r.transfer("o").values, 1 / (1 + S * 1e-4), rtol=1e-12)
    np.testing.assert_allclose(r.voltage("o").values, 2.0 / (1 + S * 1e-4), rtol=1e-12)
    np.testing.assert_allclose(r.current("R").values, r.current("C").values, rtol=1e-12)   # the same branch
    np.testing.assert_allclose(r.input_impedance().values, 100.0 + 1 / (S * 1e-6), rtol=1e-12)
    lc = lp.Circuit(Source("amp", "a", U=1.0, R_out=0.5), Inductor("L", "a", "b", 1e-3), Resistor("R", "b", "0", 8.0))
    r = lc.solve(F)
    i = 1.0 / (0.5 + S * 1e-3 + 8.0)                                         # the output resistance in series
    np.testing.assert_allclose(r.current("amp").values, i, rtol=1e-12)
    np.testing.assert_allclose(r.current("L").values, i, rtol=1e-12)
    np.testing.assert_allclose(r.input_impedance().values, S * 1e-3 + 8.0, rtol=1e-12)    # at its terminals


def test_filter_blocks():
    f = np.geomspace(20.0, 20000.0, 31)
    assert abs(butterworth(3, 1000.0).response([1000.0])[0]) == pytest.approx(2 ** -0.5, rel=1e-12)
    assert abs(butterworth(2, 1000.0, "high").response([1000.0])[0]) == pytest.approx(2 ** -0.5, rel=1e-12)
    for order in (2, 4, 8):
        lo, hi = linkwitz_riley(order, 2000.0).response(f), linkwitz_riley(order, 2000.0, "high").response(f)
        assert abs(lo[15 - 15 + np.argmin(abs(f - 2000.0))]) == pytest.approx(0.5, rel=1e-2)
        if order == 4:
            np.testing.assert_allclose(abs(lo + hi), 1.0, rtol=1e-12)          # LR4: flat, in phase
        else:
            np.testing.assert_allclose(abs(lo) ** 2 + abs(hi) ** 2 <= 1.0 + 1e-12, True)
    assert 20 * np.log10(abs(peq(80.0, 6.0, 1.0).response([80.0])[0])) == pytest.approx(6.0, abs=1e-9)
    assert 20 * np.log10(abs(peq(80.0, 6.0, 1.0).response([8000.0])[0])) == pytest.approx(0.0, abs=0.01)
    assert 20 * np.log10(abs(low_shelf(200.0, 4.0).response([2.0])[0])) == pytest.approx(4.0, abs=0.01)
    assert 20 * np.log10(abs(high_shelf(2000.0, -3.0).response([200000.0])[0])) == pytest.approx(-3.0, abs=0.01)
    np.testing.assert_allclose(delay(1e-3).response([250.0]), np.exp(-2j * np.pi * 250.0 * 1e-3), rtol=1e-12)
    assert abs((gain(-6.0) * butterworth(1, 100.0)).response([0.001])[0]) == pytest.approx(10 ** (-6 / 20), rel=1e-6)
    c = lp.Circuit(Source("amp", "in", U=1.0), Filter("lp", "in", "o", linkwitz_riley(4, 2000.0)),
                   Resistor("load", "o", "0", 4.0))                         # an ideal block: whatever its load
    np.testing.assert_allclose(c.solve(F).transfer("o").values, linkwitz_riley(4, 2000.0).response(F), rtol=1e-12)


def test_coil_models():
    for model, z in [(Semi(Re=6.0, Bl=5.0, Le=1e-3, n=0.7), 6.0 + S ** 0.7 * 1e-3),
                     (Leach(Re=6.0, Bl=5.0, K=2e-3, n=0.6), 6.0 + 2e-3 * S ** 0.6),
                     (LR2(Re=6.0, Bl=5.0, Le=3e-4, R2=2.0, L2=1e-3), 6.0 + S * 3e-4 + 2.0 * S * 1e-3 / (2.0 + S * 1e-3)),
                     (Wright(Re=6.0, Bl=5.0, Krm=1e-3, Erm=0.8, Kxm=2e-3, Exm=0.7),
                      6.0 + 1e-3 * (2 * np.pi * F) ** 0.8 + 2e-3j * (2 * np.pi * F) ** 0.7)]:
        c = lp.Circuit(Source("amp", "a", U=1.0), Coil("hp", "a", "0", model))
        r = c.solve(F)                                                      # alone: blocked (no back-EMF)
        np.testing.assert_allclose(r.input_impedance().values, z, rtol=1e-12)
        np.testing.assert_allclose(r.impedance("hp").values, z, rtol=1e-12)


def test_checks_and_round_trip():
    with pytest.raises(lp.errors.ModelError, match="Source"):
        lp.Circuit(Resistor("R", "a", "0", 1.0), Resistor("R2", "a", "0", 2.0)).check()
    with pytest.raises(lp.errors.ModelError, match="joined to one part"):
        lp.Circuit(Source("amp", "a"), Resistor("R", "a", "b", 1.0)).check()
    with pytest.raises(lp.errors.ModelError, match="already named"):
        lp.Circuit(Source("x", "a"), Resistor("x", "a", "0", 1.0))
    with pytest.raises(lp.errors.ModelError, match="model"):
        lp.Circuit(Coil("c", "a", "0", model=6.0))
    coil = Coil("c", "n1", "0", Semi(Re=8.0, Bl=7.0, Le=4e-4))
    with pytest.raises(lp.errors.ModelError, match="F1 and F2 both set the voltage of node 'n1'"):
        lp.Circuit(Source("amp", "in"), Filter("F1", "in", "n1", butterworth(4, 400.0)),
                   Filter("F2", "in", "n1", butterworth(4, 700.0, "high")), coil).check()   # two ideal outputs
    with pytest.raises(lp.errors.ModelError, match="both set the voltage of node 'in'"):
        lp.Circuit(Source("a1", "in"), Source("a2", "in", U=2.0), Coil("c", "in", "0", Semi(8.0, 7.0))).check()
    with pytest.raises(lp.errors.ModelError, match=r"\['x', 'y'\] have no path to the ground"):
        lp.Circuit(Source("amp", "in"), Filter("F", "in", "n1", butterworth(2, 400.0)), coil,
                   Resistor("R1", "x", "y", 1.0), Resistor("R2", "x", "y", 2.0)).check()
    summed = lp.Circuit(Source("amp", "in"), Filter("F1", "in", "l", butterworth(4, 400.0)), Resistor("Ra", "l", "n1", 4.0),
                        Filter("F2", "in", "h", butterworth(4, 700.0, "high")), Resistor("Rb", "h", "n1", 4.0), coil)
    summed.check()                                                    # the two outputs joined through resistors
    c = lp.Circuit(Source("amp", "in", U=2.83 + 0.5j, R_out=0.05),
                   Filter("x", "in", "w", linkwitz_riley(4, 2000.0) * peq(80.0, 3.0, 1.0) * delay(1e-4)),
                   Inductor("L", "w", "c", 2e-4), Capacitor("C", "c", "0", 1e-5),
                   Coil("main", "c", "0", LR2(Re=5.6, Bl=6.1, Le=3e-4, R2=2.0, L2=1e-4), driver="main"))
    back = lp.Circuit.from_dict(json.loads(json.dumps(c.to_dict())))
    assert back.to_dict() == c.to_dict()
    np.testing.assert_allclose(back.solve(F).x, c.solve(F).x, rtol=1e-12)
    assert c.nodes == ["in", "w", "c"] and c.branches == ["amp", "x", "L", "main"] and c.size == 7


def test_spice_netlists():
    """SPICE: written (LTspice reads it) and read back exactly; an LTspice netlist with its scale suffixes, GND,
    a Laplace source; refused elements."""
    from loudpy.circuit import spice_value
    assert [spice_value(t) for t in ("4.7k", "10uF", "1.5mH", "2Meg", "3n", "1e-3", "100p")] == \
        pytest.approx([4.7e3, 10e-6, 1.5e-3, 2e6, 3e-9, 1e-3, 100e-12])
    c = lp.Circuit(Source("amp", "in", U=2.83 + 0.5j, R_out=0.05),
                   Filter("x", "in", "w", linkwitz_riley(4, 2000.0) * peq(80.0, 3.0, 1.0) * delay(2e-5)),
                   Inductor("L1", "w", "c", 2e-4), Capacitor("C1", "c", "0", 1e-5), Resistor("R1", "c", "d", 0.3),
                   Coil("main", "d", "0", LR2(Re=5.6, Bl=6.1, Le=3e-4, R2=2.0, L2=1e-4), driver="main"))
    text = c.to_spice("crossover")
    assert "Laplace=" in text and "Vamp amp_int 0 AC" in text and text.rstrip().endswith(".end")
    back = lp.Circuit.from_spice(text)
    assert back.to_dict() == c.to_dict()
    np.testing.assert_allclose(back.solve(F).x, c.solve(F).x, rtol=1e-12)
    lt = """* C:\\\\crossover.asc
V1 in 0 AC 2.83 0
E1 f 0 in 0 Laplace=1/(1+s/6283.18)
R1 f out 4.7k
C1 out GND 10uF
+ ; a continued line
.ac dec 10 20 20k
.end
"""
    r = lp.Circuit.from_spice(lt).solve(F)
    np.testing.assert_allclose(r.transfer("out").values, 1 / (1 + S / 6283.18) / (1 + S * 4.7e3 * 10e-6), rtol=1e-10)
    with pytest.raises(lp.errors.ModelError, match="does not model"):
        lp.Circuit.from_spice("V1 a 0 AC 1\nD1 a 0 1N4148\nR1 a 0 8\n")


def test_circuit_in_documents_and_scripts():
    """A circuit drive is a node of the document (loudpy.schema): described, checked, built again and written
    as code."""
    from loudpy import schema
    from support import current_api as api
    c = lp.Circuit(Source("amp", "a", U=2.83), Filter("hp", "a", "f", butterworth(2, 100.0, "high")),
                   Coil("vc", "f", "0", Semi(Re=6.0, Bl=5.0, Le=3e-4)))
    node = {"type": "FrequencyResponse", "params": {"drive": schema.describe(c), "freqs": [100.0, 200.0]}}
    assert schema.check(node) == []
    study, kwargs = schema.build(json.loads(json.dumps(node)), model=api.speaker_model(acoustic=True))
    assert isinstance(study.drive, lp.Circuit) and study.drive.to_dict() == c.to_dict()
    text = schema.code(node["params"]["drive"])
    assert text.startswith("lp.Circuit(lp.circuit.Source(name='amp'")
    built = eval(text, {"lp": lp})                                           # the code builds it again
    np.testing.assert_allclose(built.solve(F).x, c.solve(F).x, rtol=1e-12)
