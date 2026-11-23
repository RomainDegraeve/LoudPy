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
    c = lp.Circuit(Source("amp", "in", U=2.83 + 0.5j, R_out=0.05),
                   Filter("x", "in", "w", linkwitz_riley(4, 2000.0) * peq(80.0, 3.0, 1.0) * delay(1e-4)),
                   Inductor("L", "w", "c", 2e-4), Capacitor("C", "c", "0", 1e-5),
                   Coil("main", "c", "0", LR2(Re=5.6, Bl=6.1, Le=3e-4, R2=2.0, L2=1e-4), driver="main"))
    back = lp.Circuit.from_dict(json.loads(json.dumps(c.to_dict())))
    assert back.to_dict() == c.to_dict()
    np.testing.assert_allclose(back.solve(F).x, c.solve(F).x, rtol=1e-12)
    assert c.nodes == ["in", "w", "c"] and c.branches == ["amp", "x", "L", "main"] and c.size == 7
