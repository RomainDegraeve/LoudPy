"""Output control (lp.Output): fields kept, every n-th entry, full-rate probes; the same numbers as the full results."""
import numpy as np
import pytest

import loudpy as lp
from support import current_api as api

TIP, RIM = (0.0, 0.0), (0.03, 0.002)


def _speaker(mesh_dir):
    return api._meshed(api.speaker_model(), mesh_dir, api._speaker_tag(1e-3, False, False))


def test_transient_output(mesh_dir, tmp_path):
    model = _speaker(mesh_dir)
    n, fs = 60, 48_000.0
    sig = np.sin(2 * np.pi * 400.0 * np.arange(n + 1) / fs)
    full = lp.Transient(model, drive=lp.Force(sig, fs=fs)).solve(n_steps=n)
    out = lp.Output(fields=["u"], every=7, probes={"tip": TIP, "rim": RIM})
    study = lp.Transient(model, drive=lp.Force(sig, fs=fs))
    res = study.solve(n_steps=n, output=out)
    keep = np.arange(0, len(full.times), 7)
    np.testing.assert_array_equal(res.times, full.times[keep])
    assert res.names == ["u"]
    np.testing.assert_array_equal(res.blocks[0].fields["u"], full.blocks[0].fields["u"][keep])
    assert res.probe_names == ["tip", "rim"]
    for name, pt in (("tip", TIP), ("rim", RIM)):
        for f in ("u", "v", "a"):                                          # every field, at every step
            np.testing.assert_allclose(res.probe_signal(name, f, "z").values, full.probe(f, pt, "z").values,
                                       rtol=0, atol=0)
    s = res.probe_signal("tip", "u")
    assert s.dims == ("t", "component") and len(s.coords["t"]) == len(full.times)
    assert res.meta["output"]["every"] == 7
    small, big = study.save(tmp_path / "small.h5"), tmp_path / "big.h5"
    full.save(big)
    assert (tmp_path / "small.h5").stat().st_size < 0.4 * big.stat().st_size
    back = lp.load(small)
    np.testing.assert_array_equal(back.probe_signal("rim", "a", "r").values, res.probe_signal("rim", "a", "r").values)
    with pytest.raises(lp.errors.ResultsError, match="no probe 'cone'"):
        back.probe_signal("cone")


def test_frequency_output(mesh_dir):
    model = _speaker(mesh_dir)
    freqs = np.array([100.0, 200.0, 400.0, 800.0, 1600.0])
    full = lp.FrequencyResponse(model, drive=lp.Force(1.0)).solve(freqs)
    res = lp.FrequencyResponse(model, drive=lp.Force(1.0)).solve(freqs, output=lp.Output(every=2, probes={"tip": TIP}))
    np.testing.assert_array_equal(res.frequencies, freqs[::2])
    np.testing.assert_allclose(res.probe_signal("tip", "u", "z").values, full.displacement(TIP, "z").values, rtol=1e-6)   # two solves: the solver's round-off
    assert np.iscomplexobj(res.probe_signal("tip", "u", "z").values)          # harmonic: complex amplitudes


def test_output_checks(mesh_dir):
    with pytest.raises(lp.errors.ModelError, match="positive integer"):
        lp.Output(every=0)
    with pytest.raises(lp.errors.ModelError, match="must be a point"):
        lp.Output(probes={"tip": (0.0,)})
    model = _speaker(mesh_dir)
    with pytest.raises(lp.errors.ModelError, match="no field \\['p'\\]"):
        lp.FrequencyResponse(model, drive=lp.Force(1.0)).solve([100.0], output=lp.Output(fields=["p"]))
