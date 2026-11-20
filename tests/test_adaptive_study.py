"""
Phase H3: lp.Adaptive (a study on meshes adapted to its own error estimate) and lp.convergence (a study on
uniformly refined meshes: observed orders, Richardson extrapolation). Gate H: the adaptive loop beats uniform
refinement on a model (clamped plate, eigenfrequencies), expected orders reported.
"""
import contextlib
import io
import json

import numpy as np
import pytest

import loudpy as lp
from loudpy.errors import ModelError, StateError
from loudpy.study.adaptive import estimate_errors
from support import current_api as api


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


@pytest.fixture(scope="module")
def plate(tmp_path_factory):
    m = api.plate_model()
    m.mesh(tmp_path_factory.mktemp("plate") / "plate.msh")
    return m


def test_adaptive_beats_uniform_on_the_plate(plate):
    ad = lp.Adaptive(lp.Modes(plate), target=0.008, max_iterations=4)
    res = quiet(ad.solve, f_max=4000)
    est = [h["estimate"] for h in ad.history]
    assert ad.converged and np.all(np.diff(est) < 0), ad
    assert ad.model is not plate and plate._mesh_path.name == "plate.msh"          # the user's model unchanged
    uni = quiet(lp.convergence, lp.Modes(plate), factors=(0.35, 0.18), f_max=4000)
    e_uni = estimate_errors(uni.results[0])["estimate"]
    assert ad.history[-1]["n_dofs"] < uni.n_dofs[0] and est[-1] < 0.5 * e_uni, (ad, uni.n_dofs, e_uni)
    f_fine = uni.values[1, 0]                                                       # 15 562 DOFs
    assert abs(res.values[0] - f_fine) < 0.5 * abs(uni.values[0, 0] - f_fine)
    meta = json.loads(res.meta["adaptive"])
    assert meta["target"] == 0.008 and len(meta["history"]) == len(ad.history)
    assert "converged" in repr(ad)


def test_convergence_of_the_plate_modes(plate):
    tab = quiet(lp.convergence, lp.Modes(plate), factors=(1, 0.7, 0.5, 0.35), f_max=4000)
    assert tab.values.shape == (4, 2) and tab.local_orders.shape == (2, 2)
    assert np.all(np.diff(tab.n_dofs) > 0) and np.all(np.diff(tab.values[:, 0]) < 0)    # from above, monotone
    assert np.all((tab.order > 1) & (tab.order < 4.5)), tab                      # 2p = 4 smooth; clamped corner
    finest = tab.values[-1]
    assert np.all(tab.errors[-1] < 5e-4) and np.all(np.abs(tab.extrapolated - finest) / finest < 3e-4)
    assert tab.expected == {"eigenfrequency": 4, "energy norm": 2, "L2 norm / point values": 3}
    assert "order (fit)" in repr(tab) and "expected orders" in repr(tab)
    custom = quiet(lp.convergence, lp.Modes(plate), factors=(1, 0.7), quantity=lambda r: r.values[0], f_max=4000)
    assert custom.values.shape == (2, 1) and np.isnan(custom.order).all()          # two meshes: no fit


def test_adaptive_refusals(plate, tmp_path):
    with pytest.raises(NotImplementedError):
        lp.Adaptive(lp.Transient(plate, drive=lp.Force(np.zeros(8), fs=48_000.0)))
    with pytest.raises(ModelError):
        lp.Adaptive(lp.Modes(plate), target=2)
    with pytest.raises(ModelError):
        lp.Adaptive(lp.FrequencyResponse(plate)).solve([100.0], remesh=lp.PerWavelength(bands=[20]))
    with pytest.raises(StateError):
        lp.Adaptive(lp.Modes(api.plate_model()))
    with pytest.raises(ModelError):
        lp.convergence(lp.Modes(plate), factors=(0.5, 1))
    with pytest.raises(ModelError):
        lp.convergence(lp.FrequencyResponse(plate), factors=(1, 0.5), freqs=[100.0])


@pytest.mark.slow
def test_adaptive_coupled_speaker(tmp_path):
    """Structure + air (+ PML): both physics estimated, both decrease; PML cells are not estimated; the result
    file keeps the adaptation history."""
    m = api.speaker_model(size=1e-3, acoustic=True)
    m.mesh(tmp_path / "spk.msh")
    ad = lp.Adaptive(lp.FrequencyResponse(m), target=0.02, max_iterations=1)
    res = quiet(ad.solve, [200.0, 1000.0])
    h0, h1 = ad.history
    assert h1["estimate_solid"] < 0.6 * h0["estimate_solid"] and h1["estimate_air"] < 0.6 * h0["estimate_air"]
    assert h1["n_dofs"] > h0["n_dofs"] and (tmp_path / "spk_adapt1.pos").exists()
    groups = res.model.groups
    air = res.blocks[0].mesh_of("p")
    assert any(groups[g]["role"] == "pml" for g in air.groups if g in groups)
    n_air = sum(len(g) for n, g in air.groups.items() if g.dim == 2 and groups.get(n, {}).get("role") == "air")
    e = estimate_errors(res)
    assert sum(len(w) for ph, _, w in e["cells"] if ph == "air") == n_air                 # PML cells left out
    res.save(tmp_path / "a.h5")
    assert json.loads(lp.load(tmp_path / "a.h5").meta["adaptive"])["history"][1]["n_dofs"] == h1["n_dofs"]
