"""
Phase F2: parameters (lp.Parameters, lp.P), builders taking the parameters, parametric sweeps (lp.Sweep).
Gate: a sweep equals the separate runs.
"""
import numpy as np
import pytest

import loudpy as lp
from loudpy.errors import ModelError
from loudpy.model.parameters import resolve
from support import current_api as api

STEEL = lp.Material("Steel", rho=7850.0, E=2.1e11, nu=0.3)


def test_references_and_values():
    p = lp.Parameters(h=2e-3, E=2e11, a=5.0)
    assert (0.5 * lp.P("h")).value(p) == 1e-3 and (lp.P("h") / 2 + 1e-3).value(p) == pytest.approx(2e-3)
    assert p.with_(h=1e-3)["h"] == 1e-3 and p["h"] == 2e-3                 # immutable
    with pytest.raises(ModelError, match="unknown parameters"):
        p.with_(t=1.0)
    with pytest.raises(ModelError, match="finite numbers"):
        lp.Parameters(h="thin")
    solid = lp.Solid("plate", STEEL, E=lp.P("E"), mesh_size=0.5 * lp.P("h"), damping=lp.Rayleigh(alpha=lp.P("a"), beta=0.0))
    r = resolve(solid, p)
    assert (r.E, r.mesh_size, r.damping.alpha) == (2e11, 1e-3, 5.0)
    assert solid.E == lp.P("E")                                              # the item itself is unchanged
    with pytest.raises(ModelError, match="nu = 0.7"):
        resolve(lp.Solid("plate", STEEL, nu=lp.P("nu")), lp.Parameters(nu=0.7))   # checked once resolved


def test_check_reports_unknown_parameters():
    model = lp.Model(api.PLATE_GEO, parameters=lp.Parameters(h=1e-3))
    model.add(lp.Solid("plate", STEEL, E=lp.P("E_plate"), mesh_size=lp.P("h")), lp.Clamped("interface_constrained_rz"))
    with pytest.raises(ModelError, match="E_plate"):
        model.check()
    assert "Parameters: h = 0.001" in model.tree()


def plate_model(**overrides):
    params = lp.Parameters(E=2.1e11, h=2e-3)
    model = lp.Model(api.PLATE_GEO, parameters=params)
    model.add(lp.Solid("plate", STEEL, damping=lp.Undamped(), E=overrides.get("E", lp.P("E")),
                       mesh_size=overrides.get("h", lp.P("h"))),
              lp.Clamped("interface_constrained_rz"), lp.Clamped("interface_constrained_r", dofs="r"))
    return model


def test_sweep_equals_separate_runs(tmp_path):
    """Young's modulus swept: every run equals the model built with that value; the model is unchanged."""
    model = plate_model()
    model.mesh(tmp_path / "plate.msh")
    values = [1.9e11, 2.1e11]
    sweep = lp.Sweep(lp.Modes(model), "E", values).solve(f_max=3000.0, f_min=10.0, n_per_shift=10)
    assert len(sweep) == 2 and np.array_equal(sweep.values, values)
    for v, res in zip(values, sweep):
        alone = plate_model(E=v)
        alone.mesh(tmp_path / f"alone_{v:g}.msh")
        ref = lp.Modes(alone).solve(f_max=3000.0, f_min=10.0, n_per_shift=10)
        np.testing.assert_array_equal(res.frequencies, ref.frequencies)
    f = sweep.stack(lambda r: r.frequencies[:2])
    np.testing.assert_allclose(f[0] / f[1], np.sqrt(1.9 / 2.1), rtol=1e-9)          # f ∝ √E
    assert model.parameters["E"] == 2.1e11 and model["plate"].E == lp.P("E")
    assert not (tmp_path / "plate_E_1.9e+11.msh").exists()                       # E changes no mesh input: model mesh


def test_geometric_sweep_with_a_builder(tmp_path):
    """A builder(occ, p) draws the plate of thickness p["t"]: the sweep remeshes and equals the separate runs."""
    def disc(occ, p):
        return {"plate": occ.addRectangle(0, 0, 0, 0.1, p["t"])}

    def model_for(t):
        m = lp.Model(disc, parameters=lp.Parameters(t=t))
        m.add(lp.Solid("plate", STEEL, damping=lp.Undamped(), mesh_size=1e-3),
              lp.Clamped("rim", on=lp.Box(r=0.1)), lp.Clamped("axis", on=lp.Box(r=0), dofs="r"))
        return m

    model = model_for(2e-3)
    model.mesh(tmp_path / "disc.msh")
    a = model.source.files()
    assert a["parameters"] == {"t": 2e-3} and model.with_parameters(t=3e-3).source.files() != a
    sweep = lp.Sweep(lp.Modes(model), "t", [2e-3, 3e-3]).solve(f_max=4000.0, f_min=10.0, n_per_shift=10)
    for t, res in zip([2e-3, 3e-3], sweep):
        alone = model_for(t)
        alone.mesh(tmp_path / f"alone_{t:g}.msh")
        np.testing.assert_array_equal(res.frequencies, lp.Modes(alone).solve(f_max=4000.0, f_min=10.0,
                                                                           n_per_shift=10).frequencies)
    f0 = [r.frequencies[0] for r in sweep]
    assert 1.3 < f0[1] / f0[0] < 1.55                       # bending: f ∝ t (shear lowers it a little)


def test_sweep_errors():
    model = plate_model()
    with pytest.raises(ModelError, match="no parameter"):
        lp.Sweep(lp.Modes(model), "thickness", [1.0])
    with pytest.raises(Exception, match="mesh the model first"):
        lp.Sweep(lp.Modes(model), "E", [2e11]).solve(f_max=1000.0)
