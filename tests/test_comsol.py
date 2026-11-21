"""COMSOL comparison harness (phase K): the reader and the comparison on synthetic exports, and the reference cases
of tests/data/comsol (skipped until their COMSOL exports are there)."""
from pathlib import Path

import numpy as np
import pytest

import loudpy as lp
from support import current_api as api

COMSOL = Path(__file__).parent / "data" / "comsol"

EXPORT = """% Model:              speaker.mph
% Version:            COMSOL 6.2.0.290
% Date:               Nov 21 2026, 10:12
% Table:              Point Evaluation 1
% freq (Hz)           Pressure (Pa)              abs(p) (Pa)
100                   0.010+0.002i               0.0101980390
200                   0.020-0.001i               0.0200249844
400                   -0.004+0.03i               0.0302654919
"""


def test_read_comsol_table(tmp_path):
    path = tmp_path / "export.txt"
    path.write_text(EXPORT)
    t = lp.read_comsol(path)
    assert t.columns == ["freq (Hz)", "Pressure (Pa)", "abs(p) (Pa)"]
    assert t.units["abs(p) (Pa)"] == "Pa" and t.meta["version"].startswith("COMSOL 6.2")
    np.testing.assert_allclose(t.column("Pressure"), [0.010 + 0.002j, 0.020 - 0.001j, -0.004 + 0.03j])
    np.testing.assert_allclose(t.column("abs(p)"), np.abs(t.column("Pressure")), rtol=1e-8)
    assert np.isrealobj(t.column("freq")) and np.iscomplexobj(t.column("Pressure"))
    d = t.data("freq", "Pressure", quantity="p")
    assert d.dims == ("f",) and d.quantity.name == "p"
    with pytest.raises(lp.errors.ResultsError, match="not found"):
        t.column("velocity")


def test_compare(tmp_path, verify):
    f = np.geomspace(50, 5000, 300)
    exact = lp.Data(1 / (1 + 1j * f / 800), {"f": f}, "p")
    ref = lp.Data(exact.values[::7] * (1 + 0.01 * np.sin(f[::7] / 300)), {"f": f[::7]}, "p")   # coarser, 1 % off
    c = lp.compare(exact, ref, rtol=0.02, label="synthetic")
    assert c.passed and 0.003 < c.max_error < 0.012 and "within" in str(c)
    assert lp.compare(exact, ref, metric="db").max_error < 0.1
    assert not lp.compare(exact, ref * 1.1, rtol=0.02).passed
    verify("COMSOL harness", "Comparison on a synthetic export (1 % ripple, coarser grid)", "max relative error",
           c.max_error, bound=0.012, reference="the ripple put into the reference")
    with pytest.raises(lp.errors.ResultsError, match="coordinates differ"):
        lp.compare(exact, lp.Data(ref.values, {"t": f[::7]}, "p"))


# ── reference cases: run when their COMSOL export is in tests/data/comsol ─────────────────────────
def _plate_modes(mesh_dir, table):
    model = api.plate_problem(mesh_dir)
    modes = lp.Modes(model).solve(f_min=10.0, f_max=2500.0)
    n = min(len(modes.frequencies), len(table.column("eigenfrequency")))
    ours = lp.Data(modes.frequencies[:n], {"mode": np.arange(1, n + 1.0)}, "f")
    ref = lp.Data(np.real(table.column("eigenfrequency"))[:n], {"mode": np.arange(1, n + 1.0)}, "f")
    return ours, ref, 1e-3


CASES = {"clamped_plate_modes.txt": _plate_modes}


@pytest.mark.parametrize("name", sorted(CASES))
def test_comsol_case(name, mesh_dir, verify):
    path = COMSOL / name
    if not path.exists():
        pytest.skip(f"no COMSOL export {name} yet (see tests/data/comsol/README.md)")
    ours, ref, rtol = CASES[name](mesh_dir, lp.read_comsol(path))
    c = lp.compare(ours, ref, rtol=rtol, metric="pointwise", label=name)
    verify("COMSOL comparison", name, "max relative difference", c.max_error, bound=rtol, reference="COMSOL export")
    assert c.passed, str(c)
