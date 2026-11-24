"""
A plot as plain data (lp.plot.PlotSpec): resolved on results by slicing named dimensions, every dimension not plotted
a control (slider / choice), the same numbers as the results' own evaluation; JSON round trip; a harmonic field's
phase as a control; stepping a control as a player does.
"""
import json

import numpy as np
import pytest

import loudpy as lp
from loudpy.plot.spec import PHASES, PlotSpec, quantities
from support import current_api as api

FREQS = [200.0, 500.0, 1000.0]


@pytest.fixture(scope="module")
def fsi(mesh_dir):
    model = api._meshed(api.speaker_model(acoustic=True), mesh_dir, api._speaker_tag(1e-3, True, False))
    return lp.FrequencyResponse(model, drive=lp.Force(1.0)).solve(FREQS)


def test_quantities_offered(fsi):
    q = quantities(fsi)
    assert {"spl", "pressure", "displacement", "velocity", "field"} <= set(q) and "impedance" not in q


def test_a_curve_a_map_a_polar_and_their_controls(fsi):
    arc = {"kind": "arc", "radius": 0.1, "from": 0, "to": 90, "step": 10, "center": [0.0, 0.03]}
    full = fsi.spl(at=lp.Arc(0.1, np.arange(0, 91, 10.0), (0.0, 0.03)))
    out = PlotSpec(quantity="spl", where=arc).resolve(fsi)                       # default: over f, the angle a control
    assert out.view == "curve" and out.data.dims == ("f",) and [c.dim for c in out.controls] == ["angle"]
    np.testing.assert_array_equal(out.data.values, full.isel(angle=0).values)
    polar = PlotSpec(quantity="spl", where=arc, axes=["angle"], view="polar", at={"f": 480.0}).resolve(fsi)
    (c,) = polar.controls
    assert polar.view == "polar" and c.dim == "f" and c.index == 1 and c.text() == "500 Hz" and c.numeric
    np.testing.assert_array_equal(polar.data.values, full.isel(f=1).values)
    m = PlotSpec(quantity="spl", where=arc, view="map").resolve(fsi)
    assert m.view == "map" and m.data.dims == ("f", "angle") and m.controls == []
    line = PlotSpec(quantity="displacement", where={"kind": "line", "start": [0.0, 0.0], "end": [0.04, 0.0], "n": 5},
                    component="z", axes=["s"], at={"f": {"index": 2}}).resolve(fsi)
    np.testing.assert_array_equal(line.data.values, fsi.displacement(
        at=lp.Line((0.0, 0.0), (0.04, 0.0), 5), component="z").isel(f=2).values)
    assert line.control("f").index == 2


def test_spec_round_trip_and_step(fsi):
    spec = PlotSpec(quantity="spl", where={"kind": "point", "point": [0.0, 0.2]}, axes=["f"], name="on axis")
    back = PlotSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert back == spec and spec.to_dict() == {"quantity": "spl", "where": {"kind": "point", "point": [0.0, 0.2]},
                                               "axes": ["f"], "name": "on axis"}
    with pytest.raises(lp.errors.ResultsError, match="unknown keys"):
        PlotSpec.from_dict({"quantity": "spl", "colour": "red"})
    polar = PlotSpec(quantity="spl", where={"kind": "arc", "radius": 0.1, "center": [0.0, 0.03]}, axes=["angle"],
                     view="polar")
    nxt = polar.step(fsi, "f", +1)
    assert nxt.at["f"] == {"index": 1} and polar.at == {}
    assert polar.step(fsi, "f", -1).at["f"] == {"index": 2}                      # a player wraps round
    with pytest.raises(lp.errors.ResultsError, match="no control"):
        polar.step(fsi, "t")


def test_a_harmonic_field_at_a_phase(fsi):
    out = PlotSpec(quantity="field", field="u", style={"show": "real"}, at={"f": 1000.0, "phase": 90.0}).resolve(fsi)
    assert [c.dim for c in out.controls] == ["f", "phase"] and out.control("phase").values == list(PHASES)
    u = fsi.field("u", index=2)
    np.testing.assert_allclose(out.data.values, np.real(u.values * 1j), atol=1e-30)
    plain = PlotSpec(quantity="field", field="p", at={"f": {"index": 0}}).resolve(fsi)
    assert [c.dim for c in plain.controls] == ["f"] and plain.view == "field"
    np.testing.assert_array_equal(plain.data.values, fsi.field("p", index=0).values)


def test_refusals(fsi):
    with pytest.raises(lp.errors.ResultsError, match="no quantity"):
        PlotSpec(quantity="loudness").resolve(fsi)
    with pytest.raises(lp.errors.ResultsError, match="do not offer 'impedance'"):
        PlotSpec(quantity="impedance").resolve(fsi)
    with pytest.raises(lp.errors.ResultsError, match="where is it evaluated"):
        PlotSpec(quantity="spl").resolve(fsi)
    with pytest.raises(lp.errors.ResultsError, match="axes"):
        PlotSpec(quantity="spl", where={"kind": "point", "point": [0, 0.2]}, axes=["angle"]).resolve(fsi)


def test_plots_kept_in_the_document(tmp_path, mesh_dir):
    model = api._meshed(api.speaker_model(acoustic=True), mesh_dir, api._speaker_tag(1e-3, True, False))
    spec = PlotSpec(quantity="spl", where={"kind": "point", "point": [0.0, 0.2]}, name="on axis").to_dict()
    doc = lp.Document.of(model, [], name="spk", folder=tmp_path)
    assert doc.plots == [] and "plots" not in doc.to_dict()
    ws = lp.Workspace(doc, tmp_path)
    ws.insert("plots", {**spec, "result": "fr_1"})
    assert doc.get("plots/0/name") == "on axis" and doc.check() == []
    back = lp.Document.load(doc.save(tmp_path / "spk.loudpy.json"))
    assert back.plots == [{**spec, "result": "fr_1"}]
    ws.undo()
    assert doc.plots == []
    doc.plots.append({"quantity": "spl", "colour": "red"})
    assert any("unknown keys" in m for m in doc.check()) and doc.problems()[0][0] == "plots/0"


def test_any_control_played_into_an_animation(fsi, tmp_path):
    polar = PlotSpec(quantity="spl", where={"kind": "arc", "radius": 0.1, "to": 90, "step": 10,
                                            "center": [0.0, 0.03]}, axes=["angle"], view="polar")
    gif = polar.animate(fsi, "f", tmp_path / "polar.gif", fps=4)
    from PIL import Image
    with Image.open(gif) as im:
        assert im.n_frames == len(FREQS)
    field = PlotSpec(quantity="field", field="u", style={"show": "real"}, at={"f": 500.0})
    mp4 = field.animate(fsi, "phase", tmp_path / "phase.mp4", fps=12)
    assert mp4.stat().st_size > 1000
    with pytest.raises(lp.errors.ResultsError, match="no control 'run'"):
        polar.animate(fsi, "run", tmp_path / "x.gif")
