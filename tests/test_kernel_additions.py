"""
Kernel functions that came out of the interface work (usable from scripts): progress callback, preview mesh of a
bare geometry, which items use a parameter, .geo DefineConstant defaults, MaterialLibrary.with_, drive signals,
Model.to_script; and the kernel bugs found then: meshing a model without regions, drivers off a Box's front face.
"""
import dataclasses
import runpy
import shutil
from pathlib import Path

import numpy as np
import pytest

import loudpy as lp
from support import current_api as api

ROOT = Path(__file__).resolve().parents[1]
PARAM_GEO = ROOT / "examples" / "Geometries" / "Plate" / "clamped_plate_param.geo"


def test_mesh_without_regions_says_so(tmp_path):
    model = lp.Model(api.PLATE_GEO)
    with pytest.raises(lp.errors.ModelError, match="no region to mesh"):
        model.mesh(tmp_path / "plate.msh")


def test_preview_mesh_of_a_bare_geometry(tmp_path):
    """Every domain group meshed for display, on a copy: the model keeps no region and no mesh."""
    model = lp.Model(api.PLATE_GEO)
    assert model.geometry_size() == pytest.approx(0.1, rel=0.05)
    mesh = model.preview_mesh(tmp_path / "plate_preview.msh", elements=40)
    assert mesh.names(2) == model.groups("domain") and mesh.n_nodes > 50
    assert not model.regions and model.meshed is None


def test_parameter_uses_and_geo_defaults(tmp_path):
    geo = tmp_path / "plate_param.geo"
    shutil.copy(PARAM_GEO, geo)
    model = lp.Model(geo, parameters=lp.Parameters(a=0.1, h=0.002, unused=1.0))
    assert model.source.geo_defaults == {"a": 0.1, "h": 0.002}
    model.add(lp.Solid("plate", "Paper", mesh_size=lp.P("h")))
    assert model.uses("h") == ["plate", "geometry"] and model.uses("a") == ["geometry"] and model.uses("unused") == []


def test_material_library_with():
    lib = lp.load_materials()
    paper = dataclasses.replace(lib["Paper"], E=4.2e9)
    mine = lib.with_(paper, dataclasses.replace(paper, name="Paper2"), path="mine.json")
    assert mine["Paper"].E == 4.2e9 and "Paper2" in mine and lib["Paper"].E != 4.2e9      # the original unchanged
    assert str(mine.path) == "mine.json" and len(mine) == len(lib) + 1
    with pytest.raises(lp.errors.MaterialError):
        lib.with_("Paper")


def test_signals(tmp_path):
    from scipy.io import wavfile
    s, fs = lp.signals.sine(200.0, periods=5, fs=48000)
    assert fs == 48000.0 and len(s) == 1201 and np.max(np.abs(s)) == pytest.approx(1.0, abs=1e-6)
    m, _ = lp.signals.multitone([50, 120, 330, 700], duration=0.1, fs=48000, amplitude=2.0)
    assert len(m) == 4801 and np.max(np.abs(m)) <= 2.0
    spectrum = np.abs(np.fft.rfft(m[:-1]))
    peaks = np.fft.rfftfreq(len(m) - 1, 1 / 48000)[np.argsort(spectrum)[-4:]]
    assert sorted(np.round(peaks)) == [50, 120, 330, 700]                                 # the four tones
    t = np.arange(800) / 8000.0
    wavfile.write(tmp_path / "x.wav", 8000, (np.sin(2 * np.pi * 300 * t) * 20000).astype(np.int16))
    w, fsw = lp.signals.wav(tmp_path / "x.wav", duration=0.005, amplitude=3.0)
    assert fsw == 8000.0 and len(w) == 41 and np.max(np.abs(w)) <= 3.0
    with pytest.raises(lp.errors.ModelError):
        lp.signals.multitone([])


def test_progress_callback(tmp_path):
    model = api.speaker_model(size=3e-3)
    model.mesh(tmp_path / "spk.msh")
    steps = []
    lp.progress.set_callback(lambda what, i, n: steps.append((what, i, n)))
    try:
        lp.FrequencyResponse(model, drive=lp.Force(1.0)).solve([100.0, 200.0, 300.0])
    finally:
        lp.progress.set_callback(None)
    assert [s for s in steps if s[0] == "frequency"] == [("frequency", 1, 3), ("frequency", 2, 3), ("frequency", 3, 3)]
    lp.progress.step("anything", 1, 2)                                                     # silent without a callback


def test_model_to_script_round_trip(tmp_path):
    geo = tmp_path / "plate_param.geo"
    shutil.copy(PARAM_GEO, geo)
    model = lp.Model(geo, parameters=lp.Parameters(h=0.002))
    model.add(lp.Solid("plate", "Paper", mesh_size=lp.P("h"), damping=lp.Undamped()),
              lp.Clamped("rim"), lp.Clamped("axis", dofs="r"))
    text = model.to_script(tmp_path)
    (tmp_path / "plate_model.py").write_text(text)
    again = runpy.run_path(str(tmp_path / "plate_model.py"))["model"]
    assert repr(again.regions) == repr(model.regions) and repr(again.boundaries) == repr(model.boundaries)
    assert dict(again.parameters) == {"h": 0.002} and again.source.path == geo
    with pytest.raises(lp.errors.ModelError, match="builder"):
        lp.Model(lambda occ: {}).to_script()


def _profile_driver():
    rear = [np.array([[0.050, api.Z_B], [0.010, api.Z_B - 0.02]]),
            np.array([[0.010, api.Z_B - 0.02], [0.0, api.Z_B - 0.02]])]
    front = [np.array([[0.053, api.Z_F], [0.0, api.Z_F - 0.015]])]
    return lp.Driver(list(range(3)), rear, front, coil_nodes=[0], name="profiles")


def test_box_refuses_drivers_off_its_front_face():
    hp = _profile_driver()
    port = lp.cabinets.Port((0.05, 0.05), 0.02, 0.06)
    box = lp.cabinets.Box(0.26, 0.2, 0.15, wall=0.015, ports=[port])
    box.for_drivers([hp.place("main", at=(0.15, 0.12))])                                  # fits
    with pytest.raises(lp.errors.ModelError, match="not on the front face"):
        box.for_drivers([hp.place("main", at=(0.24, 0.12))])
    with pytest.raises(lp.errors.ModelError, match="overlap"):
        box.for_drivers([hp.place("a", at=(0.15, 0.12)), hp.place("b", at=(0.17, 0.12))])
    with pytest.raises(lp.errors.ModelError, match="port"):
        box.for_drivers([hp.place("main", at=(0.09, 0.09))])
    with pytest.raises(lp.errors.ModelError, match="not on the front face"):
        lp.Scene.vented_box(box, hp, at=(0.30, 0.12))                                     # through lp.Scene too
