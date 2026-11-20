"""
Phase G1: result format 3 — round trip of every result kind, the model description (with the geometry files
and the physical data of every mesh group) and the study settings in the file, parametric sweeps in one file,
format 2 still read.
"""
import json

import h5py
import numpy as np
import pytest

import loudpy as lp
from loudpy.results.schema import _json
from support import current_api as api

STEEL = lp.Material("Steel", rho=7850.0, E=2.1e11, nu=0.3)


def normal(d):
    """A dict as it comes back from JSON (tuples → lists, NumPy scalars → numbers)."""
    return json.loads(_json(d))


def assert_same(a, b):
    """Two results equal entry for entry: blocks, meshes, fields (bit for bit), extra, metadata, description."""
    assert type(a) is type(b) and a.kind == b.kind and len(a.blocks) == len(b.blocks)
    for x, y in zip(a.blocks, b.blocks):
        assert x.label == y.label and normal(x.meta) == y.meta
        np.testing.assert_array_equal(x.values, y.values)
        assert list(x.meshes) == list(y.meshes)
        for phys in x.meshes:
            mx, my = x.meshes[phys], y.meshes[phys]
            np.testing.assert_array_equal(mx.node_tags, my.node_tags)
            np.testing.assert_array_equal(mx.coords, my.coords)
            assert list(mx.groups) == list(my.groups)
            for g in mx.groups:
                assert (mx.groups[g].dim, mx.groups[g].cell_type) == (my.groups[g].dim, my.groups[g].cell_type)
                np.testing.assert_array_equal(mx.groups[g].cells, my.groups[g].cells)
        assert list(x.fields) == list(y.fields)
        for k in x.fields:
            np.testing.assert_array_equal(np.asarray(x.fields[k]), np.asarray(y.fields[k]))
        assert sorted(x.extra) == sorted(y.extra)
        for k in x.extra:
            np.testing.assert_array_equal(np.asarray(x.extra[k]), np.asarray(y.extra[k]))
    assert normal(a.meta) == b.meta
    assert sorted(a.boundaries) == sorted(b.boundaries)
    for n in a.boundaries:
        for k in a.boundaries[n]:
            if len(a.boundaries[n][k]):
                np.testing.assert_array_equal(a.boundaries[n][k], b.boundaries[n][k])
    assert normal(a.model.data) == b.model.data and a.model.files == b.model.files
    study_a = dict(a.study)
    arrays_a, arrays_b = study_a.pop("arrays", {}), dict(b.study).pop("arrays", {})
    assert normal(study_a) == {k: v for k, v in b.study.items() if k != "arrays"}
    assert sorted(arrays_a) == sorted(arrays_b)
    for k in arrays_a:
        np.testing.assert_array_equal(arrays_a[k], arrays_b[k])


def test_round_trip_modes_frequency_time(tmp_path, mesh_dir):
    plate = api.plate_problem(mesh_dir)
    modes = lp.Modes(plate).solve(f_max=4000.0, f_min=10.0, n_per_shift=10)
    speaker = api.speaker_problem(mesh_dir, acoustic=True)
    drive = lp.Voltage(2.83, Re=6.0, Bl=7.0, Le=1e-4, n=0.7)
    freq = lp.FrequencyResponse(speaker, drive=drive).solve([100.0, 500.0])
    meca = api.speaker_problem(mesh_dir)
    signal = np.sin(2 * np.pi * 200.0 * np.arange(100) / 48_000.0)
    time = lp.Transient(meca, drive=lp.Force(signal, fs=48_000.0)).solve(n_steps=4)
    for name, res in [("modes", modes), ("freq", freq), ("time", time)]:
        back = lp.load(res.save(tmp_path / f"{name}.h5"))
        assert_same(res, back)
        with h5py.File(tmp_path / f"{name}.h5") as f:
            assert f.attrs["loudpy_format"] == "3.0" and "model" in f and "study" in f
    assert freq.study["drive"] == {"type": "Voltage", "amplitude": 2.83, "Re": 6.0, "Bl": 7.0, "Le": 1e-4, "n": 0.7}
    np.testing.assert_array_equal(lp.load(tmp_path / "time.h5").study["arrays"]["drive_amplitude"], signal)


def test_model_description(tmp_path, mesh_dir):
    """The file says what produced it: the tree, the items, the resolved physical data of every mesh group, the
    geometry files (written back identical)."""
    speaker = api.speaker_problem(mesh_dir, acoustic=True)
    res = lp.load(lp.Modes(api.speaker_problem(mesh_dir)).solve(f_max=500.0, f_min=20.0, n_per_shift=10)
                  .save(tmp_path / "m.h5"))
    d = res.model
    assert d["kind"] == "model" and "Solid 'membranne'" in d.tree and d["regions"][0]["material"] == "Paper"
    assert d.groups["submeca_spider"] == {"region": "spider", "role": "solid", "rho": 600.0, "E": 4.5e7, "nu": 0.35,
                                          "eta": 0.0, "alpha_ray": 39.9839, "beta_ray": 1.0128e-05}
    written = d.extract(tmp_path / "geometry")
    assert sorted(p.name for p in written) == ["HPNEW-Sketch.brep", "HPNEW-Sketch.geo"]
    from loudpy.geometry.sources import GeometrySource
    assert GeometrySource(tmp_path / "geometry" / "HPNEW-Sketch.geo").files() == speaker.source.files()
    # the coupled model: air and PML groups with ρ, c and the PML layer geometry
    freq = lp.FrequencyResponse(speaker, drive=lp.Force(1.0)).solve([200.0])
    groups = freq.model.groups
    assert groups["subacou_front"]["role"] == "air" and groups["subacou_front"]["c"] == lp.materials["Air"].c
    pml = [g for g in groups.values() if g["role"] == "pml"]
    assert len(pml) == 6 and all("layer" in g and g["omega_pml"] == pytest.approx(2 * np.pi * 1000.0) for g in pml)

    def disc(occ):
        return {"plate": occ.addRectangle(0, 0, 0, 0.1, 0.002)}
    model = lp.Model(disc).add(lp.Solid("plate", STEEL, damping=lp.Undamped(), mesh_size=0.004),
                               lp.Clamped("rim", on=lp.Box(r=0.1)))
    model.mesh(tmp_path / "disc.msh")
    b = lp.Modes(model).solve(f_max=5000.0, f_min=10.0, n_per_shift=6)
    assert list(b.model.files) == ["disc.py"] and b"addRectangle" in b.model.files["disc.py"]
    assert b.model["geometry"]["builder"] is True


def test_sweep_in_one_file(tmp_path):
    params = lp.Parameters(E=2.1e11)
    model = lp.Model(api.PLATE_GEO, parameters=params).add(
        lp.Solid("plate", STEEL, damping=lp.Undamped(), E=lp.P("E"), mesh_size=0.002),
        lp.Clamped("interface_constrained_rz"), lp.Clamped("interface_constrained_r", dofs="r"))
    model.mesh(tmp_path / "plate.msh")
    sweep = lp.Sweep(lp.Modes(model), "E", [1.9e11, 2.1e11]).solve(f_max=3000.0, f_min=10.0, n_per_shift=10)
    path = sweep.save(tmp_path / "sweep.h5")
    back = lp.load(path)
    assert isinstance(back, lp.results.SweepResults) and back.parameter == "E"
    np.testing.assert_array_equal(back.values, [1.9e11, 2.1e11])
    for v, r, b in zip(back.values, sweep, back):
        assert_same(r, b)
        assert b.model.parameters == {"E": v} and b.model.groups["submeca_plate"]["E"] == v


def test_format_2_still_read(tmp_path, mesh_dir):
    res = lp.Modes(api.plate_problem(mesh_dir)).solve(f_max=4000.0, f_min=10.0, n_per_shift=10)
    path = res.save(tmp_path / "v2.h5")
    with h5py.File(path, "a") as f:                               # what a format-2 file looked like
        del f["model"], f["study"]
        f.attrs["loudpy_format"] = "2.0"
    back = lp.load(path)
    assert back.model is None and back.study == {}
    np.testing.assert_array_equal(back.frequencies, res.frequencies)
