"""
Features crossed with studies, where bugs hide: a feature tested alone can break in combination (the Thiele–Small
fit failed on remeshed results, a silent node lookup gave a wrong coil velocity on them).

  frequency response   remeshed (PerWavelength bands) × drive (force, voltage) × output control (all fields,
                       probes, p only every 2nd) × storage (live, saved and read back), checked on the boundary
                       integrals (volume velocity, radiated power), Thiele–Small, probes, a derived field
  modes                output every 2nd × storage, several mode sets
  transient            output probes + every 3rd × storage
  sweep                a mesh-size parameter × remeshed bands × voltage: Thiele–Small per run, one file

and the strict node lookup (an unknown node tag raises, it is never mapped to a neighbour).
"""
import numpy as np
import pytest

import loudpy as lp
from loudpy.errors import MeshError, ResultsError
from loudpy.geometry.mesh import Mesh, MeshGroup
from support import current_api as api

FREQS = np.geomspace(40.0, 1500.0, 10)
FRONT = "interface_acou_meca_front"          # the test geometry keeps the original names (tests/data/geometries)
TIP = (0.0, 0.0128)                          # a node on the axis (dust cap)
DRIVES = {"force": lp.Force(1.0), "voltage": lp.Voltage(2.83, Re=8.0, Le=4e-4, Bl=7.0)}
OUTPUTS = {"all": None, "probes": lp.Output(fields=("u", "p"), probes={"tip": TIP}),
           "p every 2": lp.Output(fields=("p",), every=2)}


# ── the strict node lookup ────────────────────────────────────────────────────
def test_node_lookup_is_strict():
    mesh = Mesh(np.array([10, 30, 20]), np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
                {"t": MeshGroup("t", 2, "tri3", np.array([[10, 30, 20]]))})
    np.testing.assert_array_equal(mesh.rows_of([[20, 10], [30, 30]]), [[2, 0], [1, 1]])
    for bad in ([10, 25], [5], [40]):                        # inside the range, below, above: all refused
        with pytest.raises(MeshError, match="not in the mesh"):
            mesh.rows_of(bad)
    np.testing.assert_array_equal(mesh.rows_of([10, 25, 40], missing="ignore"), [0, -1, -1])
    with pytest.raises(MeshError):
        mesh.coords_of([11])
    with pytest.raises(ValueError, match="missing"):
        mesh.rows_of([25], missing="nearest")


# ── frequency response: the matrix ────────────────────────────────────────────
@pytest.fixture(scope="module")
def speaker(tmp_path_factory):
    folder = tmp_path_factory.mktemp("matrix")
    model = api.speaker_model(size=2e-3, acoustic=True)
    model.mesh(folder / "speaker.msh")
    return model, folder, {}


def solved(speaker, remesh, drive, output):
    model, folder, cache = speaker
    key = (remesh, drive, output)
    if key not in cache:
        rule = lp.PerWavelength(bands=[20.0, 600.0], folder=folder) if remesh else None
        cache[key] = lp.FrequencyResponse(model, drive=DRIVES[drive]).solve(FREQS, remesh=rule, output=OUTPUTS[output])
    return cache[key]


@pytest.mark.slow
@pytest.mark.parametrize("storage", ["live", "loaded"])
@pytest.mark.parametrize("output", list(OUTPUTS))
@pytest.mark.parametrize("drive", list(DRIVES))
@pytest.mark.parametrize("remesh", [False, True], ids=["one mesh", "remeshed"])
def test_frequency_response_matrix(speaker, remesh, drive, output, storage, tmp_path):
    live = solved(speaker, remesh, drive, output)
    res = live
    if storage == "loaded":
        live.save(tmp_path / "r.h5")
        res = lp.load(tmp_path / "r.h5")
        np.testing.assert_array_equal(res.values, live.values)

    if output == "p every 2":                                  # u not kept: what needs it fails loudly
        np.testing.assert_allclose(res.values, FREQS[::2])     # the same entries with or without remeshing
        for needs_u in (lambda: res.volume_velocity(on=FRONT), lambda: res.radiated_power(on=FRONT),
                        lambda: res.displacement(at=TIP)):
            with pytest.raises(ResultsError):
                needs_u()
        assert res.spl(at=(0.0, 0.15)).values.shape == (5,) and res.probe_names == []
        return

    ref = solved(speaker, remesh, drive, "all")                # same meshes, every field, live (another solve:
    q, P = res.volume_velocity(on=FRONT).values, res.radiated_power(on=FRONT).values     # to the GMRES tolerance)
    np.testing.assert_allclose(q, ref.volume_velocity(on=FRONT).values, rtol=1e-7)
    np.testing.assert_allclose(P, ref.radiated_power(on=FRONT).values, rtol=1e-7)
    assert np.all(P > 0)
    if remesh:                                                 # vs one mesh where its PML works (tuned to 1 kHz,
        one = solved(speaker, False, drive, "all")             # 0.1 m: it reflects much at low f, the remeshed
        hi = FREQS >= 600.0                                    # bands size the PML per wavelength)
        np.testing.assert_allclose(q[hi], one.volume_velocity(on=FRONT).values[hi], rtol=1e-2)
        np.testing.assert_allclose(P[hi], one.radiated_power(on=FRONT).values[hi], rtol=1e-2)
    vm = res.field("von_mises", index=3).values
    assert np.all(np.isfinite(vm)) and vm.max() > 0
    if storage == "loaded":                                     # read back: bit for bit
        np.testing.assert_array_equal(q, live.volume_velocity(on=FRONT).values)
        np.testing.assert_array_equal(vm, live.field("von_mises", index=3).values)

    if drive == "voltage":
        ts, ts_ref = res.thiele_small(on=FRONT), ref.thiele_small(on=FRONT)
        for k in ("fs", "Mms", "Cms", "Bl", "Sd"):
            assert ts[k] == pytest.approx(ts_ref[k], rel=1e-7), k
        if remesh:                                             # fs is below 600 Hz: within 1 % of one mesh
            assert ts["fs"] == pytest.approx(solved(speaker, False, drive, "all").thiele_small(on=FRONT)["fs"], rel=1e-2)
        if storage == "loaded":
            assert ts == live.thiele_small(on=FRONT)
    else:
        with pytest.raises(ResultsError, match="voltage"):
            res.thiele_small(on=FRONT)

    if output == "probes":
        tip = res.probe_signal("tip", "u", "z")
        np.testing.assert_allclose(tip.values, res.displacement(at=TIP, component="z").values, rtol=1e-12)
        np.testing.assert_allclose(res.probe_signal("tip", "p").values, res.probe("p", TIP).values, rtol=1e-12)
    else:
        assert res.probe_names == []


# ── modes, transient, sweeps ──────────────────────────────────────────────────
@pytest.mark.slow
@pytest.mark.parametrize("storage", ["live", "loaded"])
def test_modes_output_and_storage(tmp_path, storage):
    model = api.plate_model()
    model.mesh(tmp_path / "plate.msh")
    full = lp.Modes(model).solve(f_max=4000.0, orders=[0, 1])
    res = lp.Modes(model).solve(f_max=4000.0, orders=[0, 1], output=lp.Output(every=2))
    if storage == "loaded":
        res.save(tmp_path / "m.h5")
        res = lp.load(tmp_path / "m.h5")
    for m in (0, 1):                                          # every 2nd mode of each mode set
        np.testing.assert_allclose(res.order(m).frequencies, full.order(m).frequencies[::2], rtol=1e-12)
        np.testing.assert_allclose(res.order(m).mode(0).values, full.order(m).mode(0).values, rtol=1e-12)


@pytest.mark.slow
@pytest.mark.parametrize("storage", ["live", "loaded"])
def test_transient_output_and_storage(tmp_path, storage):
    model = api.speaker_model(size=2e-3)
    model.mesh(tmp_path / "speaker.msh")
    fs, n = 48_000.0, 30
    drive = lp.Force(np.sin(2 * np.pi * 400.0 * np.arange(n + 1) / fs), fs=fs)
    full = lp.Transient(model, drive=drive).solve(n_steps=n)
    res = lp.Transient(model, drive=drive).solve(n_steps=n, output=lp.Output(fields=("u",), every=3,
                                                                             probes={"tip": TIP}))
    if storage == "loaded":
        res.save(tmp_path / "t.h5")
        res = lp.load(tmp_path / "t.h5")
    np.testing.assert_allclose(res.times, full.times[::3], rtol=1e-12)
    tip = res.probe_signal("tip", "u", "z").values                   # full rate, whatever every= kept
    np.testing.assert_allclose(tip, full.displacement(at=TIP, component="z").values, rtol=1e-10, atol=1e-18)
    np.testing.assert_allclose(res.field("u", index=2).values, full.field("u", index=6).values, rtol=1e-12)


@pytest.mark.slow
def test_remeshed_voltage_sweep(tmp_path):
    """A mesh-size parameter (every run its own mesh) × remeshed bands × voltage: Thiele–Small per run and from
    the one sweep file."""
    model = lp.Model(api.SPEAKER_GEO, api.MATERIALS, parameters=lp.Parameters(h=2e-3))
    model.add([lp.Solid(n, m, mesh_size=lp.P("h")) for n, m in api.SPEAKER_PARTS],
              lp.Clamped("interface_constrained_rz"), lp.Clamped("interface_constrained_r", dofs="r"),
              lp.Coil("interface_forced"), lp.FluidStructure(FRONT), lp.FluidStructure("interface_acou_meca_rear"),
              lp.Air("subacou", mesh_size=0.01), lp.PML("PML", thickness=0.1, mesh_size=0.025, tuned_to=1000.0))
    model.mesh(tmp_path / "speaker.msh")
    study = lp.FrequencyResponse(model, drive=DRIVES["voltage"])
    sweep = lp.Sweep(study, "h", [2e-3, 3e-3]).solve(FREQS, remesh=lp.PerWavelength(bands=[20.0, 600.0],
                                                                                    folder=tmp_path))
    ts = [r.thiele_small(on=FRONT) for r in sweep.results]
    assert all(len(r.blocks) == 2 for r in sweep.results)
    alone = study.solve(FREQS, remesh=lp.PerWavelength(bands=[20.0, 600.0], folder=tmp_path))   # h = 2 mm alone
    for k in ("fs", "Mms", "Cms", "Bl", "Sd"):                       # a sweep run = the same study run alone
        assert ts[0][k] == pytest.approx(alone.thiele_small(on=FRONT)[k], rel=1e-7), k
    assert ts[1]["fs"] != pytest.approx(ts[0]["fs"], rel=1e-3)         # 3 mm: its own mesh (stiffer, ~6 %)
    sweep.save(tmp_path / "s.h5")
    back = lp.load(tmp_path / "s.h5")
    assert [r.thiele_small(on=FRONT) for r in back.results] == ts
