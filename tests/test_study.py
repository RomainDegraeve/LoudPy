"""
New studies (phase 5): what is new beyond the regression cases (those run through the studies
in the other test files): remeshing per band, drive checks, the ARPACK method, acoustic modes.
"""
import numpy as np
import pytest

import loudpy as lp
from support import current_api as api


def test_perwavelength_bands():
    rule = lp.PerWavelength(bands=[20, 160, 1280])
    bands = rule.split([20, 50, 160, 300, 1280, 5000])
    assert [list(b) for b in bands] == [[20, 50], [160, 300], [1280, 5000]]
    every = lp.PerWavelength(every=2).split([1, 2, 3, 4, 5])
    assert [list(b) for b in every] == [[1, 2], [3, 4], [5]]
    with pytest.raises(ValueError):
        lp.PerWavelength().split([1, 2])


def test_perwavelength_plan_sizes_the_air_at_the_band_top(tmp_path):
    """The air of a band is sized at its top frequency (the next band's start, else its last frequency), the PML at
    its start; size_at="start" keeps the former rule; the band meshes are listed with their state, nothing meshed."""
    model = api.speaker_model(acoustic=True)
    c = lp.materials["Air"].c
    rule = lp.PerWavelength(bands=[400, 1600], air_elements=6, folder=tmp_path)
    plan = rule.plan(model, [400.0, 800.0, 1600.0, 2400.0])
    assert [(b["start"], b["top"]) for b in plan] == [(400.0, 1600.0), (1600.0, 2400.0)]
    np.testing.assert_allclose([b["wavelength"] for b in plan], [c / 1600.0, c / 2400.0])
    np.testing.assert_allclose([b["pml_wavelength"] for b in plan], [c / 400.0, c / 1600.0])
    twin = rule.band_model(model, plan[1])
    assert twin["subacou"].mesh_size == min(c / 2400.0 / 6, 0.01) and twin["PML"].tuned_to == 1600.0
    assert np.isclose(twin["PML"].thickness, 0.25 * c / 1600.0) and model["PML"].thickness == 0.1
    old = lp.PerWavelength(bands=[400, 1600], size_at="start").plan(model, [400.0, 800.0])
    assert old[0]["top"] == 400.0
    assert lp.PerWavelength(every=2).plan(model, [100.0, 200.0, 300.0])[0]["top"] == 200.0
    files = rule.mesh_files(model, [400.0, 800.0, 1600.0, 2400.0])
    assert [(f["title"], f["path"].name, f["state"]) for f in files] == [
        ("air 400–1600 Hz", "mesh_400Hz.msh", "missing"), ("air 1600–2400 Hz", "mesh_1600Hz.msh", "missing")]
    with pytest.raises(ValueError, match="size_at"):
        lp.PerWavelength(bands=[20], size_at="middle")


@pytest.mark.slow
def test_remeshed_coupled_sweep(tmp_path):
    """One mesh per band, the air / PML sized by the band wavelength; the model is restored afterwards; the
    structure's mesh is the same in every band (only the air is remeshed); the meshes say what they are."""
    model = api.speaker_model(acoustic=True)
    air, pml = model["subacou"], model["PML"]
    before = (air.mesh_size, pml.thickness, pml.mesh_size, pml.tuned_to)
    rule = lp.PerWavelength(bands=[400, 1600], air_elements=6, pml_thickness=0.25, pml_elements=4,
                            air_max_size=0.01, folder=tmp_path)
    res = lp.FrequencyResponse(model).solve([400.0, 800.0, 1600.0, 2400.0], remesh=rule)
    assert len(res.blocks) == 2 and list(res.frequencies) == [400.0, 800.0, 1600.0, 2400.0]
    assert sorted(p.name for p in tmp_path.glob("mesh_*Hz.msh")) == ["mesh_1600Hz.msh", "mesh_400Hz.msh"]
    lam = [b.meta["wavelength"] for b in res.blocks]
    c = lp.materials["Air"].c
    np.testing.assert_allclose(lam, [c / 1600.0, c / 2400.0])                 # the bands' tops
    np.testing.assert_allclose([b.meta["pml_wavelength"] for b in res.blocks], [c / 400.0, c / 1600.0])
    assert res.blocks[0].meshes["air"].n_nodes != res.blocks[1].meshes["air"].n_nodes
    u0, u1 = (b.mesh_of("u").coords for b in res.blocks)
    assert np.array_equal(u0, u1)                                              # the structure is not remeshed
    assert (air.mesh_size, pml.thickness, pml.mesh_size, pml.tuned_to) == before
    assert res.spl(at=(0.0, 0.1)).shape == (4,)
    assert [f["state"] for f in rule.mesh_files(model, [400.0, 800.0, 1600.0, 2400.0])] == ["built", "built"]
    info = lp.mesh_info(tmp_path / "mesh_1600Hz.msh")
    assert info["role"] == "air 1600–2400 Hz" and info["band"] == [1600.0, 2400.0] and info["nodes"] > 0
    assert "subacou" in " ".join(info["groups"]["2"]) and info["elements"]
    from loudpy.geometry.mesh_files import fingerprint, listing
    assert res.blocks[1].meta["mesh_fingerprint"] == fingerprint(tmp_path / "mesh_1600Hz.msh")
    assert [m["role"] for m in listing(tmp_path)] == ["air 1600–2400 Hz", "air 400–1600 Hz"]


def test_drive_checks(mesh_dir):
    model = api.speaker_model()
    lp.FrequencyResponse(model, drive=lp.Voltage(1.0, Re=8, Bl=7))          # the structure alone too (its coil)
    with pytest.raises(ValueError, match="harmonic"):
        lp.FrequencyResponse(model, drive=lp.Force(np.ones(10), fs=1000))
    with pytest.raises(ValueError, match="signal"):
        lp.Transient(model, drive=lp.Force(1.0))
    with pytest.raises(RuntimeError, match="mesh the model"):
        lp.Modes(api.speaker_model()).solve(f_max=1000)
    hyst = api.speaker_model()                     # a loss factor has no time-domain form: refused, not ignored
    hyst["surround"].damping = lp.Hysteretic()
    hyst.mesh(mesh_dir / "speaker_0.001_meca.msh")
    with pytest.raises(ValueError, match="'surround' use hysteretic damping.*lp.Rayleigh"):
        lp.Transient(hyst, drive=lp.Force(np.zeros(5), fs=48000.0)).solve()


def test_modes_arpack_method(mesh_dir):
    """One shift-invert solve around f_target gives the same modes as the multishift solve."""
    model = api.speaker_model()
    model.mesh(mesh_dir / "speaker_0.001_meca.msh")
    multi = lp.Modes(model).solve(f_max=3000.0)
    one = lp.Modes(model).solve(f_max=3000.0, method="arpack", n_modes=12, f_target=800.0)
    for f in one.frequencies:
        assert np.min(np.abs(multi.frequencies - f)) / f < 1e-6


def test_acoustic_modes(tmp_path):
    """Air only, rigid walls (fixed in phase 3: the summary print used to fail)."""
    model = lp.Model(api.SPEAKER_GEO, api.MATERIALS).add(lp.Air("subacou", mesh_size=0.02))
    model.mesh(tmp_path / "air.msh")
    res = lp.Modes(model, physics="acoustic").solve(f_max=3000.0, n_modes=8, f_target=500.0)
    assert len(res) > 0 and np.all(np.diff(res.frequencies) >= 0) and res.names == ["p"]
    assert res.mode(0, "p").values.shape == (res.mesh("air").n_nodes,)
