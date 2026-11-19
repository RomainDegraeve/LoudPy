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


@pytest.mark.slow
def test_remeshed_coupled_sweep(tmp_path):
    """One mesh per band, the air / PML sized by the band wavelength; the model is restored afterwards."""
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
    np.testing.assert_allclose(lam, [c / 400.0, c / 1600.0])
    assert res.blocks[0].meshes["air"].n_nodes != res.blocks[1].meshes["air"].n_nodes
    assert (air.mesh_size, pml.thickness, pml.mesh_size, pml.tuned_to) == before
    assert res.spl(at=(0.0, 0.1)).shape == (4,)


def test_drive_checks(mesh_dir):
    model = api.speaker_model()
    with pytest.raises(NotImplementedError, match="coupled"):
        lp.FrequencyResponse(model, drive=lp.Voltage(1.0, Re=8, Bl=7))
    with pytest.raises(ValueError, match="harmonic"):
        lp.FrequencyResponse(model, drive=lp.Force(np.ones(10), fs=1000))
    with pytest.raises(ValueError, match="signal"):
        lp.Transient(model, drive=lp.Force(1.0))
    with pytest.raises(RuntimeError, match="mesh the model"):
        lp.Modes(api.speaker_model()).solve(f_max=1000)


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
