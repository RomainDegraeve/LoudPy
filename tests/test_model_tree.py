"""
Phase F1: the model tree, model.check(), mesh controls by selection (lp.MeshSize), deterministic meshing
by default, and studies that never change the user's model.
"""
import json

import numpy as np
import pytest

import loudpy as lp
from loudpy.errors import ModelError, SelectionError
from support import current_api as api


def test_tree_and_check():
    model = api.speaker_model(acoustic=True)
    model.add(lp.MeshSize(on=lp.Group("surround", "spider"), size=0.5e-3, name="suspension"))
    assert model.check() == []                      # coil, coupling: no note
    tree = model.tree(selections=True)
    for line in ("Regions (9)", "Solid 'glue': SolidGlue, Rayleigh, h 1 mm  → submeca_glue_1, submeca_glue_2",
                 "PML 'PML': thickness 100 mm, tuned to 1000.0 Hz", "Clamped 'interface_constrained_r': dofs r",
                 "MeshSize suspension: h 0.5 mm", "Mesh: not meshed"):
        assert line in tree, line
    assert str(model).startswith("Model HPNEW-Sketch.geo")

    structure = api.plate_model()
    assert structure.check() == ["no lp.Coil: modes only (a forced response needs a coil)"]


def test_check_lists_every_problem():
    model = lp.Model(api.SPEAKER_GEO, api.MATERIALS)
    model.add(lp.Solid("membranne", "NoSuchMaterial"), lp.Solid("nothing_like_this", "Paper"),
              lp.PML("PML", thickness=0.1))
    with pytest.raises(ModelError) as e:
        model.check()
    msg = str(e.value)
    assert "nothing_like_this" in msg and "NoSuchMaterial" in msg and "tuned_to" in msg


def test_mesh_size_control(tmp_path):
    """A MeshSize overrides the region's size on the groups it selects; it must select a meshed group."""
    def meshed(control):
        model = api.plate_model()
        if control:
            model.add(lp.MeshSize(on="plate", size=api.PLATE["h"] / 2))
        model.mesh(tmp_path / f"plate_{control}.msh")
        return model, json.load(open(tmp_path / f"plate_{control}.msh.json"))
    coarse, sig0 = meshed(False)
    fine, sig1 = meshed(True)
    n0 = sum(len(g) for g in coarse.meshed.groups.values() if g.dim == 2)
    n1 = sum(len(g) for g in fine.meshed.groups.values() if g.dim == 2)
    assert n1 > 2 * n0
    assert sig1["domains"][0][2] == pytest.approx(api.PLATE["h"] / 2) and sig0["domains"][0][2] == api.PLATE["h"]
    bad = api.plate_model().add(lp.MeshSize(on=lp.Box(r=(5.0, 6.0)), size=1e-3, name="far away"))
    with pytest.raises(SelectionError, match="far away"):
        bad.mesh(tmp_path / "bad.msh")


def test_identical_meshes_twice(tmp_path):
    """Deterministic meshing is the default: the same model meshed twice gives the same mesh."""
    meshes = []
    for k in range(2):
        model = api.speaker_model()
        model.mesh(tmp_path / f"speaker_{k}.msh", reuse=False)
        meshes.append(model.meshed)
    a, b = meshes
    np.testing.assert_array_equal(a.node_tags, b.node_tags)
    np.testing.assert_array_equal(a.coords, b.coords)
    for name in a.groups:
        np.testing.assert_array_equal(a.groups[name].cells, b.groups[name].cells)


def test_copy_is_independent():
    model = api.speaker_model(acoustic=True)
    twin = model.copy()
    twin["PML"].thickness = 0.5
    twin["subacou"].mesh_size = 1.0
    assert model["PML"].thickness == 0.1 and model["subacou"].mesh_size == 0.01
    assert [it.name for it in twin.regions + twin.boundaries] == [it.name for it in model.regions + model.boundaries]
    assert twin.meshed is None


@pytest.mark.slow
def test_remeshed_sweep_leaves_the_model_alone(mesh_dir, tmp_path):
    """FrequencyResponse(remesh=…) meshes copies: the user's items and mesh are unchanged."""
    model = api.speaker_problem(mesh_dir, acoustic=True)
    before = [(r.name, r.mesh_size, getattr(r, "thickness", None), getattr(r, "tuned_to", None))
              for r in model.regions]
    mesh_before = model.meshed
    rule = lp.PerWavelength(bands=[100.0, 400.0], air_max_size=0.02, folder=tmp_path)
    res = lp.FrequencyResponse(model).solve([100.0, 400.0], remesh=rule)
    assert len(res.blocks) == 2
    assert [(r.name, r.mesh_size, getattr(r, "thickness", None), getattr(r, "tuned_to", None))
            for r in model.regions] == before
    assert model.meshed is mesh_before
