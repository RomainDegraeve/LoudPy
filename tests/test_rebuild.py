"""
Result files rebuild their model: the description stores the model (or scene) as a script using the files embedded
with it, and res.model.rebuild() makes the lp.Model / lp.Scene again — the same model, the same results.
"""
import shutil

import numpy as np
import pytest

import loudpy as lp
from loudpy.errors import ResultsError
from support import current_api as api


def test_model_rebuilt_from_its_result_file(tmp_path):
    src = tmp_path / "work"
    src.mkdir()
    geo = shutil.copy(api.PLATE_GEO, src / "plate.geo")                       # a geometry that will disappear
    steel = lp.Material("Steel", rho=7850.0, E=2.1e11, nu=0.3)
    model = lp.Model(geo, parameters=lp.Parameters(h=0.002))
    model.add(lp.Solid("plate", steel, damping=lp.Undamped(), mesh_size=lp.P("h")),
              lp.Clamped("interface_constrained_rz"), lp.Clamped("interface_constrained_r", dofs="r"))
    model.mesh(src / "plate.msh")
    lp.Modes(model).solve(f_max=3000.0).save(tmp_path / "modes.h5")
    shutil.rmtree(src)

    res = lp.load(tmp_path / "modes.h5")
    assert res.model.rebuildable and "Solid(name='plate'" in res.model["script"]
    again = res.model.rebuild(tmp_path / "rebuilt")
    assert (tmp_path / "rebuilt" / "plate.geo").exists() and (tmp_path / "rebuilt" / "model.py").exists()
    assert [repr(i) for i in again.regions + again.boundaries] == [repr(i) for i in model.regions + model.boundaries]
    assert dict(again.parameters) == {"h": 0.002}
    again.mesh(tmp_path / "rebuilt" / "plate.msh")
    np.testing.assert_allclose(lp.Modes(again).solve(f_max=3000.0).values, res.values, rtol=1e-9)


def test_builder_geometry_is_not_rebuildable(tmp_path):
    def disc(occ):
        return {"plate": occ.addRectangle(0, 0, 0, 0.1, 0.002)}
    model = lp.Model(disc).add(lp.Solid("plate", lp.Material("Steel", rho=7850.0, E=2.1e11, nu=0.3),
                                        damping=lp.Undamped(), mesh_size=0.002),
                               lp.Clamped("rim", on=lp.Box(r=0.1)))
    model.mesh(tmp_path / "disc.msh")
    lp.Modes(model).solve(f_max=3000.0).save(tmp_path / "m.h5")
    res = lp.load(tmp_path / "m.h5")
    assert not res.model.rebuildable and "builder" in res.model["script_note"]
    with pytest.raises(ResultsError, match="cannot rebuild"):
        res.model.rebuild()


CABINET = """import loudpy as lp

W, H, D = {W}, {H}, {D}


def cabinet(occ):
    inner = occ.addBox(0.0, 0.0, {zb} - D, W, H, D)
    outer = occ.addBox(0.0, 0.0, {zb} - D, W, H, D + {front})
    return lp.Cabinet(inner=inner, outer=outer)
"""


@pytest.mark.slow
def test_scene_with_a_builder_cabinet_rebuilt(tmp_path, mesh_dir, cache_dir):
    """A scene whose cabinet is a builder function in a module file (as the labyrinth and column examples): the
    module is stored in the result file; the scene is rebuilt after the module is gone, with the same response."""
    import importlib
    import sys
    api.driver_modes(mesh_dir, cache_dir)
    W, H, D = api.BOX
    mod = tmp_path / "work" / "my_cabinet.py"
    mod.parent.mkdir()
    mod.write_text(CABINET.format(W=W, H=H, D=D, zb=api.Z_B, front=api.Z_F - api.Z_B))
    sys.path.insert(0, str(mod.parent))
    try:
        cabinet = importlib.import_module("my_cabinet").cabinet
        hp = lp.Driver.from_modes(cache_dir / "driver_modes.h5", [0, 1], f_max=3000.0, geometry=api.SPEAKER_GEO,
                                  rear=api.REAR_PROFILE, front=api.FRONT_PROFILE)
        scene = lp.Scene(cabinet, [hp.place("main", at=(W / 2, H / 2))])
        scene.cavity(mesh_size=(0.006, 0.03), f_max=3000.0, eta=0.02, cache_modes=False)
        scene.mesh(tmp_path / "meshes")
        res = lp.FrequencyResponse(scene, drive=lp.Force(1.0)).solve([100.0, 400.0])
        res.save(tmp_path / "box.h5")
    finally:
        sys.path.remove(str(mod.parent))
        sys.modules.pop("my_cabinet", None)
    shutil.rmtree(mod.parent)
    back = lp.load(tmp_path / "box.h5")
    assert back.model.rebuildable and "from my_cabinet import cabinet" in back.model["script"]
    again = back.model.rebuild(tmp_path / "rebuilt")
    assert (tmp_path / "rebuilt" / "my_cabinet.py").exists() and isinstance(again, lp.Scene)
    again.mesh(tmp_path / "meshes")
    res2 = lp.FrequencyResponse(again, drive=lp.Force(1.0)).solve([100.0, 400.0])
    np.testing.assert_allclose(res2.q, res.q, rtol=1e-9)
