"""
Saving results one way (loudpy.results.files): results/<simulation>/<study>_<stamp>.h5, stamped with what made them
(the folder's document gives the fingerprint), the newest one read back by study; every older format still opens
(format 1 samples, a format-2 file written by the code before format 3, format 3); exports by suffix through one
registry (loudpy.results.exporters).
"""
import json
import time

import numpy as np
import pytest

import loudpy as lp
from loudpy.results import exporters, files
from loudpy.results.schema import info
from support import current_api as api

LEGACY = "tests/data/legacy"


def _plate(mesh_dir):
    return api._meshed(api.plate_model(), mesh_dir, "plate")


def _speaker(mesh_dir):
    return api._meshed(api.speaker_model(), mesh_dir, api._speaker_tag(1e-3, False, False))


def test_one_folder_convention_and_provenance(tmp_path, mesh_dir):
    model = _plate(mesh_dir)
    study = lp.Modes(model)
    study.solve(f_max=3000.0, f_min=10.0)
    path = study.save(tmp_path, name="modes")
    assert path.parent == tmp_path / "results" / tmp_path.name and path.name.startswith("modes_")
    res = lp.load(tmp_path, study="modes")
    assert res.provenance == {"study": "modes", "simulation": tmp_path.name}     # no document: no fingerprint
    assert info(path)["provenance"] == res.provenance and info(path)["count"] == len(res)
    second = files.result_path(tmp_path, "modes", stamp=path.stem[len("modes_"):])  # the same second: a free name
    assert second != path and second.name.endswith("_2.h5")
    time.sleep(1.1)
    newer = res.save(tmp_path, name="modes")
    assert files.latest(tmp_path, "modes") == newer and len(files.results_of(tmp_path, "modes")) == 2
    with pytest.raises(FileNotFoundError, match="no result of study 'fr'"):
        lp.load(tmp_path, study="fr")
    with pytest.raises(lp.errors.ResultsError, match="study name"):
        files.result_path(tmp_path, "a/b")
    # with the simulation's document: its name, and the fingerprint of the study as the document has it
    doc = lp.Document.of(model, [{"name": "modes", "type": "Modes", "params": {"f_max": 3000.0}}], name="plate")
    doc.save(tmp_path / "plate.loudpy.json")
    path = res.save(tmp_path, name="modes")
    assert path.parent.name == "plate"
    made = lp.load(path).provenance
    assert made == {"study": "modes", "simulation": "plate", "fingerprint": doc.fingerprint("study:modes")}
    ws = lp.Workspace.open(tmp_path)
    assert ws.result(path)["state"] == "current"
    # a study stamped before it writes its own file (a scene's store=)
    fr = lp.FrequencyResponse(_speaker(mesh_dir), drive=lp.Force(1.0))
    where = lp.result_path(tmp_path, "fr", fr)
    assert fr.provenance == {"study": "fr", "simulation": "plate"} and not where.exists()
    assert fr.solve([200.0]).provenance == fr.provenance


def test_every_format_opens(tmp_path):
    two = lp.load(f"{LEGACY}/format2_plate_modes.h5")                  # written by the code before format 3
    assert two.kind == "modes" and len(two) == 2 and info(f"{LEGACY}/format2_plate_modes.h5")["format"] == "2.0"
    np.testing.assert_allclose(two.values, [509.0003632, 1976.82087939], rtol=1e-9)
    one = lp.load(f"{LEGACY}/freq_meca.h5")                             # format 1 (the former studies)
    assert one.kind == "frequency" and len(one) > 0
    three = lp.load(two.save(tmp_path / "again.h5"))
    assert info(tmp_path / "again.h5")["format"].startswith("3") and np.array_equal(three.values, two.values)


def test_a_sweep_is_one_file(tmp_path, mesh_dir):
    model = lp.Model(api.PLATE_GEO, parameters=lp.Parameters(E=2.1e11))
    steel = lp.Material("Steel", rho=7850.0, E=2.1e11, nu=0.3)
    model.add(lp.Solid("plate", steel, E=lp.P("E"), mesh_size=0.004, damping=lp.Undamped()),
              lp.Clamped("interface_constrained_rz"),
              lp.Clamped("interface_constrained_r", dofs="r"))
    model.mesh(tmp_path / "plate.msh")
    sweep = lp.Sweep(lp.Modes(model), {"E": [2.0e11, 2.1e11]})
    res = sweep.solve(f_max=2000.0, f_min=10.0)
    with pytest.raises(ValueError, match="one .h5 file"):
        res.save(tmp_path / "runs")
    path = res.save(tmp_path, name="sweep")
    assert lp.load(tmp_path, study="sweep").is_sweep and info(path)["sweep"]


def test_exports_by_suffix(tmp_path, mesh_dir):
    res = lp.FrequencyResponse(_speaker(mesh_dir), drive=lp.Force(1.0)).solve([200.0, 400.0])
    d = res.displacement(at=(0.0, 0.0), component="z")
    assert {".csv", ".nc", ".mat", ".npz"} <= set(exporters.formats(d))
    assert lp.export(d, tmp_path / "u.csv").exists() and d.export(tmp_path / "u.nc").exists()
    path = lp.export([d.rename(label="a"), d.abs().rename(label="b")], tmp_path / "two.csv")
    head = path.read_text().splitlines()[0]
    assert head == "f,a_re,a_im,b"
    rows = [{"mode": 0, "f": 1.0}, {"mode": 1, "f": 2.0}]
    assert lp.export(rows, tmp_path / "t.csv").read_text().splitlines()[0] == "mode,f"
    f = res.field("u", index=1)
    assert set(exporters.formats(f)) >= {".vtu", ".html", ".csv", ".mat", ".npz", ".gif", ".mp4"}
    assert f.export(tmp_path / "u.vtu").exists() and lp.export(f, tmp_path / "u_nodes.csv").exists()
    assert lp.export(res, tmp_path / "all.npz").exists() and lp.export(res, tmp_path / "copy.h5").exists()
    import matplotlib.pyplot as plt
    ax = d.plot()
    assert lp.export(ax, tmp_path / "u.pdf").exists()
    plt.close(ax.figure)
    with pytest.raises(ValueError, match="offered: .csv"):
        lp.export(d, tmp_path / "u.docx")
    assert json.loads(np.load(tmp_path / "all.npz")["info"].item())["kind"] \
        == "frequency"
