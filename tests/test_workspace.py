"""lp.Workspace: a simulation's folder and document edited by commands — checked and refused as a whole, undone and
redone exactly (sets, removals, insertions, batches), said to listeners; the model built again only when it changed;
the results in its folder with what made them and whether they are current."""
import shutil

import numpy as np
import pytest

import loudpy as lp
from support import current_api as api


def _workspace(tmp_path, mesh_dir):
    from loudpy.circuit import Coil, Semi, Source
    model = api.speaker_problem(mesh_dir, acoustic=True)
    next(b for b in model.boundaries if isinstance(b, lp.Coil)).motor = Semi(Re=6.0, Bl=5.0, Le=3e-4)
    studies = [{"name": "fr", "type": "FrequencyResponse", "params": {
        "drive": {"type": "CircuitRef", "params": {"name": "amp"}}, "freqs": [200.0, 600.0]}}]
    ws = lp.Workspace.of(model, tmp_path / "sim", studies=studies, name="spk",
                         circuits={"amp": lp.Circuit(Source("amp", "in", U=2.0), Coil("vc", "in", "0"))})
    return ws, model


def test_commands_undo_redo_and_listeners(tmp_path, mesh_dir):
    ws, _ = _workspace(tmp_path, mesh_dir)
    seen = []
    ws.on_change(seen.append)
    assert ws.set("studies/fr/params/freqs", [100.0, 300.0]) == [200.0, 600.0]
    assert seen[-1] == ["studies/fr/params/freqs"]
    with pytest.raises(lp.errors.ModelError):
        ws.set("circuits/amp/params/parts/amp/params/U", "loud")                   # refused: the document as it was
    assert ws.document.get("circuits/amp/params/parts/amp/params/U") == 2.0 and len(ws.undos) == 1
    with ws.batch("two parts"):
        ws.set("circuits/amp/params/parts/amp/params/U", 2.83)
        ws.insert("circuits/amp/params/parts", {"type": "circuit.Resistor", "params": {
            "name": "R", "a": "in", "b": "in", "R": 1.0}})
    assert len(ws.undos) == 2 and seen[-1] == ["circuits/amp/params/parts/amp/params/U", "circuits/amp/params/parts"]
    with pytest.raises(lp.errors.ModelError):
        with ws.batch("refused half way"):
            ws.set("studies/fr/params/freqs", [50.0])
            ws.set("circuits/amp/params/parts/R/params/R", -1.0)
    assert ws.document.get("studies/fr/params/freqs") == [100.0, 300.0] and len(ws.undos) == 2
    gone = ws.remove("circuits/amp/params/parts/R")
    assert gone["params"]["R"] == 1.0
    with pytest.raises(lp.errors.ModelError, match="unknown parameter 'note'"):
        ws.set("studies/fr/params/note", "added key")
    ws.set("studies/fr/params/note", "added key", check=False)                  # (a new key: undo removes it)
    for _ in range(2):
        assert ws.undo()
    parts = [p["params"]["name"] for p in ws.document.get("circuits/amp/params/parts")]
    assert parts == ["amp", "vc", "R"] and "note" not in ws.document.find("fr")["params"]
    assert ws.undo() and [p["params"]["name"] for p in ws.document.get("circuits/amp/params/parts")] == ["amp", "vc"]
    assert ws.document.get("circuits/amp/params/parts/amp/params/U") == 2.0
    assert ws.undo() and ws.document.get("studies/fr/params/freqs") == [200.0, 600.0] and not ws.undo()
    for _ in range(4):
        assert ws.redo()
    assert ws.document.get("studies/fr/params/note") == "added key" and not ws.redo()
    ws.off_change(seen.append)


def test_target_readiness_study_results_and_save(tmp_path, mesh_dir):
    ws, model = _workspace(tmp_path, mesh_dir)
    first = ws.target
    assert ws.target is first                                                    # not built again
    ws.set("studies/fr/params/freqs", [300.0])
    assert ws.target is first                                                    # a study: the model unchanged
    steps = {s.key: s.state for s in ws.readiness()}
    assert steps["motor"] == "done" and steps["studies"] == "done"
    target = ws.target
    target.mesh(model._mesh_path)                                                # (the committed mesh, reused)
    study, kw = ws.study("fr")
    res = study.solve(**kw)
    (ws.folder / "results").mkdir(parents=True, exist_ok=True)
    res.save(ws.folder / "results" / "fr_1.h5")
    shutil.copy(ws.folder / "results" / "fr_1.h5", ws.folder / "results" / "other.h5")
    (ws.folder / "results" / "plain.h5").write_bytes((ws.folder / "results" / "fr_1.h5").read_bytes())
    entries = {e["name"]: e for e in ws.results()}
    assert entries["fr_1"]["study"] == "fr" and entries["fr_1"]["state"] == "current"
    assert ws.result(ws.folder / "results" / "fr_1.h5") == entries["fr_1"]
    assert ws.fingerprint("study:fr") == ws.document.fingerprint("study:fr")
    ws.set("circuits/amp/params/parts/amp/params/U", 2.83)                      # what fr depends on changed
    assert {e["name"]: e["state"] for e in ws.results()}["fr_1"] == "out of date"
    ws.undo()
    assert {e["name"]: e["state"] for e in ws.results()}["fr_1"] == "current"
    ws.set("model/items/" + model.regions[0].name + "/params/mesh_size", 0.004)
    assert ws.target is not target                                               # the model changed: built again
    path = ws.save()
    again = lp.Workspace.open(ws.folder)
    assert again.path == path and again.document.to_dict() == ws.document.to_dict()
    assert (ws.folder / "spk_ui.py").exists()
    assert np.isfinite(res.impedance().values).all()


def test_tracked_edits_and_library_files(tmp_path, mesh_dir):
    ws, model = _workspace(tmp_path, mesh_dir)
    lib = ws.folder / "spk_materials.json"
    ws.folder.mkdir(parents=True, exist_ok=True)
    model.materials.with_(path=lib).save(lib)
    ws.document.model["params"]["materials"] = str(lib)
    assert str(lib.resolve()) in ws.files()
    seen = []
    ws.on_change(seen.append)
    with ws.tracking("material and a study"):                                  # edits made directly on the data
        ws.document.studies[0]["params"]["freqs"] = [100.0]
        ws.document.studies.append({"name": "m", "type": "Modes", "params": {"n": 4}})
        lib.write_text(lib.read_text().replace("{", "{ ", 1))
    assert sorted(seen[-1]) == sorted(["studies", str(lib.resolve())])        # (the list changed length: whole)
    with ws.tracking("nothing"):
        pass
    assert len(ws.undos) == 1
    with pytest.raises(RuntimeError):
        with ws.tracking("refused"):
            ws.document.studies[0]["params"]["freqs"] = [5.0]
            raise RuntimeError("its own check")
    assert ws.document.studies[0]["params"]["freqs"] == [100.0] and len(ws.undos) == 1
    with ws.tracking("one value"):
        ws.document.studies[0]["params"]["freqs"] = [150.0]
    assert seen[-1] == ["studies/fr/params/freqs/0"]                          # (as small as it can be said)
    text = lib.read_text()
    assert ws.undo() and ws.undo()
    assert [s["name"] for s in ws.document.studies] == ["fr"] and lib.read_text() != text
    assert ws.document.studies[0]["params"]["freqs"] == [200.0, 600.0]
    assert ws.redo() and lib.read_text() == text and len(ws.document.studies) == 2

