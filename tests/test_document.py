"""lp.Document: a model and its studies as plain data (JSON) — saved, read back, built again the same; the script
generated from it builds the same model and, run, meshes and solves every study."""
import json
import runpy
import shutil
import subprocess
import sys

import numpy as np
import pytest

import loudpy as lp
from support import current_api as api

STUDIES = [
    {"name": "modes", "type": "Modes", "params": {"f_max": 2500.0, "output": {"type": "Output",
                                                                                "params": {"every": 1}}}},
    {"name": "fr", "type": "FrequencyResponse", "params": {
        "drive": {"type": "Force", "params": {"amplitude": {"complex": [1.0, 0.0]}}},
        "freqs": {"from": 100.0, "to": 1000.0, "points": 4, "spacing": "log"}}},
]


def _plate(tmp_path):
    geo = tmp_path / "geometry" / "plate.geo"
    geo.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(api.PLATE_GEO, geo)
    model = lp.Model(geo, parameters=lp.Parameters(size=0.004)).add(
        lp.Solid("plate", lp.Material("Steel", rho=7800.0, E=2.1e11, nu=0.3), damping=lp.Undamped(),
                 mesh_size=lp.P("size")),
        lp.Clamped("interface_constrained_rz"), lp.Clamped("interface_constrained_r", dofs="r"),
        lp.MeshSize(lp.Near((0.0, 0.0)), 0.003))
    return model


def test_a_model_and_its_studies_round_trip(tmp_path):
    model = _plate(tmp_path)
    doc = lp.Document.of(model, STUDIES, name="plate")
    assert doc.check() == [] and doc.model["items"][0]["params"]["mesh_size"] == {"P": "size"}
    path = doc.save(tmp_path / "plate.loudpy.json")
    data = json.loads(path.read_text())
    assert data["format"] == lp.document.FORMAT and data["model"]["params"]["geometry"] == "geometry/plate.geo"
    back = lp.Document.load(path)
    assert back.build().to_script() == model.to_script()
    study, kw = back.study("fr")
    assert isinstance(study, lp.FrequencyResponse) and np.allclose(kw["freqs"], np.geomspace(100, 1000, 4))
    sweep = {"name": "sweep", "type": "Sweep", "params": {"parameters": {"size": [0.004, 0.003]}},
             "study": STUDIES[0]}
    back.studies.append(sweep)
    s, kw = back.study("sweep")
    assert isinstance(s, lp.Sweep) and kw["f_max"] == 2500.0
    moved = tmp_path / "elsewhere"                                  # the folder moves: paths still found
    shutil.copytree(tmp_path, moved, ignore=shutil.ignore_patterns("elsewhere"))
    assert lp.Document.load(moved / "plate.loudpy.json").build().source.path == moved / "geometry" / "plate.geo"


def test_refusals(tmp_path):
    doc = lp.Document.of(_plate(tmp_path), STUDIES + [STUDIES[0]])
    assert any("names must differ" in p for p in doc.check())
    doc.studies = [{"name": "bad", "type": "Modes", "params": {"f_max": -1.0}}]
    assert doc.check() == ["study 'bad': Modes.f_max: > 0, got -1.0"]
    with pytest.raises(lp.errors.ModelError, match="not a study"):
        lp.Document.of(_plate(tmp_path), [{"name": "x", "type": "Solid", "params": {}}])
    with pytest.raises(lp.errors.ModelError, match="not a loudpy document"):
        lp.Document.from_dict({"format": "other"})
    with pytest.raises(lp.errors.ModelError, match="no study 'nope'"):
        doc.study("nope")


def test_the_generated_script_builds_the_model_and_runs_the_studies(tmp_path):
    model = _plate(tmp_path)
    doc = lp.Document.of(model, STUDIES[:1], name="plate")
    doc.save(tmp_path / "plate.loudpy.json")
    script = tmp_path / "plate_ui.py"
    script.write_text(doc.to_script())
    assert runpy.run_path(str(script), run_name="imported")["model"].to_script() == model.to_script()
    run = subprocess.run([sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stderr[-2000:]
    res = lp.load(tmp_path, study="modes")                             # results/plate/modes_<stamp>.h5, stamped
    assert res.kind == "modes" and len(res) > 0 and (tmp_path / "meshes" / "plate.msh").exists()
    assert res.provenance["study"] == "modes" and res.provenance["simulation"] == "plate"
    assert res.provenance["fingerprint"] == doc.fingerprint("study:modes")    # the document's: current


def test_paths_problems_readiness_fingerprints_provenance(tmp_path, mesh_dir):
    """A document's parts by path (named items, studies, circuits), its problems with the path of their node, what the
    simulation still needs (readiness), what a study's results depend on (fingerprints) and what made them."""
    from loudpy.circuit import Coil, Semi, Source
    model = _plate(tmp_path)
    doc = lp.Document.of(model, [dict(s) for s in STUDIES], name="plate",
                         circuits={"amp": lp.Circuit(Source("amp", "in"), Coil("vc", "in", "0"))})
    assert doc.get("model/items/plate/params/mesh_size") == {"P": "size"}
    assert doc.get("studies/fr/params/freqs/to") == 1000.0 and doc.get("studies/1/name") == "fr"
    assert doc.get("circuits/amp/params/parts/vc/type") == "circuit.Coil"
    assert doc.set("studies/fr/params/freqs/to", 2000.0) == 1000.0 and doc.find("fr")["params"]["freqs"]["to"] == 2000.0
    assert doc.get("studies/nope", None) is None
    with pytest.raises(lp.errors.ModelError, match="no 'nope'"):
        doc.get("studies/nope")
    gone = doc.remove("model/items/interface_constrained_r")
    assert gone["type"] == "Clamped" and doc.insert("model/items", gone, at=2) == 2
    before = doc.fingerprint("study:fr")
    assert doc.fingerprint("study:modes") == doc.fingerprint("study:modes")
    doc.set("studies/modes/params/f_max", 2000.0)
    assert doc.fingerprint("study:fr") == before                                 # another study: no change
    doc.set("studies/fr/params/freqs/points", 5)
    assert doc.fingerprint("study:fr") != before
    t = doc.fingerprint("target")
    doc.set("model/items/plate/params/mesh_size", 0.005)
    assert doc.fingerprint("target") != t

    doc.set("studies/fr/params/freqs/points", "four")                         # a value its block refuses
    assert any(path == "studies/fr" and "points" in msg for path, msg in doc.problems())
    doc.set("studies/fr/params/freqs/points", 4)
    assert doc.problems() == [] and doc.check() == []

    steps = {s.key: s for s in doc.readiness()}
    assert steps["geometry"].state == "done" and steps["domains"].state == "done"
    assert steps["coil"].state == "todo"                                          # a forced response, no coil
    bare = lp.Model(tmp_path / "geometry" / "plate.geo")
    first = {s.key: s for s in lp.readiness(bare)}
    assert first["domains"].state == "todo" and first["studies"].state == "todo" and first["domains"].items

    speaker = api.speaker_problem(mesh_dir, acoustic=True)
    sdoc = lp.Document.of(speaker, [{"name": "fr", "type": "FrequencyResponse", "params": {
        "drive": {"type": "CircuitRef", "params": {"name": "amp"}}, "freqs": [200.0]}}],
        circuits={"amp": lp.Circuit(Source("amp", "in"), Coil("vc", "in", "0"))})
    steps = {s.key: s for s in sdoc.readiness(speaker)}
    assert steps["coil"].state == "done" and steps["motor"].state == "todo"
    assert steps["motor"].path.endswith("/params/motor")
    next(b for b in speaker.boundaries if isinstance(b, lp.Coil)).motor = Semi(Re=6.0, Bl=5.0)
    assert {s.key: s for s in sdoc.readiness(speaker)}["motor"].state == "done"
    assert {s.key: s for s in sdoc.readiness(speaker)}["circuits"].state == "done"   # the circuit it names: there
    wrong = {s.key: s for s in lp.readiness(speaker, sdoc.studies, {})}["circuits"]
    assert wrong.state == "todo" and "'amp'" in wrong.message and wrong.path == "circuits"
    assert "circuits" not in {s.key for s in lp.readiness(speaker, [])}             # none named: no step
    study, kw = sdoc.study("fr", speaker)
    res = study.solve(**kw)
    assert res.provenance == {"study": "fr", "simulation": sdoc.name, "fingerprint": sdoc.fingerprint("study:fr")}
    back = lp.load(res.save(tmp_path / "fr.h5"))
    assert back.provenance["fingerprint"] == sdoc.fingerprint("study:fr")
