"""
A simulation as one document: plain data (a JSON file) holding the 2D model or the 3D scene and its studies as
lp.schema nodes — what the interface edits and saves, and what a script is generated from.

    doc = lp.Document.of(model, studies=[{"name": "modes", "type": "Modes", "params": {"f_max": 3000.0}}])
    doc.save("speaker/speaker.loudpy.json")     JSON; file paths relative to its folder
    doc = lp.Document.load("speaker/speaker.loudpy.json")
    model = doc.build()                          the lp.Model (or lp.Scene) again
    study, solve_kwargs = doc.study("modes", model)
    text = doc.to_script()                       a runnable script: builds the model; under __main__ it meshes and
                                                 runs every study, results saved in
                                                 results/<simulation>/<study>_<stamp>.h5 (loudpy.results.files)

The JSON:

    {"format": "loudpy.document/1", "name": …,
     "model": {"type": "Model", "params": {geometry, materials, mesh_order, grading, parameters},
               "items": [regions, boundaries, mesh controls: nodes, in order]}
     — or —
     "scene": {"type": "Scene", "params": {cabinet, air, parameters},
               "drivers": [{"driver": Driver.from_modes node, "place": Driver.place node}, …],
               "settings": [Scene.cavity / Scene.exterior / Scene.merged / Scene.walls nodes]},
     "circuits": {name: Circuit node, …}       (optional) the simulation's circuits: amplifier, filters, coils
     "plots": [PlotSpec dicts]                 (optional) plots kept with the simulation (lp.plot.PlotSpec.to_dict;
                                               each may name its result: "result", and a kept export how it
                                               is written: "export": {"format", "file", "entries"})
     "studies": [{"name": …, "type": "Modes", "params": {…}}, {"name": …, "type": "Sweep", "params": {…},
                  "study": the swept study's node}, …]}

A study driven by one of the circuits names it: "drive": {"type": "CircuitRef", "params": {"name": …}}; the script
writes each circuit once as a variable (its name) and the studies use it (drive=xo).

A cabinet written as a Python builder at the top level of a module file is {"import": "module.py:function"}.
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from loudpy import docpath, schema
from loudpy.errors import ModelError

FORMAT = "loudpy.document/1"
SUFFIX = ".loudpy.json"


@dataclass
class Document:
    """A simulation as plain data (module docstring): `model` or `scene` (one of them), `studies`, a `name` and the
    `folder` its relative file paths are read from."""
    name: str = "simulation"
    model: dict | None = None
    scene: dict | None = None
    studies: list = field(default_factory=list)
    folder: Path | None = None
    circuits: dict = field(default_factory=dict)                       # name → Circuit node
    plots: list = field(default_factory=list)                          # lp.plot.PlotSpec dicts

    # ── from objects ───────────────────────────────────────────────────────────────────────────────────────────────
    @classmethod
    def of(cls, target, studies=None, name: str | None = None, folder=None, circuits=None) -> "Document":
        """The document of an lp.Model or lp.Scene (and study nodes, and named circuits: {name: lp.Circuit or its
        node}); paths kept absolute until save()."""
        import loudpy as lp
        studies = [copy.deepcopy(s) for s in (studies or [])]
        for s in studies:
            _check_study(s)
        circ = {str(k): (copy.deepcopy(c) if isinstance(c, dict) else schema.describe(c))
                for k, c in (circuits or {}).items()}
        if isinstance(target, lp.Scene):
            return cls(name or "scene", scene=_describe_scene(target), studies=studies,
                       folder=Path(folder) if folder else None, circuits=circ)
        return cls(name or Path(str(target.source.name)).stem, model=_describe_model(target), studies=studies,
                   folder=Path(folder) if folder else None, circuits=circ)

    # ── files ──────────────────────────────────────────────────────────────────────────────────────────────────────
    def to_dict(self, folder=None) -> dict:
        """The JSON content; file paths relative to `folder` (default: the document's) when given."""
        base = Path(folder) if folder is not None else self.folder
        out: dict[str, Any] = {"format": FORMAT, "name": self.name}
        rel = (lambda p: _relative(p, base)) if base is not None else (lambda p: p)
        if self.model is not None:
            out["model"] = map_files(self.model, rel)
        if self.scene is not None:
            out["scene"] = map_files(self.scene, rel)
        if self.circuits:
            out["circuits"] = copy.deepcopy(self.circuits)
        if self.plots:
            out["plots"] = copy.deepcopy(self.plots)
        out["studies"] = [map_files(s, rel) for s in self.studies]
        return out

    def save(self, path) -> Path:
        """Write the JSON (paths relative to its folder; written next to the target, then renamed)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.to_dict(path.parent), indent=1, ensure_ascii=False)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text + "\n")
        os.replace(tmp, path)
        self.folder = path.parent.resolve()
        return path

    @classmethod
    def load(cls, path) -> "Document":
        """Read a document (its relative paths are read from its folder)."""
        path = Path(path).resolve()
        data = json.loads(path.read_text())
        return cls.from_dict(data, path.parent)

    @classmethod
    def from_dict(cls, data: dict, folder=None) -> "Document":
        """A document from its JSON content."""
        if data.get("format") != FORMAT:
            raise ModelError(f"not a loudpy document (format {data.get('format')!r}, expected {FORMAT!r})")
        if ("model" in data) == ("scene" in data):
            raise ModelError("a document holds a model or a scene (one of them)")
        return cls(data.get("name", "simulation"), data.get("model"), data.get("scene"), list(data.get("studies", [])),
                   Path(folder) if folder is not None else None, dict(data.get("circuits") or {}),
                   list(data.get("plots") or []))

    def copy(self) -> "Document":
        """A deep copy (the nodes are not shared)."""
        return Document(self.name, copy.deepcopy(self.model), copy.deepcopy(self.scene), copy.deepcopy(self.studies),
                        self.folder, copy.deepcopy(self.circuits), copy.deepcopy(self.plots))

    # ── checks and objects ─────────────────────────────────────────────────────────────────────────────────────────
    def check(self) -> list[str]:
        """The problems of the values (lp.schema.check of every node), [] when none."""
        return [msg for _, msg in self.problems()]

    def problems(self) -> list[tuple[str, str]]:
        """The problems of the values, each with the path of the node it is about (loudpy.docpath):
        [("model/items/cone", "…"), ("studies/fr", "…"), …]."""
        out: list[tuple[str, str]] = []
        key = "model" if self.model is not None else "scene"
        part = self.model if self.model is not None else self.scene
        if part is None:
            return [("", "the document holds neither a model nor a scene")]
        params = dict(part.get("params", {}))
        if isinstance(params.get("cabinet"), dict) and "import" in params["cabinet"]:
            params.pop("cabinet")                                    # a Python builder: checked when built
            if part["type"] == "Scene" and ":" not in part["params"]["cabinet"]["import"]:
                out.append((f"{key}/params/cabinet", "Scene.cabinet: an import is 'module.py:function'"))
        out += [(key, e) for e in schema.check({"type": part["type"], "params": params})
                if not e.startswith("Scene.cabinet: required")]
        for kind in ("items", "settings"):
            for i, n in enumerate(part.get(kind, [])):
                where = (n.get("params") or {}).get("name") or (n.get("type") if kind == "settings" else i)
                out += [(f"{key}/{kind}/{where}", e) for e in schema.check(n)]
        for i, d in enumerate(part.get("drivers", [])):
            where = ((d.get("place") or {}).get("params") or {}).get("name") or i
            out += [(f"{key}/drivers/{where}", e) for e in schema.check(d["driver"]) + schema.check(d["place"])]
        for cname, node in self.circuits.items():
            if not is_variable_name(cname):
                out.append((f"circuits/{cname}", f"circuit {cname!r}: a name of letters, digits and _ (not first), not "
                            f"one of the script's own names ({', '.join(sorted(RESERVED))})"))
            if not isinstance(node, dict) or node.get("type") != "Circuit":
                out.append((f"circuits/{cname}", f"circuit {cname!r}: a Circuit node, got {node!r}"))
                continue
            out += [(f"circuits/{cname}", f"circuit {cname!r}: {e}") for e in schema.check(node)]
        for i, spec in enumerate(self.plots):
            from loudpy.plot.spec import PlotSpec
            where = f"plots/{spec.get('name') or i}" if isinstance(spec, dict) else f"plots/{i}"
            try:
                PlotSpec.from_dict({k: v for k, v in spec.items() if k not in ("result", "export")})
            except Exception as e:                                   # noqa: BLE001  (said with its path)
                out.append((where, f"plot {i}: {e}"))
        names = [s.get("name") for s in self.studies]
        if len(set(names)) != len(names):
            out.append(("studies", f"studies: names must differ, got {names}"))
        for s in self.studies:
            where = f"studies/{s.get('name')}"
            out += [(where, f"study {s.get('name')!r}: {e}") for e in schema.check(_node(s))]
            if "study" in s:
                out += [(where, f"study {s.get('name')!r}: {e}") for e in schema.check(_node(s["study"]))]
            for node in (s.get("params", {}).get("drive"), s.get("study", {}).get("params", {}).get("drive")):
                if isinstance(node, dict) and node.get("type") == "CircuitRef":
                    ref = (node.get("params") or {}).get("name")
                    if ref not in self.circuits:
                        out.append((f"{where}/params/drive", f"study {s.get('name')!r}: no circuit {ref!r} "
                                    f"(circuits: {sorted(self.circuits)})"))
        return out

    # ── paths: what an edit, a check or a dependency points at (loudpy.docpath) ─────────────────────────────────────
    PARTS = ("name", "model", "scene", "studies", "circuits", "plots")

    def _root(self) -> dict:
        return {k: getattr(self, k) for k in self.PARTS}

    def get(self, path, default=docpath.REQUIRED):
        """The value at a path ("studies/fr/params/freqs/to", "model/items/cone/params/mesh_size" …)."""
        return docpath.get(self._root(), path, default)

    def set(self, path, value):
        """Set the value at a path (a part of the document itself: "studies", "circuits" …); the value before."""
        segs = docpath.split(path)
        if len(segs) == 1:
            if segs[0] not in self.PARTS:
                raise ModelError(f"a document's parts are {self.PARTS}, not {segs[0]!r}")
            before = getattr(self, segs[0])
            setattr(self, segs[0], value)
            return before
        return docpath.put(self._root(), segs, value)

    def remove(self, path):
        """Remove what is at a path; what it was."""
        return docpath.remove(self._root(), path)

    def insert(self, path, value, at: int | None = None) -> int:
        """Insert into the list at a path ("studies", "model/items" …); its index."""
        return docpath.insert(self._root(), path, value, at)

    # ── what a simulation still needs; what its results depend on ──────────────────────────────────────────────────
    def readiness(self, target=None) -> list:
        """What the simulation still needs, step by step (loudpy.readiness.Step: done / todo / warning / optional,
        the path to change); target: its model or scene if already built."""
        from loudpy.readiness import Step, readiness
        try:
            target = target if target is not None else self.build()
        except Exception as e:                                       # noqa: BLE001  (said as the first step)
            return [Step("build", "The model", "todo", str(e), "model" if self.model is not None else "scene")]
        return readiness(target, self.studies, self.circuits)

    def fingerprint(self, what: str = "target") -> str:
        """A digest of what a result depends on, changed by any change of it (files by their content, large ones by
        size and date): "target" (the model or scene), "study:<name>" (the study, the circuit it names, the target),
        "circuit:<name>"."""
        import hashlib
        kind, _, name = what.partition(":")
        if kind == "target":
            data: Any = self._content(self.model if self.model is not None else self.scene)
        elif kind == "circuit":
            data = self.circuits.get(name)
        elif kind == "study":
            s = self.find(name)
            refs = [n.get("params", {}).get("name") for n in (s.get("params", {}).get("drive"),
                    s.get("study", {}).get("params", {}).get("drive")) if isinstance(n, dict) and
                    n.get("type") == "CircuitRef"]
            inner = self.find(s["of"]) if s.get("of") else None
            data = {"study": s, "inner": inner, "circuits": {r: self.circuits.get(r) for r in refs},
                    "target": self.fingerprint("target")}
        else:
            raise ModelError(f"fingerprint of 'target', 'study:<name>' or 'circuit:<name>', not {what!r}")
        text = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def _content(self, node):
        """A node with its files (absolute paths) replaced by their content's digest."""
        import hashlib

        def digest(p: str) -> str:
            q = _absolute(p, self.folder)
            try:
                st = q.stat()
            except OSError:
                return f"missing:{Path(p).name}"
            if st.st_size > 5_000_000:                               # (a modes file: its size and date)
                return f"{q.name}:{st.st_size}:{st.st_mtime_ns}"
            return hashlib.sha256(q.read_bytes()).hexdigest()[:16]
        return map_files(node, digest)

    def circuit(self, name: str):
        """The lp.Circuit called `name` (one of the simulation's circuits)."""
        if name not in self.circuits:
            raise ModelError(f"no circuit {name!r}; circuits: {sorted(self.circuits)}")
        return schema.build(self.circuits[name])

    def _circuits(self) -> dict:
        return {n: schema.build(c) for n, c in self.circuits.items()}

    def _absolute(self, node):
        return map_files(node, lambda p: str(_absolute(p, self.folder)))

    def build(self, cache: dict | None = None):
        """The lp.Model or lp.Scene of the document; cache: a dict kept between builds — the drivers built before
        from the same node and files are used again (reading a modes file takes time)."""
        if self.model is not None:
            m = self._absolute(self.model)
            model = schema.build({"type": "Model", "params": m.get("params", {})})
            items: dict = {}
            for n in m.get("items", []):
                it = schema.build(n, items=items)
                model.add(it)
                items[it.name] = it
            return model
        if self.scene is None:
            raise ModelError("the document holds neither a model nor a scene")
        import loudpy as lp
        s = self._absolute(self.scene)
        params = dict(s.get("params", {}))
        cabinet = params.pop("cabinet")
        cabinet = _import(cabinet["import"], self.folder) if isinstance(cabinet, dict) and "import" in cabinet \
            else schema.from_value(cabinet)
        made: dict[str, Any] = cache if cache is not None else {}
        placed = []
        for d in s.get("drivers", []):
            key = json.dumps([d["driver"], self._content(d["driver"])], sort_keys=True, default=str)
            if key not in made or not _same_driver(made[key], d["driver"]):
                made[key] = schema.build(d["driver"])
            placed.append(schema.build(d["place"], driver=made[key]))
        values = {k: schema.from_value(v, schema.block("Scene").param(k)) for k, v in params.items()}
        if isinstance(values.get("parameters"), dict):
            values["parameters"] = lp.Parameters(**values["parameters"])
        scene = lp.Scene(cabinet, placed, **values)
        for n in s.get("settings", []):
            schema.build(n, scene=scene)
        return scene

    def find(self, name: str) -> dict:
        """The study node called `name`."""
        for s in self.studies:
            if s.get("name") == name:
                return s
        raise ModelError(f"no study {name!r}; studies: {[s.get('name') for s in self.studies]}")

    def study(self, name: str, target=None):
        """(study, solve kwargs) of the study `name` on `target` (default: build()); a Sweep / Adaptive wraps its
        "study" node. Its results say what made them (Results.provenance: the study, the simulation, the fingerprint
        of what they depend on)."""
        s = self._absolute(self.find(name))
        target = target if target is not None else self.build()
        circuits = self._circuits()
        made = {"study": name, "simulation": self.name, "fingerprint": self.fingerprint(f"study:{name}")}
        if "study" in s:
            inner, kwargs = schema.build(_node(s["study"]), model=target, circuits=circuits)
            outer = schema.build(_node(s), study=inner)
            inner.provenance = outer.provenance = made                # (kept in its results: Results.provenance)
            return outer, kwargs
        study, kwargs = schema.build(_node(s), model=target, circuits=circuits)
        study.provenance = made
        return study, kwargs

    # ── a script ───────────────────────────────────────────────────────────────────────────────────────────────────
    def to_script(self, folder=None, meshes: str = "meshes") -> str:
        """A runnable script of the document, in `folder` (default: the document's; paths HERE / relative): it
        builds `model` (or `scene`); under __main__ it meshes in meshes/ and runs every study, its result saved
        where results go (results/<simulation>/<study>_<stamp>.h5, stamped: loudpy.results.files)."""
        base = Path(folder).resolve() if folder is not None else self.folder

        def path(p: str) -> str:
            q = _absolute(p, self.folder)
            if base is None:
                return f"Path({str(q)!r})"
            return f"HERE / {Path(os.path.relpath(q, base)).as_posix()!r}"

        lines = ['"""', f"{self.name} — written by loudpy from the document {self.name}{SUFFIX} (Document.to_script).",
                 "Run it: python3 <this file> meshes the " + ("scene" if self.scene is not None else "model") +
                 " and runs every study (results in results/).", '"""',
                 "from pathlib import Path", "", "import numpy as np  # noqa: F401", "", "import loudpy as lp", "",
                 "HERE = Path(__file__).resolve().parent", ""]
        if self.model is not None:
            m = self.model
            mb = schema.block("Model")
            args = [f"{k}={schema.code(v, mb.param(k), path)}" for k, v in m.get("params", {}).items()
                    if k != "parameters"]
            if m.get("params", {}).get("parameters"):
                args.append("parameters=lp.Parameters(" + ", ".join(
                    f"{k}={schema.code(v)}" for k, v in m["params"]["parameters"].items()) + ")")
            lines.append(f"model = lp.Model({', '.join(args)})")
            for n in m.get("items", []):
                lines.append(f"model.add({schema.code(n, None, path)})")
            target = "model"
            mesh = f"model.mesh(HERE / {meshes!r} / {self.name + '.msh'!r})"
        else:
            s = self.scene or {}
            params = dict(s.get("params", {}))
            cab = params.pop("cabinet")
            if isinstance(cab, dict) and "import" in cab:
                mod, fn = cab["import"].rsplit(":", 1)
                lines += ["import sys  # noqa: E402", "",
                          f"sys.path.insert(0, str({path(str(Path(mod).parent) or '.')}))",
                          f"from {Path(mod).stem} import {fn} as cabinet  # noqa: E402"]
            else:
                lines.append(f"cabinet = {schema.code(cab, None, path)}")
            made: dict[str, str] = {}
            placed = []
            for d in s.get("drivers", []):
                key = json.dumps(d["driver"], sort_keys=True)
                if key not in made:
                    made[key] = f"driver{len(made) + 1}"
                    lines.append(f"{made[key]} = {schema.code(d['driver'], None, path)}")
                pb = schema.block("Driver.place")
                pargs = [f"{k}={schema.code(v, pb.param(k), path)}" for k, v in d["place"].get("params", {}).items()]
                placed.append(f"{made[key]}.place({', '.join(pargs)})")
            sb = schema.block("Scene")
            args = [f"{k}={schema.code(v, sb.param(k), path)}" for k, v in params.items() if k != "parameters"]
            if params.get("parameters"):
                args.append("parameters=lp.Parameters(" + ", ".join(
                    f"{k}={schema.code(v)}" for k, v in params["parameters"].items()) + ")")
            lines.append(f"scene = lp.Scene(cabinet, [{', '.join(placed)}]{''.join(', ' + a for a in args)})")
            for n in s.get("settings", []):
                b = schema.block(n["type"])
                sargs = [f"{k}={schema.code(v, b.param(k), path)}" for k, v in n.get("params", {}).items()]
                lines.append(f"scene.{n['type'].split('.')[1]}({', '.join(sargs)})")
            target = "scene"
            mesh = f"scene.mesh(HERE / {meshes!r})"
        if self.circuits:                                             # the simulation's circuits, once each
            lines.append("")
            for cname, node in self.circuits.items():
                lines.append(f"{cname} = {schema.code(node, None, path)}")
        lines += ["", "", 'if __name__ == "__main__":', f"    {mesh}"]
        for s in self.studies:
            lines += [f"    # {s['name']}"] + [f"    {ln}" for ln in _study_code(s, target, path)]
        if not self.studies:
            lines.append("    print('no study: add one')")
        from loudpy.codefmt import wrap_lines
        return "\n".join(wrap_lines(lines)) + "\n"


# ── helpers ────────────────────────────────────────────────────────────────────────────────────────────────────────
RESERVED = {"model", "scene", "cabinet", "study", "res", "lp", "np", "Path", "HERE", "sys"}   # the script's names


def is_variable_name(name: str) -> bool:
    """A circuit name the script can use as its variable."""
    import keyword
    return (name.isidentifier() and not keyword.iskeyword(name) and name not in RESERVED
            and not (name.startswith("driver") and name[6:].isdigit()))


def _node(s: dict) -> dict:
    return {"type": s["type"], "params": s.get("params", {})}


def _check_study(s: dict):
    if not isinstance(s, dict) or "name" not in s or "type" not in s:
        raise ModelError(f"a study is {{'name': …, 'type': …, 'params': {{…}}}}, got {s!r}")
    if schema.block(s["type"]).section != "studies":
        raise ModelError(f"study {s['name']!r}: {s['type']} is not a study")


def _study_code(s: dict, target: str, path) -> list[str]:
    def make(node, ctx: str) -> tuple[str, list[str]]:
        b = schema.block(node["type"])
        init: list[str] = []
        solve: list[str] = []
        for k, v in node.get("params", {}).items():
            (solve if b.param(k).stage == "solve" else init).append(f"{k}={schema.code(v, b.param(k), path)}")
        return f"lp.{b.type}({', '.join([ctx] + init)})", solve
    if "study" in s:
        inner, solve = make(s["study"], target)
        outer, _ = make(_node(s), inner)
        study = outer
    else:
        study, solve = make(_node(s), target)
    name = s["name"]
    if s["type"] == "FrequencyResponse" and target == "scene" and not any(a.startswith("store=") for a in solve):
        return [f"study = {study}",
                f"study.solve({', '.join(solve + [f'store=lp.result_path(HERE, {name!r}, study)'])})"]
    return [f"study = {study}", f"res = study.solve({', '.join(solve)})", f"res.save(HERE, name={name!r})"]


def _relative(p: str, base: Path) -> str:
    q = Path(p)
    if not q.is_absolute():
        return p
    try:
        return Path(os.path.relpath(q, Path(base).resolve())).as_posix()
    except ValueError:                                     # another drive
        return str(q)


def _absolute(p: str, folder) -> Path:
    q = Path(p).expanduser()
    return q if q.is_absolute() or folder is None else (Path(folder) / q).resolve()


def map_files(node, fn):
    """A copy of a node (or a part of a document) with fn applied to every file path in it."""
    if isinstance(node, list):
        return [map_files(x, fn) for x in node]
    if not isinstance(node, dict):
        return node
    out = {}
    block = None
    if "type" in node:
        try:
            block = schema.block(node["type"])
        except schema.SchemaError:
            block = None
    for k, v in node.items():
        if k == "params" and isinstance(v, dict) and block is not None:
            new = {}
            for name, x in v.items():
                kind = next((p.kind for p in block.params if p.name == name), "")
                new[name] = fn(x) if kind == "file" and isinstance(x, str) and x else map_files(x, fn)
            out[k] = new
        elif k == "import" and isinstance(v, str):
            mod, _, fname = v.rpartition(":")
            out[k] = f"{fn(mod)}:{fname}"
        else:
            out[k] = map_files(v, fn)
    return out


def _import(spec: str, folder):
    import importlib.util
    mod, _, fn = spec.rpartition(":")
    path = _absolute(mod, folder)
    sp = importlib.util.spec_from_file_location(path.stem, path)
    if sp is None or sp.loader is None:
        raise ModelError(f"cannot import the cabinet builder {spec!r}")
    module = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(module)
    return getattr(module, fn)


def _describe_model(m) -> dict:
    from loudpy.model.script import default_library
    if m.source.path is None:
        raise ModelError("the geometry is a Python builder: a document needs a geometry file (.geo, STEP …)")
    params: dict[str, Any] = {"geometry": str(Path(m.source.path).resolve())}
    if not default_library(m):
        params["materials"] = str(Path(m.materials.path).resolve())
    b = schema.block("Model")
    for k in ("mesh_order", "grading"):
        if getattr(m, k) != b.param(k).default:
            params[k] = schema.to_value(getattr(m, k))
    if len(m.parameters):
        params["parameters"] = {k: schema.to_value(v) for k, v in m.parameters.items()}
    items = [schema.describe(it) for it in list(m.regions) + list(m.boundaries) + list(m.mesh_controls)]
    return {"type": "Model", "params": params, "items": items}


def _driver_params(driver, placed: str = "") -> dict:
    """The Driver.from_modes parameters of a driver (as it is now: its source)."""
    src = driver.source or {}
    if not src.get("path") or src.get("rear") is None or not isinstance(src.get("geometry"), (str, Path)):
        raise ModelError(f"driver {placed or driver.name!r} was not made by lp.Driver.from_modes with a geometry "
                         "file: a document cannot hold it")
    dp: dict[str, Any] = {"path": str(Path(src["path"]).resolve()), "orders": [int(o) for o in src["orders"]],
                          "geometry": str(Path(src["geometry"]).resolve()),
                          "rear": schema.describe(src["rear"])}
    if src.get("f_max") is not None:
        dp["f_max"] = float(src["f_max"])
    if src.get("front") is not None:
        dp["front"] = schema.describe(src["front"])
    if src.get("coil"):
        dp["coil"] = src["coil"]
    if src.get("motor") is not None:                              # given here (else the modes file's)
        dp["motor"] = schema.describe(src["motor"])
    dp["name"] = driver.name
    return dp


def _same_driver(driver, node: dict) -> bool:
    """A driver built before is still what the node says (nothing changed it since)."""
    try:
        now = _driver_params(driver)
    except ModelError:
        return False
    want = dict(node.get("params") or {})
    for k in ("path", "geometry"):
        if k in want:
            want[k] = str(Path(want[k]).resolve())
    return json.dumps(now, sort_keys=True, default=str) == json.dumps(want, sort_keys=True, default=str)


def _describe_scene(s) -> dict:
    import dataclasses
    from loudpy.model.script import builder_module
    from loudpy.scene.cabinets import Box, Step
    if isinstance(s.cabinet, (Box, Step)):
        cabinet: Any = schema.describe(s.cabinet)
        if isinstance(s.cabinet, Step):
            cabinet["params"]["path"] = str(Path(s.cabinet.path).resolve())
    else:
        mod = builder_module(s.cabinet)
        if mod is None:
            raise ModelError("the cabinet is a Python builder outside a module file: a document cannot hold it")
        cabinet = {"import": f"{mod}:{s.cabinet.__name__}"}
    params: dict[str, Any] = {"cabinet": cabinet}
    air = getattr(s.air, "name", s.air)
    if air != "Air":
        params["air"] = air
    if len(s.parameters):
        params["parameters"] = {k: schema.to_value(v) for k, v in s.parameters.items()}
    drivers = []
    for d in s.drivers:
        dp = _driver_params(d.driver, d.name)
        at = [float(v) for v in np.asarray(d.frame.origin, float)]
        drivers.append({"driver": {"type": "Driver.from_modes", "params": dp},
                        "place": {"type": "Driver.place", "params": {"name": d.name, "at": at}}})
    settings = []
    for attr in ("_cavity_spec", "_open_spec", "_walls_spec"):
        spec = getattr(s, attr)
        if spec is None:
            continue
        values = {f.name: getattr(spec, f.name) for f in dataclasses.fields(spec)}
        kind = values.pop("kind", None)
        t = {"_cavity_spec": "Scene.cavity", "_walls_spec": "Scene.walls"}.get(attr) or f"Scene.{kind}"
        if callable(values.get("fixed")):
            raise ModelError("the walls' clamped nodes are a Python function: a document cannot hold it")
        b = schema.block(t)
        settings.append({"type": t, "params": {k: schema.to_value(v) for k, v in values.items()
                                               if k in {p.name for p in b.params} and
                                               schema.to_value(v) != schema.to_value(b.param(k).default)}})
    if getattr(s, "_keep", ()):
        settings.append({"type": "Scene.keep", "params": {"meshes": list(s._keep)}})
    return {"type": "Scene", "params": params, "drivers": drivers, "settings": settings}


__all__ = ["Document", "FORMAT", "SUFFIX"]
