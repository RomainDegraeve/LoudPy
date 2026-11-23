"""
A simulation as one document: plain data (a JSON file) holding the 2D model or the 3D scene and its studies as
lp.schema nodes — what the interface edits and saves, and what a script is generated from.

    doc = lp.Document.of(model, studies=[{"name": "modes", "type": "Modes", "params": {"f_max": 3000.0}}])
    doc.save("speaker/speaker.loudpy.json")     JSON; file paths relative to its folder
    doc = lp.Document.load("speaker/speaker.loudpy.json")
    model = doc.build()                          the lp.Model (or lp.Scene) again
    study, solve_kwargs = doc.study("modes", model)
    text = doc.to_script()                       a runnable script: builds the model; under __main__ it meshes and
                                                 runs every study, results saved in results/<name>.h5

The JSON:

    {"format": "loudpy.document/1", "name": …,
     "model": {"type": "Model", "params": {geometry, materials, mesh_order, grading, parameters},
               "items": [regions, boundaries, mesh controls: nodes, in order]}
     — or —
     "scene": {"type": "Scene", "params": {cabinet, air, parameters},
               "drivers": [{"driver": Driver.from_modes node, "place": Driver.place node}, …],
               "settings": [Scene.cavity / Scene.exterior / Scene.merged / Scene.walls nodes]},
     "studies": [{"name": …, "type": "Modes", "params": {…}}, {"name": …, "type": "Sweep", "params": {…},
                  "study": the swept study's node}, …]}

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

from loudpy import schema
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

    # ── from objects ───────────────────────────────────────────────────────────────────────────────────────────────
    @classmethod
    def of(cls, target, studies=None, name: str | None = None, folder=None) -> "Document":
        """The document of an lp.Model or lp.Scene (and study nodes); paths kept absolute until save()."""
        import loudpy as lp
        studies = [copy.deepcopy(s) for s in (studies or [])]
        for s in studies:
            _check_study(s)
        if isinstance(target, lp.Scene):
            return cls(name or "scene", scene=_describe_scene(target), studies=studies,
                       folder=Path(folder) if folder else None)
        return cls(name or Path(str(target.source.name)).stem, model=_describe_model(target), studies=studies,
                   folder=Path(folder) if folder else None)

    # ── files ──────────────────────────────────────────────────────────────────────────────────────────────────────
    def to_dict(self, folder=None) -> dict:
        """The JSON content; file paths relative to `folder` (default: the document's) when given."""
        base = Path(folder) if folder is not None else self.folder
        out: dict[str, Any] = {"format": FORMAT, "name": self.name}
        rel = (lambda p: _relative(p, base)) if base is not None else (lambda p: p)
        if self.model is not None:
            out["model"] = _map_files(self.model, rel)
        if self.scene is not None:
            out["scene"] = _map_files(self.scene, rel)
        out["studies"] = [_map_files(s, rel) for s in self.studies]
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
                   Path(folder) if folder is not None else None)

    def copy(self) -> "Document":
        """A deep copy (the nodes are not shared)."""
        return Document(self.name, copy.deepcopy(self.model), copy.deepcopy(self.scene), copy.deepcopy(self.studies),
                        self.folder)

    # ── checks and objects ─────────────────────────────────────────────────────────────────────────────────────────
    def check(self) -> list[str]:
        """The problems of the values (lp.schema.check of every node), [] when none."""
        out: list[str] = []
        part = self.model if self.model is not None else self.scene
        if part is None:
            return ["the document holds neither a model nor a scene"]
        params = dict(part.get("params", {}))
        if isinstance(params.get("cabinet"), dict) and "import" in params["cabinet"]:
            params.pop("cabinet")                                    # a Python builder: checked when built
            if part["type"] == "Scene" and ":" not in part["params"]["cabinet"]["import"]:
                out.append("Scene.cabinet: an import is 'module.py:function'")
        out += [e for e in schema.check({"type": part["type"], "params": params})
                if not e.startswith("Scene.cabinet: required")]
        for n in part.get("items", []) + part.get("settings", []):
            out += schema.check(n)
        for d in part.get("drivers", []):
            out += schema.check(d["driver"]) + schema.check(d["place"])
        names = [s.get("name") for s in self.studies]
        if len(set(names)) != len(names):
            out.append(f"studies: names must differ, got {names}")
        for s in self.studies:
            out += [f"study {s.get('name')!r}: {e}" for e in schema.check(_node(s))]
            if "study" in s:
                out += [f"study {s.get('name')!r}: {e}" for e in schema.check(_node(s["study"]))]
        return out

    def _absolute(self, node):
        return _map_files(node, lambda p: str(_absolute(p, self.folder)))

    def build(self):
        """The lp.Model or lp.Scene of the document."""
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
        made: dict[str, Any] = {}
        placed = []
        for d in s.get("drivers", []):
            key = json.dumps(d["driver"], sort_keys=True)
            if key not in made:
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
        "study" node."""
        s = self._absolute(self.find(name))
        target = target if target is not None else self.build()
        if "study" in s:
            inner, kwargs = schema.build(_node(s["study"]), model=target)
            outer = schema.build(_node(s), study=inner)
            return outer, kwargs
        return schema.build(_node(s), model=target)

    # ── a script ───────────────────────────────────────────────────────────────────────────────────────────────────
    def to_script(self, folder=None, results: str = "results", meshes: str = "meshes") -> str:
        """A runnable script of the document, in `folder` (default: the document's; paths HERE / relative): it
        builds `model` (or `scene`); under __main__ it meshes in meshes/ and runs every study, its result saved
        as results/<study name>.h5."""
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
        lines += ["", "", 'if __name__ == "__main__":', f"    (HERE / {results!r}).mkdir(exist_ok=True)", f"    {mesh}"]
        for s in self.studies:
            lines += [f"    # {s['name']}"] + [f"    {ln}" for ln in _study_code(s, target, path, results)]
        if not self.studies:
            lines.append("    print('no study: add one')")
        return "\n".join(lines) + "\n"


# ── helpers ────────────────────────────────────────────────────────────────────────────────────────────────────────
def _node(s: dict) -> dict:
    return {"type": s["type"], "params": s.get("params", {})}


def _check_study(s: dict):
    if not isinstance(s, dict) or "name" not in s or "type" not in s:
        raise ModelError(f"a study is {{'name': …, 'type': …, 'params': {{…}}}}, got {s!r}")
    if schema.block(s["type"]).section != "studies":
        raise ModelError(f"study {s['name']!r}: {s['type']} is not a study")


def _study_code(s: dict, target: str, path, results: str) -> list[str]:
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
    out = f"HERE / {results!r} / {s['name'] + '.h5'!r}"
    if s["type"] == "FrequencyResponse" and target == "scene" and not any(a.startswith("store=") for a in solve):
        return [f"{study}.solve({', '.join(solve + [f'store={out}'])})"]
    return [f"{study}.solve({', '.join(solve)}).save({out})"]


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


def _map_files(node, fn):
    """A copy of a node (or a part of a document) with fn applied to every file path in it."""
    if isinstance(node, list):
        return [_map_files(x, fn) for x in node]
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
                new[name] = fn(x) if kind == "file" and isinstance(x, str) and x else _map_files(x, fn)
            out[k] = new
        elif k == "import" and isinstance(v, str):
            mod, _, fname = v.rpartition(":")
            out[k] = f"{fn(mod)}:{fname}"
        else:
            out[k] = _map_files(v, fn)
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
        src = d.driver.source or {}
        if not src.get("path") or src.get("rear") is None or not isinstance(src.get("geometry"), (str, Path)):
            raise ModelError(f"driver {d.name!r} was not made by lp.Driver.from_modes with a geometry file: a "
                             "document cannot hold it")
        dp: dict[str, Any] = {"path": str(Path(src["path"]).resolve()), "orders": [int(o) for o in src["orders"]],
                              "geometry": str(Path(src["geometry"]).resolve()),
                              "rear": schema.describe(src["rear"])}
        if src.get("f_max") is not None:
            dp["f_max"] = float(src["f_max"])
        if src.get("front") is not None:
            dp["front"] = schema.describe(src["front"])
        if src.get("coil"):
            dp["coil"] = src["coil"]
        dp["name"] = d.driver.name
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
