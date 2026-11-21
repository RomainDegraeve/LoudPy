"""
What a result file says about the model and the study that produced it (result format 3).

    d = describe(model)          JSON-able dict: loudpy version, geometry (files and their digests), parameters,
                                 regions (material, damping, mesh size), boundaries, mesh controls, mesh, the
                                 model tree, and per mesh group its physical data ("groups": region, role, ρ, E,
                                 ν, c, damping…), which is what derived fields need: a result file is enough
    files = geometry_files(model)    {name: bytes} of the geometry (a .geo and the files it merges, a CAD file),
                                 or the builder's code; embedded in the file up to EMBED_LIMIT bytes
    settings(study)              the study's settings (type, drive, frequencies or bands, …)

ModelDescription wraps the description read back from a file: res.model.tree, .parameters, .groups,
.files, .extract(folder) (writes the embedded geometry files back).
"""
from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path

import numpy as np

EMBED_LIMIT = 2_000_000                # bytes of geometry files embedded in a result file


def _value(v):
    """A JSON-friendly form of an item field (selectors, materials, damping, parameter references…)."""
    from loudpy.model.materials import Material
    from loudpy.model.parameters import P
    from loudpy.geometry.selectors import Selector
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    if isinstance(v, (Selector, P)):
        return repr(v)
    if isinstance(v, Material):
        return {"name": v.name, **v.to_dict()}
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return {"type": type(v).__name__, **{f.name: _value(getattr(v, f.name)) for f in dataclasses.fields(v)}}
    if isinstance(v, (list, tuple)):
        return [_value(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _value(x) for k, x in v.items()}
    if isinstance(v, np.ndarray):
        return v.tolist() if v.size <= 64 else f"array{v.shape}"
    return repr(v)


def _item(it) -> dict:
    return {"type": type(it).__name__, **{f.name: _value(getattr(it, f.name)) for f in dataclasses.fields(it)}}


_PART_DATA = ("rho", "E", "nu", "c", "eta", "alpha_ray", "beta_ray", "alpha", "n", "omega_pml")


def group_data(parts, owner) -> dict:
    """{mesh group: {"region", "role", ρ, E, ν, c, damping, PML layer…}} of the region parts of a meshed model."""
    out = {}
    for p in parts:
        if not hasattr(p, "tri"):
            continue
        d = {"region": owner.get(p.name, ""), "role": p.role}
        d.update({k: float(getattr(p, k)) for k in _PART_DATA if getattr(p, k, None) is not None})
        layer = getattr(p, "layer", None)
        if layer is not None:
            d["layer"] = dataclasses.asdict(layer)
        out[p.name] = d
    return out


def describe(model) -> dict:
    """The description of an lp.Model or an lp.Scene (module docstring)."""
    from loudpy import __version__
    from loudpy.scene.scene import Scene
    if isinstance(model, Scene):
        return _describe_scene(model)
    owner = {g: r for r, gs in getattr(model, "_region_groups", {}).items() for g in gs}
    mesh = None
    if model.meshed is not None:
        sig = Path(str(model._mesh_path) + ".json") if model._mesh_path else None
        mesh = {"path": str(model._mesh_path), "n_nodes": int(model.meshed.n_nodes),
                "cells": {g.name: [g.cell_type, len(g)] for g in model.meshed.groups.values()}}
        if sig is not None and sig.exists():
            import json
            stored = json.loads(sig.read_text())
            mesh["signature"] = {k: v for k, v in stored.items() if k != "pml_layers"}
    return {
        "kind": "model", "dim": 2, "loudpy": __version__,
        "geometry": {"name": model.source.name, "builder": model.source.path is None,
                     "files": model.source.files()},
        "parameters": {k: float(v) for k, v in model.parameters.items()},
        "mesh_order": model.mesh_order, "grading": model.grading,
        "materials": str(getattr(model.materials, "path", None) or "loudpy library"),
        "regions": [_item(r) for r in model.regions],
        "boundaries": [_item(b) for b in model.boundaries],
        "mesh_controls": [_item(c) for c in getattr(model, "mesh_controls", [])],
        "mesh": mesh,
        "groups": group_data(model.parts, owner),
        "tree": model.tree(),
    }


def _describe_scene(scene) -> dict:
    from loudpy import __version__
    drivers = []
    for d in scene.drivers:
        drivers.append({"name": d.name, "origin": np.asarray(d.frame.origin).tolist(),
                        "rotation": np.asarray(d.frame.rotation).tolist(), "source": _value(d.driver.source),
                        "modes": len(d.driver.modes)})
    settings = {}
    for key in ("_cavity", "_open"):
        v = getattr(scene, key, None)
        if v is not None:
            settings[key.strip("_")] = _value(v)
    return {
        "kind": "scene", "dim": 3, "loudpy": __version__,
        "cabinet": {"name": getattr(scene.cabinet, "__qualname__", repr(scene.cabinet))},
        "parameters": {k: float(v) for k, v in scene.parameters.items()},
        "air": _value(scene.air), "configuration": scene.configuration,
        "drivers": drivers, "settings": settings,
        "folder": str(scene.folder) if scene.folder else None,
        "tree": scene.tree(),
    }


def geometry_files(model) -> dict:
    """{name: bytes} of the model's geometry (files, or the builder's / cabinet's code) to embed in a result."""
    from loudpy.scene.scene import Scene
    out = {}
    if isinstance(model, Scene):
        fn = model.cabinet
        try:
            out[f"{getattr(fn, '__name__', 'cabinet')}.py"] = inspect.getsource(fn).encode()
        except (OSError, TypeError):
            pass
        return out
    src = model.source
    if src.path is None:
        try:
            out[f"{getattr(src.builder, '__name__', 'builder')}.py"] = inspect.getsource(src.builder).encode()
        except (OSError, TypeError):
            pass
        return out
    for name in src.files():
        p = src.path if name == src.path.name else (Path(name) if Path(name).is_absolute() else src.path.parent / name)
        if p.exists():
            out[name] = p.read_bytes()
    if sum(len(b) for b in out.values()) > EMBED_LIMIT:
        return {}
    return out


def settings(study) -> dict:
    """The settings of a study: its type and its public, simple attributes (drive as a description)."""
    out = {"type": type(study).__name__}
    for k, v in vars(study).items():
        if k.startswith("_") or k in ("model", "results", "is_scene"):
            continue
        if callable(v) and not dataclasses.is_dataclass(v):
            continue
        if k == "drive" and v is not None:
            d = _item(v) if dataclasses.is_dataclass(v) else {"type": type(v).__name__}
            amp = np.asarray(getattr(v, "amplitude", 0.0))
            if amp.size > 64:                                   # a signal: kept as an array of the file
                d["amplitude"] = "arrays/drive_amplitude"
                out.setdefault("arrays", {})["drive_amplitude"] = amp
            out["drive"] = d
            continue
        out[k] = _value(v)
    return out


class ModelDescription:
    """The model description read from a result file (format 3): dict access, .tree, .parameters, .groups,
    .files, .extract(folder)."""

    def __init__(self, data: dict, files: dict | None = None):
        self.data = dict(data)
        self.files = dict(files or {})

    def __getitem__(self, key):
        return self.data[key]

    def get(self, key, default=None):
        """An entry of the description (dict.get)."""
        return self.data.get(key, default)

    def __contains__(self, key):
        return key in self.data

    @property
    def tree(self) -> str:
        """The model tree as printed when the results were computed."""
        return self.data.get("tree", "")

    @property
    def parameters(self) -> dict:
        """The parameter values of the model."""
        return dict(self.data.get("parameters", {}))

    @property
    def groups(self) -> dict:
        """{mesh group: physical data} (region, role, ρ, E, ν, c, damping…)."""
        return dict(self.data.get("groups", {}))

    def extract(self, folder) -> list[Path]:
        """Write the embedded geometry files (or the builder's code) into `folder`; their paths."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        out = []
        for name, data in self.files.items():
            p = folder / Path(name).name
            p.write_bytes(data)
            out.append(p)
        return out

    def __repr__(self):
        return self.tree or f"ModelDescription({self.data.get('kind', '?')})"
