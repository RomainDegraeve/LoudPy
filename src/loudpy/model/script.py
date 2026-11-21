"""
A model or a scene written back as Python code (Model.to_script, Scene.to_script): the script builds the same object
when it runs — to keep, share or edit a model made or changed from Python or from the interface.

    text = model.to_script(folder="my_study")       paths relative to the script's folder (HERE / …)
    Path("my_study/model.py").write_text(text)

A 2D model: its geometry file (a Python builder cannot be written out), material library, parameters and items (only
the fields that differ from their defaults). A scene: an lp.cabinets.Box / Step cabinet, drivers made by
lp.Driver.from_modes (their modes file, orders, geometry, wet profiles), the air and parameters; a cabinet builder
defined at the top level of a module file is imported from that module.
"""
from __future__ import annotations

import dataclasses
import inspect
import os
from pathlib import Path

import numpy as np

from loudpy.errors import ModelError

NAMES = ["Model", "Parameters", "P", "Material", "Solid", "Air", "PML", "Clamped", "Coil", "FluidStructure",
         "Radiation", "MeshSize", "Rayleigh", "Hysteretic", "Undamped", "Group", "Box", "Near", "Interface",
         "BoundaryOf"]


def _path(p, folder, flat=None) -> str:
    """Code for a path: HERE / 'name' for a file stored next to the script (flat: the set of such files, a result
    file's embedded files), HERE / 'relative' when the script's folder is known, else the absolute path."""
    p = Path(p).resolve()
    if flat and p in flat:
        return f"HERE / {p.name!r}"
    if folder is None:
        return f"Path({str(p)!r})"
    return f"HERE / {Path(os.path.relpath(p, Path(folder).resolve())).as_posix()!r}"


def default_library(model) -> bool:
    """True when the model uses loudpy's own material library (nothing to write out or store)."""
    return _default_library(model)


def _default_library(model) -> bool:
    import loudpy
    lib = getattr(model.materials, "path", None)
    packaged = Path(loudpy.__file__).parent / "data" / "materials.json"
    return lib is None or Path(lib).resolve() == packaged.resolve()


def item_code(it) -> str:
    """An item as code with only the fields that differ from their defaults: Solid(name='cone', material='Paper')."""
    parts = []
    for f in dataclasses.fields(it):
        v = getattr(it, f.name)
        if f.default is not dataclasses.MISSING:
            default = f.default
        elif f.default_factory is not dataclasses.MISSING:          # type: ignore[misc]
            default = f.default_factory()                           # type: ignore[misc]
        else:
            parts.append(f"{f.name}={v!r}")
            continue
        if f.name == "name" or repr(v) != repr(default):
            parts.append(f"{f.name}={v!r}")
    return f"{type(it).__name__}({', '.join(parts)})"


def builder_module(fn) -> Path | None:
    """The file of the module defining a cabinet / geometry builder at its top level (None: a lambda, a nested function,
    a builder defined in an interactive session)."""
    try:
        src = inspect.getsourcefile(fn)
    except TypeError:
        return None
    if not src or not Path(src).is_file() or getattr(fn, "__qualname__", "") != getattr(fn, "__name__", None) \
            or fn.__name__ == "<lambda>":
        return None
    return Path(src).resolve()


def model_script(model, folder=None, title: str | None = None, flat=None) -> str:
    """The code of a 2D model (module docstring); folder: where the script will be (paths relative to it); flat: files
    stored next to the script (a result file's embedded files)."""
    m = model
    if m.source.path is None:
        raise ModelError("the geometry is a Python builder: it cannot be written out — keep your script, or save the "
                         "geometry as a .geo / STEP file")
    lines = ['"""', f"{title or m.source.name} — written by loudpy (Model.to_script).", '"""',
             "from pathlib import Path", "", "import loudpy as lp",
             f"from loudpy import ({', '.join(NAMES)})  # noqa: F401", "",
             "HERE = Path(__file__).resolve().parent", ""]
    args = [_path(m.source.path, folder, flat)]
    if not _default_library(m):
        args.append(f"materials={_path(m.materials.path, folder, flat)}")
    if len(m.parameters):
        args.append(f"parameters=Parameters({', '.join(f'{k}={float(v)!r}' for k, v in m.parameters.items())})")
    if m.mesh_order != 2:
        args.append(f"mesh_order={m.mesh_order!r}")
    if m.grading != 0.3:
        args.append(f"grading={m.grading!r}")
    lines.append(f"model = Model({', '.join(args)})")
    items = list(m.regions) + list(m.boundaries) + list(m.mesh_controls)
    if items:
        lines.append("model.add(")
        lines += [f"    {item_code(it)}," for it in items]
        lines.append(")")
    lines.append("")
    return "\n".join(lines)


def scene_script(scene, folder=None, title: str | None = None, flat=None) -> str:
    """The code of a scene (module docstring); folder: where the script will be (paths relative to it); flat: files
    stored next to the script (a result file's embedded files). A cabinet builder defined at the top level of a
    module file is imported from it (the module is found next to the script when it is one of `flat`)."""
    from loudpy.scene.cabinets import Box, Step
    s = scene
    if isinstance(s.cabinet, (Box, Step)):
        cab = repr(s.cabinet)
        if isinstance(s.cabinet, Step):
            cab = cab.replace(repr(s.cabinet.path), _path(s.cabinet.path, folder, flat))
        cabinet_lines = [f"cabinet = {cab}"]
    else:
        mod = builder_module(s.cabinet)
        if mod is None:
            raise ModelError("the cabinet is a Python builder outside a module file (a lambda, a nested function, an "
                             "interactive session): it cannot be written out — keep your script")
        where = "HERE" if flat and mod in flat else f"str({_path(mod.parent, folder, flat)})"
        cabinet_lines = ["import sys", "", f"sys.path.insert(0, str({where}))",
                         f"from {mod.stem} import {s.cabinet.__name__} as cabinet  # noqa: E402"]
    lines = ['"""', f"{title or 'scene'} — a 3D scene written by loudpy (Scene.to_script).", '"""',
             "from pathlib import Path", "", "import loudpy as lp",
             "from loudpy import Profile, Group, Interface, BoundaryOf, Box as SelectBox, Near, P  # noqa: F401",
             "from loudpy.scene.cabinets import Box, Step, Port, Brace  # noqa: F401", "",
             "HERE = Path(__file__).resolve().parent", "", *cabinet_lines, ""]
    drivers: list[str] = []
    made: dict[int, str] = {}
    for d in s.drivers:
        src = d.driver.source or {}
        if not src.get("path") or src.get("rear") is None:
            raise ModelError(f"driver {d.name!r} was not made by lp.Driver.from_modes: it cannot be written out")
        key = id(d.driver)
        if key not in made:
            var = f"hp{len(made) + 1}"
            made[key] = var
            args = [_path(src["path"], folder, flat), repr(list(src["orders"]))]
            if src.get("f_max") is not None:
                args.append(f"f_max={src['f_max']!r}")
            args += [f"geometry={_path(src['geometry'], folder, flat)}", f"rear={src['rear']!r}"]
            if src.get("front") is not None:
                args.append(f"front={src['front']!r}")
            if src.get("coil"):
                args.append(f"coil={src['coil']!r}")
            args.append(f"name={d.driver.name!r}")
            lines.append(f"{var} = lp.Driver.from_modes({', '.join(args)})")
        o = tuple(float(v) for v in np.asarray(d.frame.origin, float))
        drivers.append(f"{made[key]}.place({d.name!r}, at={o!r})")
    params = f", parameters=lp.Parameters({', '.join(f'{k}={float(v)!r}' for k, v in s.parameters.items())})" \
        if len(s.parameters) else ""
    lines += ["", f"scene = lp.Scene(cabinet, [{', '.join(drivers)}], air={getattr(s.air, 'name', s.air)!r}{params})"]
    for attr, method in (("_cavity_spec", "cavity"), ("_open_spec", None), ("_walls_spec", "walls")):
        spec = getattr(s, attr)
        if spec is None:
            continue
        if callable(getattr(spec, "fixed", None)):
            raise ModelError("the walls' clamped nodes are a Python function: it cannot be written out")
        fields = {f.name: getattr(spec, f.name) for f in dataclasses.fields(spec)}
        name = method or fields.pop("kind")
        lines.append(f"scene.{name}({', '.join(f'{k}={v!r}' for k, v in fields.items())})")
    lines.append("")
    return "\n".join(lines)
