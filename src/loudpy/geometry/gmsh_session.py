"""
gmsh access in one place.

`gmsh_session()` is a context manager that initialises gmsh, sets quiet defaults and always
finalises it (even after an error), so no global gmsh state leaks between calls. `gmsh_module()`
returns the initialised module for code that builds a model step by step (the 3D builders).
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path


def gmsh_module():
    """The gmsh module, initialised and quiet (kept open: used by the step-by-step 3D builders)."""
    import gmsh
    if not gmsh.isInitialized():
        gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    return gmsh


@contextmanager
def in_metres(gmsh, cad: bool = True):
    """While a CAD file is read: lengths converted to metres from the unit the file declares (gmsh's
    Geometry.OCCTargetUnit), the option put back afterwards — it is global to the process, and a later export or
    import (a STEP written, a .brep in metres) must not inherit it. cad False: nothing changes."""
    if not cad:
        yield
        return
    before = gmsh.option.getString("Geometry.OCCTargetUnit")
    gmsh.option.setString("Geometry.OCCTargetUnit", "M")
    try:
        yield
    finally:
        gmsh.option.setString("Geometry.OCCTargetUnit", before)


@contextmanager
def gmsh_session(terminal: bool = False, threads: int | None = None):
    """
    with gmsh_session() as gmsh: ...   — initialise, yield the module, always finalise.
    threads: General.NumThreads (None: gmsh default).
    """
    import gmsh
    owner = not gmsh.isInitialized()
    if owner:
        gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if terminal else 0)
        if threads is not None:
            gmsh.option.setNumber("General.NumThreads", threads)
        yield gmsh
    finally:
        if owner and gmsh.isInitialized():
            gmsh.finalize()


def physical_groups(geo_path: str, dim: int | None = None) -> list[tuple[int, str]]:
    """(dim, name) of the named physical groups of a .geo/.msh file (dim None: all)."""
    with gmsh_session() as gmsh:
        gmsh.open(str(geo_path))
        out = [(d, gmsh.model.getPhysicalName(d, t)) for d, t in gmsh.model.getPhysicalGroups()]
    return [(d, n.strip()) for d, n in out if n and (dim is None or d == dim)]


_SI_PREFIX = {"$": 1.0, ".MILLI.": 1e-3, ".CENTI.": 1e-2, ".DECI.": 1e-1, ".MICRO.": 1e-6, ".KILO.": 1e3}
_NAMED_UNITS = {"INCH": 0.0254, "FOOT": 0.3048, "MILLIMETRE": 1e-3, "MILLIMETER": 1e-3, "METRE": 1.0, "METER": 1.0}


def step_extent(path) -> float | None:
    """The largest |coordinate| of a STEP file's points, in metres (its declared length unit: an SI unit with its
    prefix, or a named one — inch, foot); None when the file says no unit loudpy knows."""
    import re
    text = Path(path).read_text(errors="ignore")
    unit = None
    m = re.search(r"LENGTH_UNIT\s*\(\s*\)\s*NAMED_UNIT\s*\(\s*\*\s*\)\s*SI_UNIT\s*\(\s*([^,]+?)\s*,\s*\.METRE\.\s*\)", text)
    if m:
        unit = _SI_PREFIX.get(m.group(1).strip())
    else:
        c = re.search(r"CONVERSION_BASED_UNIT\s*\(\s*'([^']+)'", text)
        unit = _NAMED_UNITS.get(c.group(1).strip().upper()) if c else None
    if unit is None:
        return None
    coords = re.findall(r"CARTESIAN_POINT\s*\(\s*'[^']*'\s*,\s*\(([^)]*)\)", text)
    values = [abs(float(v)) for c in coords for v in c.split(",") if v.strip()]
    return max(values) * unit if values else None


def step_correction(path, measured: float) -> float:
    """The factor that brings a STEP read by gmsh to metres. OpenCASCADE keeps a unit state across reads in one
    process, so a second STEP read may come in millimetres although Geometry.OCCTargetUnit says metres (gmsh
    4.x): the largest coordinate read (`measured`) is compared with the file's own, converted from its declared unit;
    a clean power of 1000 between them is corrected, anything else is left as it is (1.0)."""
    if Path(path).suffix.lower() not in (".step", ".stp") or not measured:
        return 1.0
    try:
        expected = step_extent(path)
    except (OSError, ValueError):
        return 1.0
    if not expected:
        return 1.0
    ratio = expected / measured
    for k in (1e-9, 1e-6, 1e-3, 1e3, 1e6, 1e9):
        if abs(ratio / k - 1.0) < 1e-3:
            return k
    return 1.0


def max_extent(gmsh, dimtags=None) -> float:
    """The largest |coordinate| of the entities (all of the model by default), from their bounding boxes."""
    boxes = [gmsh.model.getBoundingBox(d, t) for d, t in (dimtags or gmsh.model.getEntities())]
    return max((max(abs(v) for v in b) for b in boxes), default=0.0)
