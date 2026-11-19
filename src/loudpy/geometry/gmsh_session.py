"""
gmsh access in one place.

`gmsh_session()` is a context manager that initialises gmsh, sets quiet defaults and always
finalises it (even after an error), so no global gmsh state leaks between calls. `gmsh_module()`
returns the initialised module for code that builds a model step by step (the 3D builders).
"""
from __future__ import annotations

from contextlib import contextmanager


def gmsh_module():
    """The gmsh module, initialised and quiet (kept open: used by the step-by-step 3D builders)."""
    import gmsh
    if not gmsh.isInitialized():
        gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    return gmsh


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
