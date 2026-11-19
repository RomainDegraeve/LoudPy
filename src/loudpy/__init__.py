"""
loudpy — fluid–structure finite elements for loudspeakers.

    import loudpy as lp
    model = lp.Model("speaker.geo")
    model.add(lp.Solid("membranne", "Paper"), lp.Clamped("interface_constrained"), lp.Coil("interface_forced"))
    model.mesh("mesh.msh")

    res = lp.Modes(model).solve(n=20)
    lp.plot.mode(res, 0)

Layers (each importable on its own): loudpy.model, loudpy.geometry, loudpy.fem, loudpy.physics,
loudpy.solvers, loudpy.study, loudpy.scene, loudpy.results, loudpy.plot. The low-level toolkit of
the 3D coupling (loudpy.Studies.Cavity_coupling, loudpy.Studies.Exterior_dtn) and the internal
engines stay importable from their modules; they are not part of the top-level namespace.
See docs/ (Sphinx) for the user guide and the API reference.
"""

import os as _os

# scipy (OpenBLAS / MKL) and numba may each load an OpenMP runtime; allow both (set before they load).
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ── public API ───────────────────────────────────────────────────────────────
from .model import (
    Model, Solid, Air, PML, Clamped, Coil, FluidStructure, Radiation,
    Rayleigh, Hysteretic, Undamped, Force, Voltage,
    Material, MaterialLibrary, load_materials, materials,
)
from .geometry import Frame, Mesh
from .results import load, spl, Results, FrequencyResults, ModeResults, TimeResults
from .study import Modes, FrequencyResponse, PerWavelength, Transient
from .scene import Driver, Profile, Scene, Cabinet

__version__ = "0.1.0"


def __getattr__(name):
    """lp.plot is imported on first use (keeps `import loudpy` free of matplotlib)."""
    if name == "plot":
        import importlib
        return importlib.import_module("loudpy.plot")
    raise AttributeError(f"module 'loudpy' has no attribute {name!r}")


__all__ = [
    # model
    "Model", "Solid", "Air", "PML", "Clamped", "Coil", "FluidStructure", "Radiation",
    "Rayleigh", "Hysteretic", "Undamped", "Force", "Voltage",
    "Material", "MaterialLibrary", "load_materials", "materials",
    # geometry
    "Frame", "Mesh",
    # studies
    "Modes", "FrequencyResponse", "PerWavelength", "Transient",
    # 3D scenes
    "Driver", "Profile", "Scene", "Cabinet",
    # results
    "load", "spl", "Results", "FrequencyResults", "ModeResults", "TimeResults",
    # lazily imported
    "plot",
]
