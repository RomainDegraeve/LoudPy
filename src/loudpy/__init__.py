"""
loudpy — fluid–structure finite elements for loudspeakers.

    import loudpy as lp
    model = lp.Model("speaker.geo")
    model.add(lp.Solid("cone", "Paper"), lp.Clamped("rims", on=lp.Group("surround_rim", "spider_rim")),
              lp.Clamped("axis", dofs="r"), lp.Coil("coil_drive"))
    model.mesh("mesh.msh")

    modes = lp.Modes(model).solve(f_max=5000)
    lp.plot.mode(modes, 0)

Layers (each importable on its own, dependencies point down): loudpy.model, loudpy.geometry,
loudpy.fem, loudpy.physics, loudpy.solvers, loudpy.study, loudpy.scene, loudpy.results, loudpy.plot.
The 3D coupling toolkit (loudpy.physics.modal, loudpy.physics.exterior) is importable from its
modules; it is not part of the top-level namespace. See docs/ (Sphinx) for the user guide and the
API reference.
"""

# ── public API ───────────────────────────────────────────────────────────────
from .model import (
    Model, MeshSize, Parameters, P, Solid, Air, PML, Clamped, Coil, FluidStructure, Radiation,
    Rayleigh, Hysteretic, Undamped, Force, Voltage,
    Material, MaterialLibrary, air, load_materials, materials,
)
from .geometry import Frame, Mesh, Group, Box, Near, Interface, BoundaryOf
from .geometry.mesh_files import mesh_info
from .results import (load, load_data, spl, compare, read_comsol, Output, Results, FrequencyResults, ModeResults, TimeResults, Data, Field,
                      Point, Points, Polar, Arc, Line, On)
from .study import (Modes, FrequencyResponse, PerWavelength, Transient, Sweep, SweepResults, Adaptive,
                    Convergence, convergence)
from .scene import Driver, Profile, Scene, Cabinet, cabinets

from . import circuit, docpath, errors, progress, schema, signals
from .circuit import Circuit
from .document import Document
from .readiness import readiness, suggest
from .results.exporters import export
from .results.files import result_path
from .workspace import Workspace
from ._log import set_verbosity

__version__ = "0.2.0"


def __getattr__(name):
    """lp.plot is imported on first use (keeps `import loudpy` free of matplotlib)."""
    if name == "plot":
        import importlib
        return importlib.import_module("loudpy.plot")
    raise AttributeError(f"module 'loudpy' has no attribute {name!r}")


__all__ = [
    # model
    "Model", "MeshSize", "Parameters", "P", "Solid", "Air", "PML", "Clamped", "Coil", "FluidStructure", "Radiation",
    "Rayleigh", "Hysteretic", "Undamped", "Force", "Voltage",
    "Material", "MaterialLibrary", "air", "load_materials", "materials",
    # geometry and selections
    "Frame", "Mesh", "Group", "Box", "Near", "Interface", "BoundaryOf", "mesh_info",
    # studies
    "Modes", "FrequencyResponse", "PerWavelength", "Transient", "Sweep", "SweepResults", "Adaptive", "Convergence",
    "convergence",
    # 3D scenes
    "Driver", "Profile", "Scene", "Cabinet", "cabinets",
    # the electrical side: amplifier, filters, coils
    "Circuit", "circuit",
    # results and evaluation
    "load", "load_data", "spl", "compare", "read_comsol", "Output", "Results", "FrequencyResults", "ModeResults", "TimeResults", "Data", "Field",
    "Point", "Points", "Polar", "Arc", "Line", "On",
    # messages and errors, lazily imported plotting
    "set_verbosity", "errors", "progress", "schema", "signals", "plot",
    # a whole simulation as plain data, what it still needs
    "Document", "docpath", "readiness", "suggest", "Workspace", "export", "result_path",
]
