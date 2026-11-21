"""
loudpy — fluid–structure finite elements for loudspeakers.

    import loudpy as lp
    model = lp.Model("speaker.geo")
    model.add(lp.Solid("membranne", "Paper"), lp.Clamped("interface_constrained_rz"),
              lp.Clamped("interface_constrained_r", dofs="r"), lp.Coil("interface_forced"))
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
from .results import (load, spl, compare, read_comsol, Output, Results, FrequencyResults, ModeResults, TimeResults, Data, Field,
                      Point, Points, Polar, Arc, Line)
from .study import (Modes, FrequencyResponse, PerWavelength, Transient, Sweep, SweepResults, Adaptive,
                    Convergence, convergence)
from .scene import Driver, Profile, Scene, Cabinet

from . import errors
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
    "Frame", "Mesh", "Group", "Box", "Near", "Interface", "BoundaryOf",
    # studies
    "Modes", "FrequencyResponse", "PerWavelength", "Transient", "Sweep", "SweepResults", "Adaptive", "Convergence",
    "convergence",
    # 3D scenes
    "Driver", "Profile", "Scene", "Cabinet",
    # results and evaluation
    "load", "spl", "compare", "read_comsol", "Output", "Results", "FrequencyResults", "ModeResults", "TimeResults", "Data", "Field",
    "Point", "Points", "Polar", "Arc", "Line",
    # messages and errors, lazily imported plotting
    "set_verbosity", "errors", "plot",
]
