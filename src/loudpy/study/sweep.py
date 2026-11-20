"""
Parametric sweeps: one study run for each value of a parameter, on copies of the model.

    model = lp.Model("plate.geo", parameters=lp.Parameters(E=2.1e11))
    model.add(lp.Solid("plate", steel, E=lp.P("E"), …), …)
    model.mesh("plate.msh")
    sweep = lp.Sweep(lp.Modes(model), "E", [1.9e11, 2.0e11, 2.1e11]).solve(f_max=3000)
    sweep.values                     # the parameter values
    sweep[1]                         # the results of the second run (ModeResults)
    sweep.stack(lambda r: r.frequencies[:3])     # (n_values, 3) array

Each run uses model.with_parameters(name=value): the user's model is not changed. Every variant is meshed
next to the model's mesh (<mesh stem>_<name>_<value>.msh, reused while unchanged); a parameter that changes no
mesh input gives the same mesh as the model. A scene (3D) is meshed in <scene folder>/<name>_<value>/.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from loudpy.errors import ModelError, StateError
from loudpy.results.sweep import SweepResults

__all__ = ["Sweep", "SweepResults"]


class Sweep:
    """Runs `study` (an lp.Modes, lp.FrequencyResponse, … on a meshed model) for every value of `parameter`."""

    def __init__(self, study, parameter: str, values, mesh_folder=None):
        model = study.model
        if parameter not in model.parameters:
            raise ModelError(f"the model has no parameter {parameter!r}; parameters: {sorted(model.parameters)}")
        if len(np.atleast_1d(values)) == 0:
            raise ModelError("lp.Sweep needs at least one value")
        self.study, self.parameter = study, parameter
        self.values = [float(v) for v in np.atleast_1d(values)]
        self.mesh_folder = Path(mesh_folder) if mesh_folder is not None else None

    def _mesh_path(self, value) -> Path:
        if getattr(self.study, "is_scene", False):                       # a scene: one folder per value
            if self.study.model.folder is None and self.mesh_folder is None:
                raise StateError("mesh the scene first (scene.mesh(folder)) or give lp.Sweep(…, mesh_folder=)")
            folder = self.mesh_folder or self.study.model.folder
            return folder / f"{self.parameter}_{value:g}"
        base = self.study.model._mesh_path
        if base is None and self.mesh_folder is None:
            raise StateError("mesh the model first (model.mesh(path)) or give lp.Sweep(…, mesh_folder=)")
        folder = self.mesh_folder or base.parent
        stem = base.stem if base is not None else "mesh"
        return folder / f"{stem}_{self.parameter}_{value:g}.msh"

    def solve(self, *args, **kwargs) -> SweepResults:
        """Calls study.solve(*args, **kwargs) on each variant of the model."""
        results = []
        for v in self.values:
            variant = self.study.model.with_parameters(**{self.parameter: v})
            variant.mesh(self._mesh_path(v))
            study = copy.copy(self.study)
            study.model, study.results = variant, None
            results.append(study.solve(*args, **kwargs))
        self.results = SweepResults(self.parameter, self.values, results)
        return self.results
