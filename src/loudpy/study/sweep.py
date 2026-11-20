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
mesh input gives the same mesh as the model.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from loudpy.errors import ModelError, StateError

__all__ = ["Sweep", "SweepResults"]


class SweepResults:
    """The results of a sweep: `parameter`, `values` and one results object per value."""

    def __init__(self, parameter: str, values, results: list):
        self.parameter = parameter
        self.values = np.asarray(values, float)
        self.results = list(results)

    def __len__(self):
        return len(self.results)

    def __getitem__(self, i):
        return self.results[i]

    def __iter__(self):
        return iter(self.results)

    def stack(self, fn) -> np.ndarray:
        """np.stack of fn(results) over the values (fn gives arrays of one shape)."""
        return np.stack([np.asarray(fn(r)) for r in self.results])

    def save(self, folder, stem: str = "sweep") -> list[Path]:
        """One result file per value: <folder>/<stem>_<parameter>_<value>.h5."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        out = []
        for v, r in zip(self.values, self.results):
            path = folder / f"{stem}_{self.parameter}_{v:g}.h5"
            r.meta.update(sweep_parameter=self.parameter, sweep_value=float(v))
            r.save(path)
            out.append(path)
        return out

    def __repr__(self):
        return f"SweepResults({self.parameter}: {len(self)} values {self.values.min():g} … {self.values.max():g})"


class Sweep:
    """Runs `study` (an lp.Modes, lp.FrequencyResponse, … on a meshed model) for every value of `parameter`."""

    def __init__(self, study, parameter: str, values, mesh_folder=None):
        model = study.model
        if getattr(study, "is_scene", False):
            raise NotImplementedError("lp.Sweep takes a study of an lp.Model (scenes: phase F3)")
        if parameter not in model.parameters:
            raise ModelError(f"the model has no parameter {parameter!r}; parameters: {sorted(model.parameters)}")
        if len(np.atleast_1d(values)) == 0:
            raise ModelError("lp.Sweep needs at least one value")
        self.study, self.parameter = study, parameter
        self.values = [float(v) for v in np.atleast_1d(values)]
        self.mesh_folder = Path(mesh_folder) if mesh_folder is not None else None

    def _mesh_path(self, value) -> Path:
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
