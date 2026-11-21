"""
Parametric sweeps: the study run on a copy of the model for every combination of parameter values (as COMSOL's
Parametric Sweep: the parameters are those of the model, lp.Parameters; geometry and items rebuild per run).

    model = lp.Model("plate.geo", parameters=lp.Parameters(E=2.1e11, h=0.002))   # h: a DefineConstant of the .geo
    model.add(lp.Solid("plate", steel, E=lp.P("E"), …), …)
    model.mesh("plate.msh")

    lp.Sweep(lp.Modes(model), "E", [1.9e11, 2.1e11])                       # one parameter
    lp.Sweep(lp.Modes(model), {"E": [1.9e11, 2.1e11], "h": [1e-3, 2e-3]})  # all combinations (2 × 2 runs)
    lp.Sweep(lp.Modes(model), {"E": [1.9e11, 2.1e11], "h": [1e-3, 2e-3]}, combine="zip")   # pairs (2 runs)

    sweep = … .solve(f_max=3000)          # SweepResults
    sweep.table()                         # the runs (pandas)
    sweep.evaluate(lambda r: r.frequencies[:3], name="f")    # lp.Data over E × h (× mode)
    sweep.at(E=2.1e11, h=2e-3)            # one run's results
    sweep.save("sweep.h5")                # one file; lp.load gives the SweepResults back

Each run uses model.with_parameters(**run): the user's model is not changed. Every variant is meshed next to the
model's mesh (<mesh stem>_<name>_<value>….msh, reused while unchanged); a run that changes no mesh input gives
the same mesh as the model. A scene (3D) is meshed in <scene folder>/<name>_<value>…/.
Geometry parameters: arguments of a builder builder(occ, p), and DefineConstant[…] entries of a .geo.
"""
from __future__ import annotations

import copy
import itertools
from pathlib import Path

import numpy as np

from loudpy.errors import ModelError, StateError
from loudpy.results.sweep import SweepResults

__all__ = ["Sweep", "SweepResults"]


class Sweep:
    """Runs `study` (an lp.Modes, lp.FrequencyResponse, … on a meshed model) for every run of the parameters:
    Sweep(study, name, values) or Sweep(study, {name: values, …}, combine="grid" | "zip")."""

    def __init__(self, study, parameters, values=None, mesh_folder=None, combine: str = "grid"):
        model = study.model
        if isinstance(parameters, str):
            if values is None:
                raise ModelError("lp.Sweep(study, name, values): give the values")
            spec = {parameters: values}
        elif isinstance(parameters, dict):
            if values is not None:
                raise ModelError("lp.Sweep(study, {name: values, …}): the values are in the dict")
            spec = dict(parameters)
        else:
            raise ModelError("lp.Sweep(study, name, values) or lp.Sweep(study, {name: values, …})")
        if combine not in ("grid", "zip"):
            raise ModelError(f"lp.Sweep(combine=…): 'grid' (all combinations) or 'zip' (pairs), not {combine!r}")
        unknown = [k for k in spec if k not in model.parameters]
        if unknown:
            raise ModelError(f"the model has no parameter {unknown}; parameters: {sorted(model.parameters)}")
        lists = {k: [float(v) for v in np.atleast_1d(vs)] for k, vs in spec.items()}
        if any(len(v) == 0 for v in lists.values()):
            raise ModelError("lp.Sweep needs at least one value per parameter")
        if combine == "zip" and len({len(v) for v in lists.values()}) > 1:
            raise ModelError(f"lp.Sweep(combine='zip'): the value lists differ in length "
                             f"({ {k: len(v) for k, v in lists.items()} })")
        names = list(lists)
        if combine == "zip":
            runs = [dict(zip(names, vals)) for vals in zip(*lists.values())]
        else:
            runs = [dict(zip(names, vals)) for vals in itertools.product(*lists.values())]
        self.study, self.names, self.runs = study, names, runs
        self.combine = "single" if len(names) == 1 else combine
        self.mesh_folder = Path(mesh_folder) if mesh_folder is not None else None

    @property
    def parameter(self) -> str:
        """The first swept parameter."""
        return self.names[0]

    @property
    def values(self) -> list:
        """The values of the first parameter, per run."""
        return [r[self.names[0]] for r in self.runs]

    def _tag(self, run) -> str:
        return "_".join(f"{k}_{v:g}" for k, v in run.items())

    def _mesh_path(self, run) -> Path:
        if getattr(self.study, "is_scene", False):                       # a scene: one folder per run
            if self.study.model.folder is None and self.mesh_folder is None:
                raise StateError("mesh the scene first (scene.mesh(folder)) or give lp.Sweep(…, mesh_folder=)")
            folder = self.mesh_folder or self.study.model.folder
            return folder / self._tag(run)
        base = self.study.model._mesh_path
        if base is None and self.mesh_folder is None:
            raise StateError("mesh the model first (model.mesh(path)) or give lp.Sweep(…, mesh_folder=)")
        folder = self.mesh_folder or base.parent
        stem = base.stem if base is not None else "mesh"
        return folder / f"{stem}_{self._tag(run)}.msh"

    def solve(self, *args, **kwargs) -> SweepResults:
        """Calls ``study.solve(*args, **kwargs)`` on each variant of the model (one run after the other)."""
        import logging
        log = logging.getLogger("loudpy")
        results = []
        for k, run in enumerate(self.runs):
            log.info(f"sweep run {k + 1}/{len(self.runs)}: {run}")
            variant = self.study.model.with_parameters(**run)
            variant.mesh(self._mesh_path(run))
            study = copy.copy(self.study)
            study.model, study.results = variant, None
            results.append(study.solve(*args, **kwargs))
        self.results = SweepResults(self.names, None, results, combine=self.combine, runs=self.runs)
        return self.results
