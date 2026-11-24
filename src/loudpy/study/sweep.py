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

Each run uses model.with_parameters(**run): the user's model is not changed. A run that changes no mesh input (a
material value, a damping coefficient …) uses the model's own mesh; the others are meshed next to it
(<mesh stem>_<name>_<value>….msh, reused while unchanged). A scene (3D) is meshed in <scene folder>/<name>_<value>…/.
Geometry parameters: arguments of a builder builder(occ, p), and DefineConstant[…] entries of a .geo.

A frequency response driven by a circuit (or a voltage) whose swept parameters reach only the electrical side — the
circuit's values (a crossover's fc, an EQ's gain, a resistor) or the drivers' motors — runs by redrive: the structure
and the air are solved once (the first run), every other run is that result driven by the circuit with its values
(Results.redrive: exact, in moments). redrive=True asks for it (refused, saying why, when a swept parameter reaches the
geometry, an item, a material or a scene's cabinet and air), False never; None (default): when it applies.
"""
from __future__ import annotations

import copy
import itertools
from pathlib import Path

import numpy as np

from loudpy.errors import ModelError, StateError
from loudpy.results.sweep import SweepResults

__all__ = ["Sweep", "SweepResults"]


def mechanical_parameters(target) -> set[str]:
    """The parameters a model or scene feeds to its structure and air: those of its items (a coil's motor left out:
    electrical), of its geometry (a .geo's DefineConstant, all of them for a builder taking the parameters), of a
    scene's cabinet (all of them for a cabinet written as a builder) and air settings."""
    import dataclasses
    from loudpy.model.parameters import names_in
    out: set = set()
    if hasattr(target, "regions"):                                    # a 2D model
        for it in [*target.regions, *target.boundaries, *target.mesh_controls]:
            fields = [f.name for f in dataclasses.fields(it)] if dataclasses.is_dataclass(it) else []
            out |= names_in([getattr(it, f) for f in fields if f != "motor"])
        src = target.source
        out |= set(src.geo_constants)
        if src.takes_parameters:
            out |= set(target.parameters)
        return out
    cab = getattr(target, "cabinet", None)                            # a scene
    if dataclasses.is_dataclass(cab):
        out |= names_in(cab)
    elif cab is not None:
        out |= set(target.parameters)
    for spec in (getattr(target, "_cavity_spec", None), getattr(target, "_open_spec", None),
                 getattr(target, "_walls_spec", None)):
        out |= names_in(spec)
    return out


class Sweep:
    """Runs `study` (an lp.Modes, lp.FrequencyResponse, … on a meshed model) for every run of the parameters:
    Sweep(study, name, values) or Sweep(study, {name: values, …}, combine="grid" | "zip")."""

    def __init__(self, study, parameters, values=None, mesh_folder=None, combine: str = "grid",
                 redrive: bool | None = None):
        model = study.model
        self.redrive = redrive
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

    def electrical_only(self) -> tuple[bool, str]:
        """Whether the swept parameters reach only the electrical side of a frequency response (its circuit's values,
        the drivers' motors) — then the runs are redrives of one solve — and why not."""
        from loudpy.circuit import Circuit
        from loudpy.model.parameters import names_in
        from loudpy.physics.electric import motors_of
        study, swept = self.study, set(self.names)
        if type(study).__name__ != "FrequencyResponse":
            return False, f"a {type(study).__name__} is solved again for every run"
        target = study.model
        electric = set(study.drive.parameters) if isinstance(study.drive, Circuit) else set()
        for m in motors_of(target).values():
            electric |= names_in(m)
        outside = sorted(swept - electric)
        if outside:
            return False, f"{outside}: not values of the circuit or of a motor"
        used = sorted(swept & mechanical_parameters(target))
        if used:
            return False, f"{used} also reach the structure or the air (geometry, items, cabinet …)"
        return True, ""

    def solve(self, *args, **kwargs) -> SweepResults:
        """Calls ``study.solve(*args, **kwargs)`` on each variant of the model (one run after the other); by redrive
        when the swept parameters reach only the circuit (module docstring)."""
        if self.redrive is not False:
            ok, why = self.electrical_only()
            if self.redrive and not ok:
                raise ModelError(f"lp.Sweep(redrive=True): {why}")
            if ok:
                return self._solve_by_redrive(*args, **kwargs)
        import logging
        log = logging.getLogger("loudpy")
        results = []
        from loudpy import progress
        for k, run in enumerate(self.runs):
            progress.step("sweep run", k + 1, len(self.runs))
            log.info(f"sweep run {k + 1}/{len(self.runs)}: {run}")
            results.append(self._run(run, *args, **kwargs))
        self.results = SweepResults(self.names, None, results, combine=self.combine, runs=self.runs)
        return self.results

    def _run(self, run, *args, **kwargs):
        """The study solved on the model with the run's values."""
        variant = self.study.model.with_parameters(**run)
        base = getattr(self.study.model, "_mesh_path", None)
        if (not getattr(self.study, "is_scene", False) and self.mesh_folder is None and base is not None
                and variant.mesh_reusable(base)):
            variant.mesh(base)                         # the run changes no mesh input: the model's own mesh
        else:
            variant.mesh(self._mesh_path(run))
        study = copy.copy(self.study)
        study.model, study.results = variant, None
        return study.solve(*args, **kwargs)

    def _solve_by_redrive(self, *args, **kwargs) -> SweepResults:
        """The first run solved, the others that result driven by the circuit with their values (no new solve)."""
        import logging
        from loudpy import progress
        from loudpy.physics.electric import as_circuit, motors_of
        from loudpy.results.description import ModelDescription
        log = logging.getLogger("loudpy")
        target = self.study.model
        log.info(f"sweep of {self.names} by redrive: the structure and the air solved once, the circuit "
                 f"{len(self.runs)} times")
        progress.step("sweep run", 1, len(self.runs))
        first = self._run(self.runs[0], *args, **kwargs)
        if not first.redrivable:
            raise ModelError("the first run keeps no unit response per coil: lp.Sweep(…, redrive=False)")
        coils = [d.name for d in target.drivers] if getattr(self.study, "is_scene", False) else [None]
        results = [first]
        for k, run in enumerate(self.runs[1:], 2):
            progress.step("sweep run", k, len(self.runs))
            params = target.parameters.with_(**run)
            res = first.redrive(as_circuit(self.study.drive, coils, motors_of(target), params))
            if first.model is not None:                                # (its description: this run's values)
                res.model = ModelDescription({**first.model.data, "parameters": dict(params)}, first.model.files)
            results.append(res)
        self.results = SweepResults(self.names, None, results, combine=self.combine, runs=self.runs)
        return self.results
