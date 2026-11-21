"""
SweepResults: the results of a parametric sweep (lp.Sweep), one complete result per run.

    sweep.names                          the swept parameters, e.g. ["E", "h"]
    sweep.runs                           one {name: value} per run
    sweep.combine                        "grid" (all combinations), "zip" (specified combinations)
    sweep[i], iter(sweep)                the result of each run (FrequencyResults, ModeResults …)
    sweep.table()                        the runs as a table (pandas DataFrame)
    sweep.evaluate(lambda r: r.spl(at=(0, 0.3)))      lp.Data over the parameters (and the result's own axes)
    sweep.evaluate(lambda r: r.frequencies[:3], name="f")
    sweep.save("sweep.h5")               one result file (format 3: /runs/<i>), read back by lp.load

evaluate gives labelled data: on a grid one axis per parameter (E × h × f …), otherwise a "run" axis with the
parameter values attached; it plots and exports like any lp.Data (.export("x.nc") for netCDF, .csv, .mat …).
"""
from __future__ import annotations

import json
from typing import Any
from pathlib import Path

import numpy as np

from loudpy.errors import ResultsError

__all__ = ["SweepResults"]


class SweepResults:
    """The results of a sweep: the parameter values of every run and one results object per run."""
    is_sweep = True
    kind = "sweep"

    def __init__(self, parameter, values=None, results: list | None = None, source=None, combine: str = "single",
                 runs: list | None = None):
        """SweepResults(name, values, results) (one parameter), or SweepResults(names, None, results,
        combine=…, runs=[{name: value}, …])."""
        self.results = list(results or [])
        self.source = source
        if runs is not None:
            self.names = list(parameter)
            self.runs = [{k: float(r[k]) for k in self.names} for r in runs]
            self.combine = combine
        else:
            self.names = [str(parameter)]
            self.runs = [{self.names[0]: float(v)} for v in np.atleast_1d(values)]
            self.combine = "single"
        if len(self.runs) != len(self.results):
            raise ResultsError(f"{len(self.runs)} runs for {len(self.results)} results")

    # ── the parameters ──────────────────────────────────────────────────────
    @property
    def parameter(self) -> str:
        """The swept parameter (the first one of a multi-parameter sweep)."""
        return self.names[0]

    @property
    def values(self) -> np.ndarray:
        """Parameter values per run: (n_runs,) for one parameter, (n_runs, n_parameters) otherwise."""
        v = np.array([[r[k] for k in self.names] for r in self.runs], float)
        return v[:, 0] if len(self.names) == 1 else v

    @property
    def axes(self) -> dict:
        """Grid axes {name: sorted unique values} (a grid sweep, or one parameter); {} for zipped runs."""
        if self.combine == "zip" and len(self.names) > 1:
            return {}
        return {k: np.unique([r[k] for r in self.runs]) for k in self.names}

    def table(self):
        """The runs as a pandas DataFrame (one row per run, one column per parameter)."""
        import pandas as pd
        return pd.DataFrame(self.runs, index=pd.RangeIndex(len(self.runs), name="run"))

    # ── access ──────────────────────────────────────────────────────────────
    def __len__(self):
        return len(self.results)

    def __getitem__(self, i):
        return self.results[i]

    def __iter__(self):
        return iter(self.results)

    def at(self, **values):
        """The result of the run with these parameter values (exact match)."""
        for r, res in zip(self.runs, self.results):
            if all(np.isclose(r[k], float(v), rtol=1e-12, atol=0) for k, v in values.items()):
                return res
        raise ResultsError(f"no run with {values} (runs: {self.runs})")

    def stack(self, fn) -> np.ndarray:
        """np.stack of fn(result) over the runs (plain arrays; evaluate() gives labelled data)."""
        return np.stack([np.asarray(fn(r)) for r in self.results])

    def evaluate(self, fn, name: str | None = None):
        """
        lp.Data of fn(result) over the runs. fn returns an lp.Data (its axes are kept after the parameter
        axes), an array or a number (quantity `name`). A grid sweep gives one axis per parameter; zipped runs
        a "run" axis (the parameter values per run in .meta["runs"]); one parameter its own axis.
        """
        from loudpy.results.data import Data
        outs = [fn(r) for r in self.results]
        first = outs[0]
        q: Any
        if isinstance(first, Data):
            inner = [np.asarray(o.values) for o in outs]
            inner_coords, q, meta = dict(first.coords), first.quantity, dict(first.meta)
        else:
            inner = [np.asarray(o) for o in outs]
            inner_coords = {f"axis{k}": np.arange(n) for k, n in enumerate(inner[0].shape)}
            q, meta = (name or "value"), {}
        if any(v.shape != inner[0].shape for v in inner):
            raise ResultsError("evaluate: fn gives values of different shapes over the runs")
        stacked = np.stack(inner)
        meta = {**meta, "sweep": self.names, "combine": self.combine, "runs": self.runs}
        axes = self.axes
        if axes and len(self.names) > 1:
            shape = tuple(len(a) for a in axes.values())
            if int(np.prod(shape)) != len(self.runs):
                raise ResultsError("evaluate: the runs do not fill the grid")
            out = np.full(shape + inner[0].shape, np.nan, dtype=np.result_type(stacked, float))
            for r, v in zip(self.runs, stacked):
                idx = tuple(int(np.searchsorted(axes[k], r[k])) for k in self.names)
                out[idx] = v
            coords = {**{k: axes[k] for k in self.names}, **inner_coords}
            return Data(out, coords, q, meta=meta)
        if len(self.names) == 1:
            values = np.array([r[self.names[0]] for r in self.runs])
            return Data(stacked, {self.names[0]: values, **inner_coords}, q, meta=meta)
        return Data(stacked, {"run": np.arange(len(self.runs)), **inner_coords}, q, meta=meta)

    # ── files ───────────────────────────────────────────────────────────────
    def save(self, path, stem: str = "sweep"):
        """
        One result file with every run (path ending in .h5; format 3, read back by lp.load). A folder
        (no .h5 suffix): one file per run, <folder>/<stem>_<name>_<value>….h5.
        """
        path = Path(path)
        if path.suffix.lower() in (".h5", ".hdf5"):
            from loudpy.results.schema import write
            write(self, path)
            return path
        path.mkdir(parents=True, exist_ok=True)
        out = []
        for run, r in zip(self.runs, self.results):
            p = path / (stem + "".join(f"_{k}_{v:g}" for k, v in run.items()) + ".h5")
            r.meta.update(sweep=json.dumps(run))
            r.save(p)
            out.append(p)
        return out

    def __repr__(self):
        span = ", ".join(f"{k} {min(r[k] for r in self.runs):g} … {max(r[k] for r in self.runs):g}" for k in self.names)
        return f"SweepResults({len(self)} runs, {self.combine}: {span})"
