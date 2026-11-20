"""
SweepResults: the results of a parametric sweep (lp.Sweep), one complete result per parameter value.

    sweep.parameter, sweep.values        the swept parameter and its values
    sweep[i], iter(sweep)                the result of each run (FrequencyResults, ModeResults …)
    sweep.stack(lambda r: r.frequencies[:3])     (n_values, …) array
    sweep.save("sweep.h5")               one result file (format 3: /runs/<i>), read back by lp.load
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["SweepResults"]


class SweepResults:
    """The results of a sweep: `parameter`, `values` and one results object per value."""
    is_sweep = True
    kind = "sweep"

    def __init__(self, parameter: str, values, results: list, source=None):
        self.parameter = parameter
        self.values = np.asarray(values, float)
        self.results = list(results)
        self.source = source

    def __len__(self):
        return len(self.results)

    def __getitem__(self, i):
        return self.results[i]

    def __iter__(self):
        return iter(self.results)

    def stack(self, fn) -> np.ndarray:
        """np.stack of fn(results) over the values (fn gives arrays of one shape)."""
        return np.stack([np.asarray(fn(r)) for r in self.results])

    def save(self, path, stem: str = "sweep"):
        """
        One result file with every run (path ending in .h5; format 3, read back by lp.load). A folder
        (no .h5 suffix): one file per value, <folder>/<stem>_<parameter>_<value>.h5 (former layout).
        """
        path = Path(path)
        if path.suffix.lower() in (".h5", ".hdf5"):
            from loudpy.results.schema import write
            write(self, path)
            return path
        path.mkdir(parents=True, exist_ok=True)
        out = []
        for v, r in zip(self.values, self.results):
            p = path / f"{stem}_{self.parameter}_{v:g}.h5"
            r.meta.update(sweep_parameter=self.parameter, sweep_value=float(v))
            r.save(p)
            out.append(p)
        return out

    def __repr__(self):
        return f"SweepResults({self.parameter}: {len(self)} values {self.values.min():g} … {self.values.max():g})"
