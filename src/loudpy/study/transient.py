"""
Transient: time response of the structure to a force signal on the coil.

    res = lp.Transient(model, drive=lp.Force(signal, fs=48_000)).solve()

Geometrically nonlinear (total Lagrangian) axisymmetric structure with Rayleigh damping,
generalised-α time integration (loudpy.solvers.time). signal[k] is the force [N] at t = k / fs;
the run starts at rest and covers len(signal) − 1 steps (or n_steps / duration).
"""
from __future__ import annotations

import numpy as np

from loudpy.model.drives import Force
from loudpy.physics.registry import select
from loudpy.results.results import TimeResults
from loudpy.study import records
from loudpy.study.base import Study


class Transient(Study):
    """Time response (generalised-alpha, Newton–Raphson for the nonlinear coil) to a Force signal.

        res = lp.Transient(model, drive=lp.Force(signal, fs=48000)).solve(duration=0.1)"""
    def __init__(self, model, drive: Force, nonlinear: bool = True, rho_inf: float = 0.8,
                 nr_tol: float = 1e-5, nr_max_iter: int = 10):
        super().__init__(model)
        if not (isinstance(drive, Force) and drive.is_signal and drive.fs):
            raise ValueError("Transient needs lp.Force(signal, fs=...)")
        if not nonlinear:
            raise NotImplementedError("only the nonlinear (total Lagrangian) transient is available; for "
                                      "small amplitudes it gives the linear response")
        self.drive, self.rho_inf, self.nr_tol, self.nr_max_iter = drive, rho_inf, nr_tol, nr_max_iter

    def solve(self, n_steps: int | None = None, duration: float | None = None) -> TimeResults:
        from loudpy.Studies import TimeStudy
        sig = np.asarray(self.drive.amplitude, float)
        dt = 1.0 / float(self.drive.fs)
        if duration is not None:
            n_steps = int(round(duration / dt))
        n_steps = len(sig) - 1 if n_steps is None else int(n_steps)
        if n_steps > len(sig) - 1:
            raise ValueError(f"the signal has {len(sig)} samples: at most {len(sig) - 1} steps")
        problem = self._problem()
        coil = select(problem.fem_objects, "coil")
        if not coil:
            raise ValueError("the model has no coil boundary (lp.Coil)")
        direction = "_z_" if tuple(coil[0].direction) == (1,) else "_r_"
        engine = TimeStudy(problem)
        engine.assemble_init()
        engine.solve_time_domain_rayleigh_nl(force_amplitude=1.0, force_direction=direction, force_signal=sig,
                                             dt=dt, n_steps=n_steps, nr_max_iter=self.nr_max_iter,
                                             nr_tol=self.nr_tol, rho_inf=self.rho_inf)
        blocks = records.time_blocks(engine._results, engine.dof_maps(), engine.fem_objects)
        meta = {"study": "Transient", "fs": float(self.drive.fs), "geometry": str(self.model.geometry)}
        self.results = records.time_results(blocks, meta, records.boundaries_of(engine._interface_meta))
        return self.results
