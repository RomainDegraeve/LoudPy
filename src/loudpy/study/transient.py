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
from loudpy.results.results import Block, TimeResults
from loudpy.study.base import Study
from loudpy.errors import ModelError


class Transient(Study):
    """Time response (generalised-alpha, Newton–Raphson for the nonlinear coil) to a Force signal.

        res = lp.Transient(model, drive=lp.Force(signal, fs=48000)).solve(duration=0.1)"""
    def __init__(self, model, drive: Force, nonlinear: bool = True, rho_inf: float = 0.8,
                 nr_tol: float = 1e-5, nr_max_iter: int = 10):
        super().__init__(model)
        if not (isinstance(drive, Force) and drive.is_signal and drive.fs):
            raise ModelError("Transient needs lp.Force(signal, fs=...)")
        if not nonlinear:
            raise NotImplementedError("only the nonlinear (total Lagrangian) transient is available; for "
                                      "small amplitudes it gives the linear response")
        self.drive, self.rho_inf, self.nr_tol, self.nr_max_iter = drive, rho_inf, nr_tol, nr_max_iter

    def _check_damping(self):
        """Hysteretic damping (a loss factor, complex stiffness) has no time-domain form: refuse it rather
        than run the region undamped."""
        from loudpy.model.damping import Hysteretic
        hyst = [r.name for r in self.model.regions if isinstance(getattr(r, "damping", None), Hysteretic)]
        if hyst:
            raise ModelError(f"Transient: {', '.join(map(repr, hyst))} use hysteretic damping (a loss factor), which "
                             f"exists only in frequency-domain studies; use lp.Rayleigh() (the material's fit) or "
                             f"lp.Rayleigh.from_loss_factor(eta) for a transient study")

    def solve(self, n_steps: int | None = None, duration: float | None = None) -> TimeResults:
        from loudpy.fem.assembly import cell_tabulation
        from loudpy.fem.kernels import _boundary_load_kernel
        from loudpy.physics.structural import MecaAssemblerNL
        from loudpy.physics.systems import meshes, nodal, System
        from loudpy.solvers.time import free_dof_reducer, generalized_alpha_newton
        sig = np.asarray(self.drive.amplitude, float)
        dt = 1.0 / float(self.drive.fs)
        if duration is not None:
            n_steps = int(round(duration / dt))
        n_steps = len(sig) - 1 if n_steps is None else int(n_steps)
        if n_steps > len(sig) - 1:
            raise ModelError(f"the signal has {len(sig)} samples: at most {len(sig) - 1} steps")
        self._check_damping()
        parts = self._parts()
        coils = select(parts, "coil")
        if not coils:
            raise ModelError("the model has no coil boundary (lp.Coil)")
        coil = coils[0]
        comp = 1 if tuple(coil.direction) == (1,) else 0               # axial (z) or radial (r) drive

        nl = MecaAssemblerNL(parts)
        n_dof = nl.matrix_size
        free = nl._free_dofs(parts)
        nl._free = free
        K_T0, M_full, _, _ = nl.assemble_tangent(np.zeros(n_dof))
        M_red = M_full[free, :][:, free].tocsc()
        reduce = free_dof_reducer(K_T0, free)                           # one CSR pattern for every tangent

        # unit coil force pattern on the free DOFs (consistent nodal forces, normalised to a sum of 1)
        pos = {int(t): i for i, t in enumerate(coil.node_tags)}
        F = np.zeros(n_dof)
        edges = np.asarray(coil.edges)
        ec = np.ascontiguousarray([[coil.node_coords[pos[int(t)]][:2] for t in edge] for edge in edges], dtype=float)
        N, dN, w = cell_tabulation(1, edges.shape[1])
        fe = np.empty(edges.shape)
        _boundary_load_kernel(ec, N, dN, w, np.ones((len(ec), len(w))), True, fe)   # ∫ N 2πr ds
        for edge, f_edge in zip(edges, fe):
            for k, t in enumerate(edge):
                F[2 * nl.tag_to_local[int(t)] + comp] += f_edge[k]
        F_red = F[free] / F[free].sum()

        def tangent(u_full):
            K_T, _, C, F_int = nl.assemble_tangent(u_full)
            return K_T, C, F_int

        t, U, V, A, _ = generalized_alpha_newton(
            tangent, M_red, reduce, free, n_dof, F_red, sig, dt, n_steps, rho_inf=self.rho_inf,
            nr_max_iter=self.nr_max_iter, nr_tol=self.nr_tol, reuse_tangent=True, reuse_max_iter=3)
        n = len(nl.unique_tags)
        fields = {"u": nodal(np.asarray(U), n, 2), "v": nodal(np.asarray(V), n, 2), "a": nodal(np.asarray(A), n, 2)}
        extra = {"force": np.real(sig)} if len(sig) == len(t) else {}
        params = dict(dt=dt, n_steps=n_steps, nr_max_iter=self.nr_max_iter, nr_tol=self.nr_tol, rho_inf=self.rho_inf,
                      t_total=float(t[-1]), force_amplitude=1.0, force_direction="_z_" if comp else "_r_")
        block = Block(meshes(System(nl, None, None, None), parts), np.asarray(t), fields, extra, params, "run0")
        meta = {"study": "Transient", "fs": float(self.drive.fs), "geometry": str(self.model.geometry)}
        self.results = TimeResults([block], meta, self.model.boundary_nodes())
        return self.results
