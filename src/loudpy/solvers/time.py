"""
Time integration (no physics): generalised-α (Chung–Hulbert) with Newton–Raphson iterations
for a nonlinear structural system

    M a + C(u) v + F_int(u) = F_ext(t),

given by a callback that returns, at a full-size displacement u, the tangent stiffness K_T, the
damping C and the internal forces F_int. The unknowns are the free DOFs `free` of the full
numbering; blocked DOFs stay 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import warnings

import numpy as np

from loudpy.errors import ConvergenceWarning

from loudpy.solvers.linear import factorize


@dataclass(frozen=True)
class GeneralizedAlpha:
    """Parameters from the spectral radius at infinity rho_inf (1: no dissipation)."""
    rho_inf: float = 0.8

    @property
    def alpha_m(self):
        """α_m of the generalized-α scheme (from ρ∞)."""
        return (2.0 * self.rho_inf - 1.0) / (self.rho_inf + 1.0)

    @property
    def alpha_f(self):
        """α_f of the generalized-α scheme (from ρ∞)."""
        return self.rho_inf / (self.rho_inf + 1.0)

    @property
    def gamma(self):
        """Newmark γ = ½ − α_m + α_f."""
        return 0.5 - self.alpha_m + self.alpha_f

    @property
    def beta(self):
        """Newmark β = ¼ (1 − α_m + α_f)²."""
        return 0.25 * (1.0 - self.alpha_m + self.alpha_f) ** 2


def free_dof_reducer(template, free):
    """
    A → A[free, :][:, free] for matrices sharing the CSR pattern of `template`, by a precomputed
    index map (falls back to slicing for another pattern).
    """
    tmpl = template.tocsr().copy()
    tmpl.data = np.arange(1, tmpl.nnz + 1, dtype=np.float64)
    red = tmpl[free, :][:, free].tocsc()
    red_idx = red.data.astype(np.int64) - 1

    def reduce(A_full):
        A_full = A_full.tocsr()
        if A_full.nnz != tmpl.nnz or not np.array_equal(A_full.indices, tmpl.indices):
            return A_full[free, :][:, free].tocsc()
        out = red.copy()
        out.data = A_full.data[red_idx]
        return out
    return reduce


def generalized_alpha_newton(tangent, M_red, reduce, free, n_dof, F_base, signal, dt, n_steps,
                             rho_inf=0.8, nr_max_iter=10, nr_tol=1e-5, reuse_tangent=True,
                             reuse_max_iter=3, log=None):
    """
    Integrate from rest over n_steps of dt, with F_ext(t_k) = F_base · Re(signal[k]).

    tangent(u_full) → (K_T_full, C_full, F_int_full);  M_red: reduced mass;  reduce: A_full →
    A_red (free_dof_reducer).
    reuse_tangent: modified Newton — the LU of the effective matrix is kept across iterations and
    time steps and refactorised only when an iteration count exceeds reuse_max_iter (the
    convergence test nr_tol is unchanged, so the solution has the same accuracy).

    Returns time (n_steps+1,), U, V, A (n_steps+1, n_dof) and the number of factorisations.
    """
    if log is None:                                   # progress messages: logger "loudpy"
        from loudpy._log import info as log
    ga = GeneralizedAlpha(rho_inf)
    alpha_m, alpha_f, gam, beta = ga.alpha_m, ga.alpha_f, ga.gamma, ga.beta
    c1 = 1.0 / (beta * dt ** 2)
    c2 = gam / (beta * dt)
    n_free = len(free)
    lu_eff, n_factor = None, 0

    u_n = np.zeros(n_free, dtype=np.float64)
    v_n = np.zeros(n_free, dtype=np.float64)
    a_n = np.zeros(n_free, dtype=np.float64)
    results_u = [np.zeros(n_dof, dtype=np.float64)]
    results_v = [np.zeros(n_dof, dtype=np.float64)]
    results_a = [np.zeros(n_dof, dtype=np.float64)]
    time_array = [0.0]

    for step in range(n_steps):
        t_n1 = (step + 1) * dt

        F_ext_n = F_base * float(np.real(signal[step]))
        F_ext_n1 = F_base * float(np.real(signal[step + 1]))
        F_ext_mid = (1.0 - alpha_f) * F_ext_n1 + alpha_f * F_ext_n

        # predictor
        u_iter = u_n + dt * v_n + dt ** 2 * (0.5 - beta) * a_n
        v_iter = v_n + dt * (1.0 - gam) * a_n
        a_iter = np.zeros_like(a_n)

        # Newton–Raphson
        res_norm = np.inf
        for k in range(nr_max_iter):
            u_mid = (1.0 - alpha_f) * u_iter + alpha_f * u_n
            v_mid = (1.0 - alpha_f) * v_iter + alpha_f * v_n
            a_mid = (1.0 - alpha_m) * a_iter + alpha_m * a_n

            u_full_mid = np.zeros(n_dof, dtype=np.float64)
            u_full_mid[free] = u_mid
            K_T_full, C_full_it, F_int_full = tangent(u_full_mid)

            C_red = reduce(C_full_it)
            F_int_red = F_int_full[free]
            Residual = (F_ext_mid
                        - F_int_red
                        - M_red @ a_mid
                        - C_red @ v_mid)
            res_norm = np.linalg.norm(Residual)
            f_norm = np.linalg.norm(F_ext_mid)

            if step == 0 and k == 0:
                K_red = reduce(K_T_full)
                K_eff_diag0 = ((1.0 - alpha_f) * K_red
                               + (1.0 - alpha_f) * c2 * C_red
                               + (1.0 - alpha_m) * c1 * M_red).diagonal()
                log(f"  K_eff diag mean={K_eff_diag0.mean():.6e}  min={K_eff_diag0.min():.6e}")

            if res_norm < nr_tol * (f_norm + 1e-12):
                break

            if lu_eff is None or not reuse_tangent or k >= reuse_max_iter:
                K_red = reduce(K_T_full)
                K_eff = ((1.0 - alpha_f) * K_red
                         + (1.0 - alpha_f) * c2 * C_red
                         + (1.0 - alpha_m) * c1 * M_red)
                lu_eff = factorize(K_eff)
                n_factor += 1

            delta_u = np.real(lu_eff.solve(Residual))
            if step == 0 and k == 0:
                log(f"  delta_u    max={np.max(np.abs(delta_u)):.6e}  norm={np.linalg.norm(delta_u):.6e}")

            u_iter += delta_u
            a_iter += c1 * delta_u
            v_iter += c2 * delta_u
        else:
            warnings.warn(f"Newton–Raphson did not converge at step {step} (residual {res_norm:.3e} > {nr_tol:g}); "
                          f"the step is kept", ConvergenceWarning, stacklevel=2)

        u_n, v_n, a_n = u_iter, v_iter, a_iter
        u_full_step = np.zeros(n_dof, dtype=np.float64)
        v_full_step = np.zeros(n_dof, dtype=np.float64)
        a_full_step = np.zeros(n_dof, dtype=np.float64)
        u_full_step[free] = u_n
        v_full_step[free] = v_n
        a_full_step[free] = a_n
        results_u.append(u_full_step)
        results_v.append(v_full_step)
        results_a.append(a_full_step)
        time_array.append(t_n1)

        if step % 10 == 0:
            log(f"  step {step:>5}/{n_steps}  t={t_n1:.4f}s  |u|_max={np.max(np.abs(u_n)):.3e}  "
                f"(NR iters: {k + 1})")

    log(f"Nonlinear Generalized-Alpha sweep complete ({n_factor} factorisations).")
    return (np.asarray(time_array), np.asarray(results_u), np.asarray(results_v),
            np.asarray(results_a), n_factor)
