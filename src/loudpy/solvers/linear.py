"""
fast_linalg.py — sparse direct solvers shared by the LoudPy studies.

  factorize(A)        LU with a fill-reducing ordering (minimum degree on Aᵀ+A, SuperLU symmetric
                      mode, threshold pivoting); checked on a test solve and, if not accurate
                      enough, redone with SuperLU's default (COLAMD, partial pivoting).
                      On the coupled FSI matrices: 3–4× faster, half the fill, same solution.
  SweepSolver         A(f) x = b for a sequence of nearby frequencies: the LU of an earlier
                      frequency is a GMRES preconditioner for the next ones (a few iterations
                      instead of a factorisation); refactorises when GMRES needs more than
                      max_iter iterations or does not converge.
  QEPShiftInvert      (A − σB)⁻¹ B of the state-space form of s²M + sC + K, from an LU of the
                      n×n matrix Q(σ) = K + σC + σ²M instead of the 2n×2n block matrix.

Set the environment variable LOUDPY_FAST_SOLVERS=0 to fall back to plain spsolve / splu.
"""
from __future__ import annotations

import os
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

ENABLED = os.environ.get("LOUDPY_FAST_SOLVERS", "1") != "0"
_SYM = dict(permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.1, options=dict(SymmetricMode=True))


def factorize(A, check_tol=1e-8):
    """Sparse LU (object with .solve) with a fill-reducing ordering, checked, with fallback."""
    A = sp.csc_matrix(A)
    if not ENABLED:
        return spla.splu(A)
    try:
        lu = spla.splu(A, **_SYM)
        # backward error of a test solve (independent of the conditioning of A)
        b = np.random.default_rng(0).standard_normal(A.shape[0]).astype(np.result_type(A.dtype, np.float64))
        err = np.linalg.norm(A @ lu.solve(b) - b) / np.linalg.norm(b)
        if np.isfinite(err) and err < check_tol:
            return lu
    except RuntimeError:                                  # singular in symmetric mode
        pass
    return spla.splu(A)


def solve(A, b):
    """One-off solve with factorize()."""
    return factorize(A).solve(b)


class SweepSolver:
    """
    Solves A x = b for successive, similar matrices (frequency sweeps), choosing by measurement
    what is fastest for this problem:

      - ordering: on the first matrix, the fast ordering (factorize) and SuperLU's default are
        both timed; the faster one is used for the whole sweep (small 2D systems often prefer
        the default);
      - reuse: later calls run GMRES (target relative residual rtol, at most max_iter iterations)
        preconditioned by the stored LU and warm-started from the previous solution; the result is
        accepted if its true relative residual is below `accept`, otherwise A is factorised and
        becomes the new preconditioner. Reuse is switched off for the sweep as soon as it is
        slower on average than a factorisation (counting failed attempts).
    reuse=False: factorise every call.
    """

    def __init__(self, rtol=1e-10, accept=1e-8, max_iter=25, reuse=True):
        self.rtol, self.accept, self.max_iter, self.reuse = rtol, accept, max_iter, reuse and ENABLED
        self.reset()

    def reset(self):
        """Forget the factorisation and the statistics."""
        self.lu, self.shape, self.x_prev = None, None, None
        self.n_factor, self.n_gmres, self.iters = 0, 0, 0
        self.fast_order, self.t_factor, self.t_reuse, self.n_try = None, None, 0.0, 0

    def _factor(self, A):
        t0 = time.perf_counter()
        if self.fast_order is None and ENABLED:            # first matrix: time both orderings
            lu_fast = factorize(A)
            t1 = time.perf_counter()
            lu_def = spla.splu(A)
            t2 = time.perf_counter()
            self.fast_order = (t1 - t0) <= (t2 - t1)
            lu, dt = (lu_fast, t1 - t0) if self.fast_order else (lu_def, t2 - t1)
        else:
            lu = factorize(A) if (self.fast_order and ENABLED) else spla.splu(A)
            dt = time.perf_counter() - t0
        self.t_factor = dt if self.t_factor is None else 0.5 * (self.t_factor + dt)
        self.n_factor += 1
        return lu

    def solve(self, A, b):
        """Solve A x = b, reusing the last LU as a GMRES preconditioner while it converges quickly, refactorising
        otherwise."""
        A = sp.csc_matrix(A)
        if self.shape != A.shape:
            self.reset()
            self.shape = A.shape
        if self.reuse and self.lu is not None:
            t0 = time.perf_counter()
            it = [0]
            M = spla.LinearOperator(A.shape, matvec=self.lu.solve, dtype=complex)
            x0 = self.x_prev if self.x_prev is not None and self.x_prev.shape == b.shape else None
            x, info = spla.gmres(A, b, x0=x0, M=M, rtol=self.rtol, atol=0.0, restart=self.max_iter,
                                 maxiter=1, callback=lambda r: it.__setitem__(0, it[0] + 1),
                                 callback_type="pr_norm")
            # accepted on the true residual (GMRES may stop just above rtol at the precision floor)
            ok = np.all(np.isfinite(x)) and np.linalg.norm(A @ x - b) <= self.accept * np.linalg.norm(b)
            self.t_reuse += time.perf_counter() - t0
            self.n_try += 1
            if ok:
                self.n_gmres += 1
                self.iters += it[0]
            # average cost of a reuse attempt (failures included) vs a factorisation
            if self.n_try >= 3 and self.t_reuse / max(self.n_gmres, 1) > 0.8 * self.t_factor:
                self.reuse = False
            if ok:
                self.x_prev = x
                return x
        self.lu = self._factor(A)
        x = self.lu.solve(b.astype(np.result_type(A.dtype, b.dtype)))
        self.x_prev = x
        return x


class QEPShiftInvert:
    """
    Applies S = (A − σB)⁻¹B for the state-space pencil of s²M + sC + K,
        A = [[0, I], [−K, −C]],   B = [[I, 0], [0, M]],
    using only an LU of Q(σ) = K + σC + σ²M (n×n):
        (A − σB) z = y,  y = (y1, y2)   ⇔   z1 = −Q⁻¹ (y2 + (C + σM) y1),   z2 = y1 + σ z1.
    """

    def __init__(self, K, M, C, sigma):
        self.n, self.sigma = K.shape[0], sigma
        self.M = sp.csr_matrix(M)
        self.CsM = sp.csr_matrix(C + sigma * M)
        if ENABLED:
            self.lu = factorize((K + sigma * C + sigma ** 2 * M).astype(complex))
        else:                                    # former path: LU of the 2n×2n block matrix
            In = sp.eye(self.n, format="csc", dtype=complex)
            self.lu2 = spla.splu(sp.bmat([[-sigma * In, In], [-K, -(C + sigma * M)]], format="csc"))

    def solve_shifted(self, y):
        """z = (A − σB)⁻¹ y."""
        n = self.n
        if not ENABLED:
            return self.lu2.solve(y.astype(complex))
        y1, y2 = y[:n], y[n:]
        z1 = -self.lu.solve(y2 + self.CsM @ y1)
        return np.concatenate([z1, y1 + self.sigma * z1])

    def __call__(self, x):
        """S x = (A − σB)⁻¹ B x."""
        n = self.n
        return self.solve_shifted(np.concatenate([x[:n], self.M @ x[n:]]))
