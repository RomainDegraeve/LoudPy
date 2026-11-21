# Benchmarks — LoudPy solver speed-ups

`bench_solvers.py <case> <tag>` times one study on the HPNEW driver and saves its results.
`compare_solvers.py <tag>` compares a run with `baseline`: speed-up and maximum relative difference.
`LOUDPY_FAST_SOLVERS=0` restores the former solvers. That gives an exact A/B in the same code.

| case | what is timed | before | after | speed-up | difference |
|---|---|---|---|---|---|
| `fsi` | `lp.FrequencyResponse` (structure + air), force drive, FSI + PML, 59k DOF, 40 frequencies | 42.1 s | 9.9 s | **4.3×** | 1e-9 |
| `fsi_elec` | same, voltage drive (electrical DOF) | 36.6 s | 10.8 s | **3.4×** | 2e-9 |
| `meca` | `lp.FrequencyResponse` (structure), 9k DOF, 200 frequencies | 4.3 s | 4.2 s | 1.0× | 0 |
| `eigen` | `lp.Modes` (multishift), 68 modes ≤ 20 kHz | 33.6 s | 14.9 s | **2.3×** | Δf/f 4e-9 |
| `fsi_eigen` | `lp.Modes(physics="coupled")` (exact DtN), 76 coupled modes ≤ 3.2 kHz | 79.8 s | 28.0 s | **2.8×** | Δf/f 2e-5, Δζ 9e-6 |
| `time` | `lp.Transient`, 150 steps at 48 kHz | 13.8 s | 6.5 s | **2.1×** | 2e-7 (Newton tolerance) |

## What changed
The solvers are in `src/loudpy/solvers/` (`linear.py`, `eigen.py`, `time.py`). The timings were measured on the former study classes, which used the same solvers.

- **`factorize`:** SuperLU with a minimum-degree ordering on Aᵀ+A, in symmetric mode with threshold pivoting.
  - It is checked on the backward error of a test solve. If that check fails, it falls back to SuperLU's default.
  - FSI matrix: 0.5 s instead of 1.6–2.2 s per factorisation, with half the fill.
  - The same ordering without symmetric mode is about 60× slower, so the mode matters.
- **`SweepSolver`**, used by `lp.FrequencyResponse`:
  - The LU of an earlier frequency preconditions GMRES at the next ones: 7–18 iterations instead of a factorisation.
  - A solve is accepted on its true residual (≤ 1e-8). If GMRES does not get there, the solver refactorises.
  - The ordering and the reuse are both chosen by measurement, so a small system (`meca`) is never slowed down.
  - Turn reuse off with `lp.FrequencyResponse(..., reuse_factorization=False)`. `study.solver_stats` reports what was done.
- **`QEPShiftInvert`**, used by the eigen shift-invert: (A − σB)⁻¹B of the state-space QEP is built from an LU of the n×n matrix Q(σ) = K + σC + σ²M, instead of the 2n×2n block matrix.
- **ARPACK** (`solvers.eigen.arpack_eigs`): the Krylov space is 2.5k instead of 4k.
  - That gives the same eigenpairs at a lower cost per restart.
  - If it does not converge, it retries with 4k.
  - This is the main eigen gain: ARPACK's internal work dominated the run, not the factorisations.
- **Time domain:**
  - The internal forces are scattered with `np.bincount` instead of a Python double loop: tangent assembly went from 21 to 12 ms.
  - The reduction to free DOFs uses a precomputed index map.
  - Modified Newton (`reuse_tangent=True`): the LU of the effective matrix is kept across iterations and steps, because with dt ≪ 1/f it is dominated by the mass term. It is refactorised only when a step needs more than `reuse_max_iter` iterations. The convergence test is unchanged.

## Tried, not kept
- **Threads over frequencies:** only about 2× on the M2 (SuperLU only partly releases the GIL), and incompatible with the factorisation reuse.
- **ARPACK ncv = 2k + 1:** faster, but it changed the set of modes found (69 instead of 68).

## Note: iCloud
The repository sits in an iCloud-synced folder, and numba's cache there can hang or fail ("Operation canceled").
Run with a local cache:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache python3 ...
```
