# loudpy roadmap — towards a professional, verified FE library

Approved on 2026-11-20. Goal: one consistent API across physics, studies, saving and plotting, ready to
use, with a test pipeline that checks accuracy (not only regressions). New physics (BEM, magnetics,
thermal, thermoviscous losses, porous media, anisotropy, motor nonlinearities, parametric sweeps,
PETSc/MPI) comes later.

Decisions: own generic FE core (no external FE library); elements: triangles P1–P3, quads Q1–Q2 (2D /
axisymmetric), tetrahedra P1–P3 (3D), shells for cabinet walls coupled directly to the air; both 2D
geometry workflows (gmsh curve loops and Physical Surfaces / STEP), names are labels only; SI units with
plausibility checks; error estimator + adaptive remeshing loop + convergence tool; raster 3D panels in
PDFs + interactive HTML viewer; material values marked "typical, unverified"; Python ≥ 3.13; full re-run
of the examples at the end; COMSOL comparison harness now, reference results later.

Cost unit: one *session* ≈ one long working block (3–5 h). Status: ✅ done · ◐ in progress · ○ to do.

| Phase | Content | Sessions | Status |
|---|---|---|---|
| A | Foundations: no import side effects, OpenMP, exceptions, SI checks, packaging, CI, verification tests | 1 | ✅ |
| B | Materials and air: library v2 (units, temperature, provenance), damping separate, air from (T, p, RH), one air for 2D and 3D | 1 | ✅ |
| C | Selections and geometry: selection objects, three geometry sources, nothing read from names, PML by selection, driver profiles from selections | 3 | ○ |
| D | Own generic FE core: Lagrange P1–P3 on lines / triangles / quads / tetrahedra, quadrature, spaces, generic kernels and assembly, curved (isoparametric) elements; old kernels removed | 5 | ○ |
| E | Physics interfaces and couplings: one contribution protocol, generic system builder, sign convention fixed, electrical circuit as physics, 3D on the same protocol | 4 | ○ |
| F | One model for 2D and 3D: `lp.Model(dim=…)`, typed mesh and solver settings, one remeshing rule, immutable model during solves, deterministic meshing by default, examples rewritten | 3 | ○ |
| G | Results format 3: one results family, self-contained files, public arrays, derived fields (stress, strain, velocity, intensity) and values (power, volume velocity), interpolation in tetrahedra, ParaView series | 3 | ○ |
| H | Error estimation and adaptivity: recovery and residual indicators, `lp.Adaptive`, `lp.convergence` | 3 | ○ |
| I | Shells: MITC3/MITC4 Reissner–Mindlin, direct coupling to the 3D air, wall-mode study, elastic-wall enclosure example | 4 | ○ |
| J | Plotting: scoped style everywhere, selections in 3D views, interactive HTML viewer, image-comparison tests | 1.5 | ○ |
| K | Verification and validation pipeline: manufactured solutions, analytical benchmarks, COMSOL harness, generated verification report | 3 (spread) | ◐ |
| L | Code quality and documentation: no private cross-imports, 100 % public docstrings, mypy, theory and verification manuals | 2 | ○ |
| M | Full re-run of every example, comparison table, version 0.2.0 | 2 + ~6 h compute | ○ |

Gates (each phase is committed and pushed only when its gate passes):

- **A** CI green; `import loudpy` changes nothing (environment, logging, matplotlib); tests pass without
  `KMP_DUPLICATE_LIB_OK`; analytical validations in the test suite.
- **B** 2D and 3D use identical air; 3D references regenerated on purpose, change documented.
- **C** existing models give the same parts, orientations and PML layers; P2 regressions unchanged.
- **D** patch tests; exact integration; manufactured-solution orders ≈ p + 1 for P1 / P2 / P3 and Q1 / Q2;
  old operators reproduced at 1e-12.
- **E** same fields (rear air included) at 1e-10; coupled modes and impedance unchanged.
- **F** one model pattern for 2D and 3D; identical meshes twice; no user object mutated; examples run.
- **G** format-3 round trip; far field from the file alone; derived fields vs analytical cases.
- **H** estimator effectivity 0.8–1.5; adaptive loop beats uniform refinement; expected orders reported.
- **I** plate modes vs analytical within 0.5 %; rigid-wall limit recovered; convergence.
- **J** image tests; HTML export; global matplotlib style untouched.
- **L** docstring and private-import tests; mypy; docs build without warnings.
- **M** all examples at full size; differences explained (expected: +0.06 % in 3D from the air, ~1e-6
  from the meshes).

## Phase A — done

- `import loudpy` has no side effects: no environment variable; logging silent by default
  (`lp.set_verbosity`); `lp.plot` functions draw in the loudpy style inside a local style context (the
  global matplotlib settings change only with `lp.plot.style.use()`).
- OpenMP: no clash in a pip environment (Numba uses its `workqueue` threading layer; gmsh, VTK and
  SciPy coexist); the workaround is gone (troubleshooting note in the docs).
- `loudpy.errors`: `LoudpyError` hierarchy (each class also derives from the built-in it replaces) and
  `ConvergenceWarning` / `PlausibilityWarning`; Newton–Raphson, GMRES and ARPACK non-convergence are
  warnings, no longer log lines only.
- `loudpy.checks`: valid and plausible SI ranges for every physical input (materials, damping,
  regions, PML, drives, study frequencies, remeshing rules, scene air).
- Packaging: SPDX license expression, version from `loudpy.__version__`, core dependencies only (NumPy,
  SciPy, Numba, h5py, gmsh) with extras `[plot]`, `[3d]`, `[all]`, relaxed minimum versions,
  CHANGELOG, CITATION.cff.
- Verification tests: clamped plate vs Kirchhoff (within 0.5 %), rectangular cavity (2e-4, observed
  order ≈ 3.9 for the theoretical 4), pulsating hemisphere (2.2 %, limited by straight-sided P2
  geometry: curved elements in phase D).
- CI: ruff errors, fast tests on Linux and macOS at every push, slow tests weekly, docs with warnings
  as errors; failing tests are reported as GitHub annotations. Green on the first full run.
- Found by CI: two undefined names (a `MultiDriver.surface` calling a moved function, a type
  annotation), and results that were not reproducible across machines at the regression tolerances.
  The Numba kernels are now compiled without `fastmath` (IEEE arithmetic, no CPU-dependent
  reassociation; no speed cost), and the tolerances come from the measured floating-point spread
  (`tests/support/reference.py`): operators 1e-10, forced responses 1e-6, damped coupled modes in
  near-degenerate pairs 1e-4 (√ε sensitivity). The references are unchanged.
- `LICENSE`: the official GPLv3 text (gnu.org); `pdm.lock` refreshed for the new dependencies.

## Phase B — done

- Material library format 2: `kind` solid / fluid, SI units listed in the file, reference
  temperature and source of every material (the solids: "typical value, unverified"), damping fits
  grouped (`rayleigh = (α, β)`, `loss_factor`) and used only when lp.Rayleigh / lp.Hysteretic get no
  coefficient. PML settings moved from the air to `lp.PML` (same default values: strength 3, order 1).
  Format-1 files are still read; `MaterialLibrary.save` writes format 2.
- `lp.air(temperature, pressure, humidity)`: ideal-gas mixture of dry air and water vapour (Buck
  saturation pressure). Speed of sound within 0.048 % of Cramer (1993) over 0–40 °C, 80–101 kPa,
  0–100 % RH (`test_air_matches_cramer`). Standard air: ρ = 1.2041 kg/m³, c = 343.235 m/s.
- One air: the library's Air, `lp.Scene(…, air=…)` (default: the library's Air) and the toolkit
  defaults `RHO_AIR`, `C_AIR` are the same values (test). Examples no longer hard-code 343 m/s.
- References. First the refactor alone, with the former air values put back (2D 1.204 / 343.2,
  3D 1.2 / 343.0): every committed reference passes, so nothing else changed. Then with the new
  air, the 12 references that depend on it were regenerated, each change checked against the air
  change (ρ +8.08e-5, c +1.02e-4 in 2D; ρ +3.4e-3, c +6.85e-4 in 3D):

  | reference | max relative change | explained by |
  |---|---|---|
  | operator ρ | 8.08e-5 | ρ ratio, exactly |
  | operator Q | 1.4e-4 | air entries × (c_old/c_new)² − 1 = −2.038e-4 exactly; PML entries via the stretching |
  | operator H | 1.65e-4 | air entries unchanged (Laplacian); only the PML entries move (stretching depends on c) |
  | 2D FSI force 200 / 1500 Hz, voltage | 7.6e-5 – 1.2e-4 | air pressure ∝ ρ, k = ω/c |
  | 3D rectangular cavity | 6.85e-4 | eigenfrequencies ∝ c, exactly |
  | 3D pulsating hemisphere (far field) | 1.2e-2 | phase k·r at 1 kHz, r = 1 m: Δk·r = 0.0125 rad |
  | 3D driver in box, two drivers, open scenes | 4.4e-4 – 7.2e-3 | air spring ρc², amplified near resonances |

  The structure-only references (operators K, M, C, modes, transient, plate) are unchanged; their
  regenerated values differ by ≤ 1e-11 (the fastmath change of phase A, within the tolerances) and
  were not rewritten.
- Found on the way, for later phases: the `test_dtn_system` fingerprint (sums over the whole coupled
  matrix) is dominated by the structure block and cannot see a 1e-4 change of the air block (it
  moved by 7e-12): phase E will fingerprint each block separately. The library value of SolidGlue
  `rayleigh α = 80e-6 1/s` is ~1e6 times smaller than the other materials' (probably meant 80):
  kept as it was, to be confirmed by the author.
