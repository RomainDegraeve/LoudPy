# Changelog

All notable changes to loudpy. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/) (0.x: the API may still change between minor versions).

## [Unreleased] — branch `api-refresh`

The roadmap of this work is in `docs/ROADMAP.md`.

### Added
- Layered package (`model`, `geometry`, `fem`, `physics`, `solvers`, `study`, `scene`, `results`, `plot`)
  with one public namespace: `import loudpy as lp`.
- Studies `lp.Modes`, `lp.FrequencyResponse` (with `lp.PerWavelength` remeshing), `lp.Transient`;
  3D scenes `lp.Scene`, `lp.Driver`, `lp.Cabinet`.
- Evaluated results: `lp.Data` (labelled arrays) and `lp.Field` (values on a mesh) with units,
  selections (`lp.Point`, `lp.Points`, `lp.Polar`, `lp.Arc`, `lp.Line`), `.plot()` and exports
  (`.csv`, `.mat`, `.npz`, `.vtu`, pandas).
- `loudpy.plot`: curves, maps, polar plots, fields, mode grids, 3D views (`View3D`, `scene_view`),
  animations, one figure style.
- Result file format 2 (HDF5); format 1 files remain readable.
- `loudpy.errors`: exception hierarchy (`LoudpyError`, `ModelError`, `SelectionError`, `MeshError`,
  `StateError`, `SolverError`, `ResultsError` …) and warnings (`ConvergenceWarning`,
  `PlausibilityWarning`).
- `loudpy.checks`: SI plausibility checks of every physical input (errors for invalid values,
  warnings for values that are probably not in SI units).
- Verification tests against analytical solutions (clamped plate, rectangular cavity, pulsating
  sphere radiation).
- Continuous integration (GitHub Actions): lint, tests, documentation.

### Changed
- `import loudpy` has no side effects: no environment variable, logging silent by default
  (`lp.set_verbosity("info")` to see progress), the plot style no longer changes matplotlib's
  global settings (`lp.plot.style.use()` to opt in).
- Non-convergence of Newton–Raphson, GMRES and ARPACK is reported as a `ConvergenceWarning`.
- Numba kernels compiled without `fastmath`: results reproducible across CPUs (IEEE arithmetic).
- Packaging: SPDX license expression, version read from `loudpy.__version__`, optional extras
  `loudpy[plot]`, `loudpy[3d]`, `loudpy[all]`; minimum dependency versions relaxed.

### Fixed
- `MultiDriver.surface` called a function that had moved (removed; use `DriverResponse.surface`).

### Removed
- The former packages (`Studies`, `Plotter`, `Files_Loader`, `Files_Saver`, `SubDomains`, `Interfaces`,
  `Meshing`, `assembly`, `SimObjects`, …) and the `KMP_DUPLICATE_LIB_OK` workaround.

## [0.0.11] and earlier

The former API (`Problem`, `DomainSpec*`, `EigenStudy`, `FreqStudy`, `TimeStudy`, `Plotter`).
