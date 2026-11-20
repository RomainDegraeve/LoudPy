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
- `lp.air(temperature, pressure, humidity)`: air as an ideal-gas mixture of dry air and water vapour
  (speed of sound within 0.05 % of Cramer 1993 over 0–40 °C, 80–101 kPa, 0–100 % RH; verification test).
- Material library format 2 (`"format": "loudpy-materials/2"`): kinds `solid` / `fluid`, reference
  temperature and source of each material, damping fits grouped under `damping`;
  `MaterialLibrary.save`. Format-1 libraries are still read.
- `lp.Scene(…, air=…)`: the air of a scene, a library name or an `lp.Material`.
- `lp.Rayleigh.from_loss_factor(eta, band=(f1, f2))` and `Rayleigh.loss_factor(f)`.

### Changed
- One air everywhere: the library's `Air` is computed by `lp.air()` at 20 °C, 101.325 kPa, dry
  (ρ = 1.2041 kg/m³, c = 343.24 m/s), and 2D models, 3D scenes and the 3D toolkit defaults
  (`RHO_AIR`, `C_AIR`) all use it. Before, 2D used 1.204 / 343.2 and 3D used 1.2 / 343.0: 2D results
  with air change by ~1e-4, 3D results by ~7e-4 in frequency (up to ~1 % in phase-sensitive far
  fields). The affected test references were regenerated on purpose (see `docs/ROADMAP.md`, phase B).
- `Material`: fields `kind` (`"solid"` / `"fluid"`, was `"elastic"` / `"acoustic"`), `rayleigh =
  (alpha, beta)` and `loss_factor` (were `alpha_ray`, `beta_ray`, `eta`), `temperature`, `pressure`,
  `humidity`, `source`. Library solid values are marked "typical value, unverified".
- PML settings belong to `lp.PML` (`strength=3.0`, `order=1.0` by default), no longer to the air material.
- `lp.Scene(cabinet, drivers, rho=…, c=…)` → `lp.Scene(cabinet, drivers, air=…)`.
- `import loudpy` has no side effects: no environment variable, logging silent by default
  (`lp.set_verbosity("info")` to see progress), the plot style no longer changes matplotlib's
  global settings (`lp.plot.style.use()` to opt in).
- Non-convergence of Newton–Raphson, GMRES and ARPACK is reported as a `ConvergenceWarning`.
- Numba kernels compiled without `fastmath`: results reproducible across CPUs (IEEE arithmetic).
- Packaging: SPDX license expression, version read from `loudpy.__version__`, optional extras
  `loudpy[plot]`, `loudpy[3d]`, `loudpy[all]`; minimum dependency versions relaxed.

### Fixed
- Library material `SolidGlue` (cured epoxy): its Rayleigh fit (α = 8e-5 1/s, β = 1.2e-7 s) gave a loss
  factor of 1e-4 – 8e-4 instead of its own 0.03; now the fit of η = 0.03 at 100 Hz and 1 kHz
  (α = 17.136 1/s, β = 4.34e-6 s). Results of models with Rayleigh-damped glue change (damping only).
- `MultiDriver.surface` called a function that had moved (removed; use `DriverResponse.surface`).

### Removed
- The former packages (`Studies`, `Plotter`, `Files_Loader`, `Files_Saver`, `SubDomains`, `Interfaces`,
  `Meshing`, `assembly`, `SimObjects`, …) and the `KMP_DUPLICATE_LIB_OK` workaround.

## [0.0.11] and earlier

The former API (`Problem`, `DomainSpec*`, `EigenStudy`, `FreqStudy`, `TimeStudy`, `Plotter`).
