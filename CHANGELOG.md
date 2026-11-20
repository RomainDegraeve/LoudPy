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
- Selections: `lp.Group`, `lp.Box`, `lp.Near`, `lp.Interface`, `lp.BoundaryOf`, combined with `|`, `&`,
  `-`, given to any region or boundary as `on=` (the name stays its label). Boundaries are selected on
  the mesh: interfaces, clamps and the coil no longer need curves drawn in the `.geo`.
  `Model.select(selector)` shows what a selector selects; `Field.restrict` accepts selectors.
- `lp.PML(…, on=selector)`: the base curves of the layers, selected on the curves of the geometry (no
  `PML_*` group needed). The geometry of the layers is stored with the mesh (`pml_layers` in
  `<mesh>.msh.json`), not in the surface names; meshes written before are still read.

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
- Names are labels only: `lp.Clamped(…, dofs="rz")` (default fully clamped; was read from `_r_` /
  `_rz_` in the group names: write `lp.Clamped("interface_constrained_r", dofs="r")` for a radial
  condition), `lp.Coil(…, direction="z")`; `lp.FluidStructure` / `lp.Radiation` lose `side=`.
- Mesh signature version 2 (`<mesh>.msh.json`): records the meshed groups, not the selector names.
  Meshes saved before are rebuilt once. A saved mesh that keeps more boundary groups than the model
  needs is reused.
- `Model.add` refuses two items with the same name (the name is the label of the results).
- PML surfaces of new meshes are named after their curve group or region (`PML_r+_1_front`, `absorber_1`,
  `…_Corner`), without the layer geometry (`__r0=…__t=…`). `Model.groups("domain")` lists the curve loops
  only; the `PML_*` groups are curve groups.
- `import loudpy` has no side effects: no environment variable, logging silent by default
  (`lp.set_verbosity("info")` to see progress), the plot style no longer changes matplotlib's
  global settings (`lp.plot.style.use()` to opt in).
- Non-convergence of Newton–Raphson, GMRES and ARPACK is reported as a `ConvergenceWarning`.
- Numba kernels compiled without `fastmath`: results reproducible across CPUs (IEEE arithmetic).
- Packaging: SPDX license expression, version read from `loudpy.__version__`, optional extras
  `loudpy[plot]`, `loudpy[3d]`, `loudpy[all]`; minimum dependency versions relaxed.

### Fixed
- **2D fluid–structure coupling** (all coupled results change): (1) the coupling sign was set per
  boundary group while the edge normals follow the drawing direction of each curve, so the curves
  drawn the other way coupled with the wrong sign (HPNEW speaker: 38 % of the rear edges, 5 % of the
  front edges; swept volume 14–18 % low); (2) the front/rear convention (rear air matrices × −1) gave
  the front pressure the wrong sign and reversed the reaction of the rear air on the structure
  (energy not conserved by 0.2–4 %). The coupling now uses the outward normal of the adjacent air on
  every edge, from the mesh, and the same formulation in every air region: energy balance exact to
  1e-10, radiated powers positive, swept volume exact. Speaker response: up to ±2 dB in front, +6 dB
  behind. New verification tests.
- Library material `SolidGlue` (cured epoxy): its Rayleigh fit (α = 8e-5 1/s, β = 1.2e-7 s) gave a loss
  factor of 1e-4 – 8e-4 instead of its own 0.03; now the fit of η = 0.03 at 100 Hz and 1 kHz
  (α = 17.136 1/s, β = 4.34e-6 s). Results of models with Rayleigh-damped glue change (damping only).
- PML layers grow along the outward normal of the air they bound (from the geometry). The former rule
  pointed them away from the centroid of the drawing, which put a layer inside its air when the
  drawing extended beyond it (e.g. a structure above a top PML); the models of the examples were not
  affected (identical meshes and operators).
- `MultiDriver.surface` called a function that had moved (removed; use `DriverResponse.surface`).

### Removed
- The former packages (`Studies`, `Plotter`, `Files_Loader`, `Files_Saver`, `SubDomains`, `Interfaces`,
  `Meshing`, `assembly`, `SimObjects`, …) and the `KMP_DUPLICATE_LIB_OK` workaround.

## [0.0.11] and earlier

The former API (`Problem`, `DomainSpec*`, `EigenStudy`, `FreqStudy`, `TimeStudy`, `Plotter`).
