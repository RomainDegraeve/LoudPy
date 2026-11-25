# Changelog

All notable changes to loudpy. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/) (0.x: the API may still change between minor versions).

## [Unreleased]

- Saving results, one way: `study.save(folder, name="fr")` / `res.save(folder, name="fr")` write
  `<folder>/results/<simulation>/<study>_<stamp>.h5` stamped with what made them (study, simulation, and the
  fingerprint of the folder's document when it holds the study); `lp.result_path(folder, name, study)` for a scene's
  `store=` (the study stamped); `lp.load(folder, study="fr")` reads the newest. Scripts generated from a document save
  this way. `lp.results.schema.info(path)`: a result file's head (the one place besides read() that knows the
  layout; lp.Workspace uses it). Format 3 stays (formats 2 and 1 still read; a format-2 sample in the tests).
- `lp.export(obj, path)`: every export by suffix through one registry (`loudpy.results.exporters`: Data, curves, tables,
  fields, results, 3D views, figures, animations); `Data.export`, `Field.export`, `Results.export` call it.
  `Field.table()` / `Field.header()` public.

### Changed (breaking)
- `SweepResults.save(folder)` no longer writes one file per run (one file, or `name=` into a simulation folder);
  `Document.to_script(results=…)` removed (results go where the convention puts them).
- Wall views name their panels: `View3D.label(points, texts)`; `wall_view(…, labels=True | {panel: name} | False)`
  (a STEP solid "solid 4", a Box side "left"; `res.wall_view(f, labels={"4": "back"})`).
- `lp.plot.PlotSpec`: a plot as plain data (quantity, where, axes, the values of the other dimensions, view,
  style), resolved on results by slicing named dimensions; every dimension not plotted is a control (slider / choice:
  frequency, time, mode, sweep run, the phase of a harmonic field), `.step()` moves one, `.animate(res, dim, path)`
  plays one into a GIF / MP4. `lp.Document(...).plots` keeps plot specs with a simulation.
- Tests: `pytest --browser` runs the interface's browser tests (marker `browser`, Playwright + Chromium).
- Item types declare their facts next to their classes (`lp.Solid.facts`: what they select, the medium they fill,
  their physics and role, their key parameter, where they fit — `loudpy.model.facts`); `lp.schema` shows them
  (`Block.facts`, `Block.key`, `Param.key`) and `lp.schema.register(cls, params)` makes a new item type known.
  `lp.suggest(model)`: per geometry group the item types that fit it, from the geometry only (the media on each side
  of its curves, the axis, straight lines and arcs). `lp.readiness` reads the declared roles, and its coupling step
  now names the groups between the structure and the air that are neither coupled nor held (a warning).
- Result files rebuild their model: the description stores the model or scene as a script using the files stored
  with it (now also a scene's cabinet — STEP file or the module of a cabinet builder — and its drivers' geometry,
  and a non-default material library); `res.model.rebuild(folder)` → the `lp.Model` / `lp.Scene` again,
  `res.model.rebuildable`. `Scene.to_script` writes a cabinet builder defined in a module file as an import.
- Examples: every `run.py` builds its model or scene at the top and meshes / solves only under
  `if __name__ == "__main__":` (an interface or a notebook can open it without computing); 05 and 06 give their
  PML a base thickness and tuning (the remeshed bands set their own, results unchanged); 03 radiates through
  `lp.Radiation("dtn")` (its label had kept the former group name). `enclosure.sealed_scene(mesh=False)` etc.
- `lp.cabinets.Box`: drivers checked against the front face when only the depth is a parameter.

### Changed (breaking)
- The examples folder is `examples/` (was `LoudPy_exemples/`).
- Readable group names in the example geometries: `cone` (was `submeca_membranne`), `dustcap`, `surround`, `former`,
  `coil`, `glue_1/2`, `spider`, `air_front` / `air_rear` (`subacou_*`), `wet_front` / `wet_rear`
  (`interface_acou_meca_*`), `axis`, `surround_rim`, `spider_rim` (`interface_constrained_*`), `coil_drive`
  (`interface_forced_z_1`), `PML_front_r` / `_z`, `PML_rear_r` / `_z`, `dtn_front` / `dtn_rear`; the plate:
  `plate`, `rim`, `axis`. Same meshes and results (checked bit for bit). Results stored before:
  `lp.load(path, rename={old: new})` (the examples' table: `examples/shared/names.py`). The tests use frozen copies
  of the original geometries (`tests/data/geometries`).
- One way to plot: `.plot()` on every result object, `.view()` for a 3D View3D. Removed: `lp.plot.scene_view`,
  `lp.plot.scene_view3d` (now `scene_res.plot(f)` / `scene_res.view(f)`, with six default cameras, the cut through
  the driver axis and the dashed outline of the removed part by default), `lp.plot.view` and `Field.view3d` (now
  `Field.view`).
- `lp.Driver.from_modes(…, coil=None)`: the coil is the model's `lp.Coil` item (from the file's model description),
  not a boundary named "interface_forced".
- Fail loudly: `Mesh.rows_of` / `coords_of` (and every internal node lookup) raise `MeshError` for a node tag that is
  not in the mesh, instead of mapping it to a neighbouring row; `rows_of(tags, missing="ignore")` gives −1 where a
  caller means it (`Field.at_nodes`).
- `lp.Output(every=n)` counts over the whole frequency / time list (it counted per mesh block: a remeshed sweep kept
  other frequencies than a one-mesh one); for modes, per mode set as before.

### Added
- `lp.progress`: a callback reporting the steps of long computations (frequencies, time steps, mode shifts, sweep
  runs), silent unless set.
- `lp.signals`: drive signals for lp.Transient — `sine`, `multitone` (Schroeder phases), `wav` (a WAV file).
- `Model.preview_mesh()` (a display mesh of a bare geometry, no region needed), `Model.geometry_size()`,
  `Model.uses(name)` (what refers to a parameter), `GeometrySource.geo_defaults` (DefineConstant defaults of a .geo),
  `MaterialLibrary.with_(*materials, path=…)`.
- `Model.to_script()` / `Scene.to_script()`: a model or a scene written back as Python code that builds it again;
  `lp.Driver.from_modes` keeps its profiles and coil in `driver.source` for that.
- `tests/test_feature_matrix.py`: features × studies (remeshed × drive × output control × save / load on boundary
  integrals, Thiele–Small, probes, derived fields; modes, transient and remeshed voltage sweeps × output / storage).
- `lp.load(path, rename={old: new})` / `loudpy.results.io.rename_groups`: group and item names renamed on read.
- `lp.cabinets`: `Box` (parametric: inner sizes, slanted back, walls, braces, ports), `Step` (your CAD file: the
  inside air found and checked, driver holes closed, port bores named), `Port` (with its Helmholtz tuning),
  `Brace`, `volume()`; ready-made scenes `lp.Scene.sealed_box(…, radiating=)` and `lp.Scene.vented_box(…)`
  (settings from `f_max`), `scene.cameras()` / `lp.plot.default_cameras` (six default views).
- Output control: `lp.Output(fields=…, every=n, probes={name: point})` on `solve(output=…)` of `lp.Modes`,
  `lp.FrequencyResponse` and `lp.Transient` (2D models): fields kept, every n-th entry, full-rate probe signals saved
  with the results (`res.probe_signal(name, field, component)`, `res.probe_names`). The multitone example uses it
  (its result file: 8.7 GB → about 70 MB).
- Multi-parameter sweeps: `lp.Sweep(study, {name: values, …}, combine="grid" | "zip")` (all combinations or pairs,
  as COMSOL's Parametric Sweep); geometry parameters from `DefineConstant[…]` of a `.geo` file (besides builders);
  `SweepResults.evaluate(fn)` → `lp.Data` over the parameters, `.table()`, `.at(**values)`, `.names`, `.runs`.
- `Model.mesh_reusable(path)`; a sweep run that changes no mesh input (a material value…) uses the model's mesh.
- Examples reorganised: one folder per example, numbered in reading order (`01_structure_modes` …
  `15_column_multi_driver`, `validation/`, `shared/` for the 3D box), each with `README.md`, `run.py`, `plot.py` (and
  `plot_3d.py`); paths relative to each script (run from anywhere); sweep examples `08_sweep_geometry` (parametric
  plate `.geo`) and `09_sweep_materials`; former scripts in `LoudPy_exemples/legacy/`.
- Exports: netCDF for labelled data (`data.export("x.nc")`, no extra package), `lp.load_data` (`.nc`, `.npz`);
  VTKHDF time series of fields (`res.export("x.vtkhdf")`, one HDF5 file for ParaView / VTK / pyvista).
- 3D views saved and turned: `View3D.save("x.png" | ".pdf" | ".svg" | ".html" | ".gltf" | ".glb" | ".vtm")` (figure,
  self-contained interactive viewer with every surface's own colours, glTF model with colours, ParaView multiblock),
  `View3D.interactive()` (window, or inline in Jupyter), `View3D.plotter()`, `View3D.image(scale=…)`;
  `field.view3d()` / `lp.plot.view(field, cut=…)` (a 3D field, or a 2D displacement revolved) and
  `lp.plot.scene_view3d(res, f, camera, cut=…)` (one view of a scene result) return a View3D. Examples 02, 06, 11
  save HTML (and glb) files; `--interactive` opens them in a window.
- Result blocks keep their own boundaries (`Block.boundaries`, `/blocks/<i>/boundaries` in result files;
  `res.boundaries_of(block)`).

### Fixed
- `Model.mesh()` without any region failed deep in the mesher (a NumPy error); it now says to add regions (or to
  use `model.preview_mesh()`).
- `lp.cabinets.Box` accepted drivers off its front face, overlapping each other or a port (found later as a broken
  mesh); `for_drivers` (called by lp.Scene) now refuses them with the reason.
- A scene cavity whose f_max is below its first acoustic mode (only the constant mode kept) crashed with an
  IndexError in the static correction of the driver coupling; it now raises a ModelError saying to raise the
  cavity's f_max.
- `res.thiele_small()`, `volume_velocity` and `radiated_power(on=…)` on sweeps remeshed per wavelength (one mesh per
  band): the boundaries were stored once (the last band's node tags) — the integrals failed and the coil velocity used
  for Sd was read at wrong nodes on the other bands. Result files written before are read correctly: each boundary is
  found again on every band's mesh from the geometry (example 06: fs = 111.5 Hz, Qts = 1.81, Sd = 78 cm², Vas = 1.0 L,
  no re-run).
- Remeshed sweeps stored no physical data of the mesh groups in their model description (`res.model.groups` empty):
  taken from a meshed band copy now; Vas falls back to the air region's material for older files.

## [0.2.0] — 2026-11-21

The API refresh (branch `api-refresh`, phases A–M of `docs/ROADMAP.md`): own generic FE core, physics on one
system builder, one model pattern in 2D and 3D, result format 3, error estimation and adaptivity, shells, plotting,
the verification pipeline, code quality; every example re-run with this version (comparison with the former outputs
in the phase M section of the roadmap).

### Added
- Type checking: `mypy` configuration in `pyproject.toml` and a CI job; the package passes it.
- `lp.read_comsol(path)` (COMSOL text exports → `ComsolTable`, `.data(x, y)`) and `lp.compare(ours, reference,
  rtol=…, metric=…)` (interpolated comparison, maximum error and where). Verification report generated from the
  tests' own measurements: `pytest --slow --verification-report docs/guide/verification_report.rst`.
- `lp.plot.selection(model | scene | mesh, selector)`: what a selector selects, on the mesh (2D domains and
  edges; 3D surface groups). `lp.plot.html(field(s), path)` / `Field.export("x.html")`: an interactive viewer in
  one self-contained HTML file (WebGL, frames with a slider). Image-comparison tests of the plots
  (`tests/reference/images`, `--update-reference`) and a test that no lp.plot function changes the global
  matplotlib settings.
- Shells (`loudpy.fem.shell`): flat Reissner–Mindlin MITC3 triangles and MITC4 quadrilaterals in 3D, six DOFs per
  node (drilling rotation tied to the in-plane rotation), warped quadrilaterals linked to their mean plane;
  `shell_matrices`, `modes` (symmetric-mode sparse LU), `fixed_dofs`, `feature_nodes`.
  `loudpy.physics.shell`: `ShellStructure` (material, thickness, damping, supports, in-vacuo modes),
  `wall_shells` (MITC3 cells on the TRI6 faces of a P2 air mesh, normals into the air), the conforming
  shell–air coupling and the direct coupled system (`direct_builder`, `direct_modes`, `direct_response`).
- Material library: MDF, Plywood, Aluminium, Steel (typical values, unverified).
- Elastic walls in scenes: `scene.walls(material, thickness, damping=…, fixed="edges", f_max=…)` (cavity
  configuration): shells on the cavity walls coupled through their in-vacuo modes; `lp.Modes(scene,
  physics="walls")`; `SceneResults.wall_frequencies`, `wall_field(f)`, `wall_mode(n)`, `wall_volume_velocity()`,
  `driver_volume_velocity()`. `physics.modal.static_correction` (function); `DriverCavityCoupling(walls=…)`.
  Example `Coupling_3D/Driver_in_cavity/Running_walls.py`; guide page "Elastic walls (shells)".
- `lp.Adaptive(study, target=…, max_iterations=…)`: a study on meshes adapted to its own error estimate (solids:
  energy norm of the strain recovery; air: pressure-gradient recovery; relative, largest over frequencies /
  modes); `lp.convergence(study, factors=…, quantity=…)`: a study on uniformly refined meshes with observed
  orders, Richardson extrapolation and the expected orders; `loudpy.study.adaptive.estimate_errors(res)`.
- `loudpy.fem.adapt`: element sizes from error indicators (equidistribution) written as a gmsh size view;
  `model.mesh(path, size_field="sizes.pos")` meshes with it (part of the mesh signature).
- `loudpy.fem.estimate`: a-posteriori error estimation — polynomial-preserving gradient recovery, recovery
  indicators (scalar, axisymmetric elastic energy) and an explicit residual indicator.
  The recovery is a sparse operator (`recovery_operator`) and falls back to a lower fit degree on patches one
  cell across a thin layer; `gradient_energy`, `elastic_energy`: the norms.
- Derived values: `res.volume_velocity(on=…)`, `res.radiated_power(on=… | radius=…)`, `res.thiele_small(on=…)`
  (voltage drive); ParaView series `res.export("x.pvd")`.
- Derived fields: `res.field("strain" | "stress" | "von_mises" | "v" | "a" | "particle_velocity" |
  "intensity", …)` (axisymmetric, from the element gradients and the material data stored with the results;
  `loudpy.results.derived`); strain / stress components `rr`, `zz`, `rz`, `tt`.
- `FrequencyResults.far_field(at, radius, center, baffle)`: pressure outside the 2D mesh by the
  Kirchhoff–Helmholtz integral over a sphere in the air (free field or infinite baffle), from the result
  file alone. `Results.arrays()`: the results as plain NumPy arrays.
- Result format 3: every result file holds the model description (tree, items, resolved physical data per
  mesh group, the geometry files or the builder's code) and the study settings (`res.model`, `res.study`,
  `res.model.extract(folder)`); a parametric sweep is one file (`SweepResults.save("x.h5")`, read by
  `lp.load`). Formats 2 and 1 are still read.

- One model pattern in 2D and 3D: `lp.Model(cabinet, dim=3, drivers=[…])` builds a scene; scenes have
  `print` / `tree()`, `check()`, `copy()`, parameters (`lp.P` in their settings, `cabinet(occ, p)`),
  `with_parameters()`, and run in `lp.Sweep`.

- Parameters and sweeps: `lp.Parameters(…)` on the model, `lp.P(name)` in any numeric field of an item,
  geometry builders `builder(occ, p)`, `model.with_parameters(…)`, and `lp.Sweep(study, name, values)` →
  `SweepResults` (values, one results object per value, `stack`, `save`).

- `print(model)` / `model.tree(selections=True)`: the model as a tree (geometry, materials, regions,
  boundaries, mesh controls, mesh); `model.check()` checks selections, materials and settings without
  meshing and lists every problem at once; `lp.MeshSize(on=…, size=…)`: element size by selection;
  `model.copy()`.
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
- Geometry sources: `lp.Model` takes a `.geo` of curve loops (as before) or of surfaces (Physical
  Surfaces), a CAD file (`.step`/`.stp`/`.iges`/`.igs` in their own unit, `.brep` in metres: its faces,
  or the closed regions of a sketch) or a builder function `builder(occ) → {label: surfaces}`. Surfaces
  are fragmented together (conforming mesh); unnamed surfaces are selected by position and take the
  region's name in the mesh. The HPNEW speaker can be built from its STEP sketch directly.
- `lp.Profile(selector, start=None, end=None)`: driver wet profiles selected on the curves of the 2D
  geometry (one chain, or the shortest way between two points) instead of gmsh curve tags (still
  accepted); `lp.Driver.from_modes(geometry=…)` also takes a builder or the driver's `lp.Model`.
- Regions and boundaries combine with `|`, `&`, `-` like selectors; a string naming a region of the model
  stands for that region in `lp.Interface` / `lp.BoundaryOf`.
- `lp.PML(…, on=selector)`: the base curves of the layers, selected on the curves of the geometry (no
  `PML_*` group needed). The geometry of the layers is stored with the mesh (`pml_layers` in
  `<mesh>.msh.json`), not in the surface names; meshes written before are still read.

### Changed
- Interpolation (`probe`, `pressure`, `spl`, `Field.at` …) uses the mesh's own elements: exact for the FE
  field in 2D (any order, quadrilaterals, curved cells) and in **tetrahedra** (3D was the nearest node,
  2D was linear on sub-triangles). `Mesh.interpolation(points, gradient=True)` also gives the gradient.

- Meshing is deterministic by default (`model.mesh(…, deterministic=True)`); the thread count no longer
  decides whether a saved mesh is reused. `FrequencyResponse(remesh=…)` meshes copies of the model and no
  longer changes the regions' sizes while it runs.

- Internal: every study solves a system declared on one builder (`loudpy.physics.builder`): fields
  (displacement, pressure, coil current, radiation auxiliaries, modal coordinates), terms as functions of
  s = jω, contributions by role. The voice-coil circuit is a physics like the others; the 3D scenes use
  the same builder with low-rank boundary blocks and a block-elimination solver. Same results (see
  ROADMAP, phase E).

- 3D scenes: P2 tetrahedra are curved (isoparametric). New 3D meshes place the mid-edge nodes on the
  CAD surfaces, and the 3D operators follow them: the pulsating-hemisphere radiation error falls from
  1.8 % to 0.011 % (h = 3.5 cm, 200 Hz). Meshes saved before keep their straight edges until rebuilt.
  Folded curved cells raise a `MeshError`.
- Internal: every finite-element operator (structure, Fourier orders, nonlinear, air with PML,
  coupling, coil force, DtN, 3D air) is assembled by one generic core (`loudpy.fem`: Lagrange P1–P3 /
  Q1–Q2 elements, quadrature, function spaces, Numba kernels). Same results (see ROADMAP, phase D).
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
  `…_Corner`), without the layer geometry (`__r0=…__t=…`). `Model.groups("domain")` lists the domains
  only; the `PML_*` groups are curve groups.
- Curve loops of a `.geo` are its closed physical curve groups, whatever their names (no `sub` prefix
  needed).
- `import loudpy` has no side effects: no environment variable, logging silent by default
  (`lp.set_verbosity("info")` to see progress), the plot style no longer changes matplotlib's
  global settings (`lp.plot.style.use()` to opt in).
- Non-convergence of Newton–Raphson, GMRES and ARPACK is reported as a `ConvergenceWarning`.
- Numba kernels compiled without `fastmath`: results reproducible across CPUs (IEEE arithmetic).
- Packaging: SPDX license expression, version read from `loudpy.__version__`, optional extras
  `loudpy[plot]`, `loudpy[3d]`, `loudpy[all]`; minimum dependency versions relaxed.

### Fixed

- 3D meshes with curved mid-edge nodes could contain folded cells near strongly curved surfaces (a driver
  cut-out on a coarse cavity mesh): their mid-edge nodes are moved back to straight edges when the mesh is
  read (`fem.tet10.untangle_p2`). The 3D mesh signatures record the node placement (old straight meshes are
  rebuilt once) and 3D meshing uses one thread.
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
- Library materials `Rubber` and `PhenolicCloth`: their Rayleigh fits did not follow their own loss
  factor (Rubber fitted at 50 Hz–1 kHz, 0.086 instead of 0.15 at 100 Hz; PhenolicCloth about twice too
  damped everywhere). Now the fits of η at 100 Hz and 1 kHz like the other materials: Rubber (85.6798,
  2.1703e-5), PhenolicCloth (39.9839, 1.0128e-5). Damped results of models using `lp.Rayleigh()` on them
  change (speaker: lower modal damping where the spider works).
- `lp.Transient` ignored hysteretic damping (`lp.Hysteretic`, a loss factor that has no time-domain form):
  those regions ran undamped without a warning. It now raises an error pointing to `lp.Rayleigh`.
- `MultiDriver.surface` called a function that had moved (removed; use `DriverResponse.surface`).

### Removed
- The former packages (`Studies`, `Plotter`, `Files_Loader`, `Files_Saver`, `SubDomains`, `Interfaces`,
  `Meshing`, `assembly`, `SimObjects`, …) and the `KMP_DUPLICATE_LIB_OK` workaround.

## [0.0.11] and earlier

The former API (`Problem`, `DomainSpec*`, `EigenStudy`, `FreqStudy`, `TimeStudy`, `Plotter`).
