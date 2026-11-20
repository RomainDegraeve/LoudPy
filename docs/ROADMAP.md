# loudpy roadmap — towards a professional, verified FE library

Approved on 2026-11-20, reviewed after phase C (same day, see "Review after phase C"). Goal: one
consistent API across physics, studies, saving and plotting, ready to use, organised like COMSOL / ANSYS
(geometry → selections → materials and physics → mesh → studies → results), with a test pipeline that
checks accuracy (not only regressions). New physics (BEM, magnetics, thermal, thermoviscous losses, porous
media, anisotropy, motor nonlinearities, PETSc/MPI) comes later; parameters and parametric sweeps were
moved into phase F.

Decisions: own generic FE core (no external FE library); elements: triangles P1–P3, quads Q1–Q2 (2D /
axisymmetric), tetrahedra P1–P3 (3D), shells for cabinet walls coupled directly to the air; both 2D
geometry workflows (gmsh curve loops and Physical Surfaces / STEP), names are labels only; SI units with
plausibility checks; error estimator + adaptive remeshing loop + convergence tool; raster 3D panels in
PDFs + interactive HTML viewer; material values marked "typical, unverified"; Python ≥ 3.13; full re-run
of the examples at the end; COMSOL comparison harness now, reference results later.
After phase C (user, 2026-11-20): **compact model items + a COMSOL-like tree** (keep `lp.Solid` / `lp.Air` …
bundling domain, physics, material, damping; add the tree view, mesh controls by selection, study steps,
derived values, the model description saved with the results); **parameters and sweeps in phase F**;
**phase D next** (API phases after it, on the new core); the coupled 2D examples are re-run in phase M
only (an earlier re-run, "phase R", was considered and dropped); Rubber and PhenolicCloth damping
refitted (done).

Cost unit: one *session* ≈ one long working block (3–5 h). Status: ✅ done · ◐ in progress · ○ to do.

| Phase | Content | Sessions | Status |
|---|---|---|---|
| A | Foundations: no import side effects, OpenMP, exceptions, SI checks, packaging, CI, verification tests | 1 | ✅ |
| B | Materials and air: library v2 (units, temperature, provenance), damping separate, air from (T, p, RH), one air for 2D and 3D | 1 | ✅ |
| C | Selections and geometry: selection objects, three geometry sources, nothing read from names, PML by selection, driver profiles from selections | 3 | ✅ |
| D | Own generic FE core: Lagrange P1–P3 on lines / triangles / quads / tetrahedra, quadrature, spaces, generic kernels and assembly, curved (isoparametric) elements; old kernels removed | 5 | ✅ |
| E | Physics interfaces and couplings: one contribution protocol, generic system builder, electrical circuit as physics, 3D on the same protocol (the sign convention was fixed in C) | 4 | ✅ |
| F | One model for 2D and 3D: `lp.Model(dim=…)`, COMSOL-like tree (print, `check()`, mesh controls by selection, study steps), parameters and parametric sweeps, selectors and geometry sources in 3D scenes, typed mesh and solver settings, one remeshing rule, immutable model during solves, deterministic meshing by default, examples rewritten | 4 | ✅ |
| G | Results format 3: one results family, self-contained files (with the model description and parameters), public arrays, derived fields (stress, strain, velocity, intensity) and values (power, volume velocity, impedance, SPL, Thiele–Small parameters), interpolation in tetrahedra, ParaView series | 3 | ✅ |
| H | Error estimation and adaptivity: recovery and residual indicators, `lp.Adaptive`, `lp.convergence` | 3 | ✅ |
| I | Shells: MITC3/MITC4 Reissner–Mindlin, direct coupling to the 3D air, wall-mode study, elastic-wall enclosure example | 4 | ◐ |
| J | Plotting: scoped style everywhere, selections in 3D views, interactive HTML viewer, image-comparison tests | 1.5 | ○ |
| K | Verification and validation pipeline: manufactured solutions, analytical benchmarks, COMSOL harness, generated verification report | 3 (spread) | ◐ |
| L | Code quality and documentation: no private cross-imports, 100 % public docstrings, mypy, theory and verification manuals | 2 | ○ |
| M | Full re-run of every example (incl. the coupled 2D ones, whose stored results predate the C1 coupling fix and the damping refits), comparison table, version 0.2.0 | 2 + ~6 h compute | ○ |

Gates (each phase is committed and pushed only when its gate passes):

- **A** CI green; `import loudpy` changes nothing (environment, logging, matplotlib); tests pass without
  `KMP_DUPLICATE_LIB_OK`; analytical validations in the test suite.
- **B** 2D and 3D use identical air; 3D references regenerated on purpose, change documented.
- **C** existing models give the same parts and PML layers; P2 regressions unchanged except the coupled
  ones, which change by the coupling fix below, checked by physical invariants (energy balance, swept volume).
- **D** patch tests; exact integration; manufactured-solution orders ≈ p + 1 for P1 / P2 / P3 and Q1 / Q2;
  old operators reproduced at 1e-12 (same machine, former code in a git worktree; the committed
  references at their tolerances); assembly not slower than the former kernels by more than 1.5×.
- **E** same fields (rear air included) at 1e-10 as after phase C; coupled modes and impedance unchanged;
  energy balance and swept-volume checks still exact.
- **F** one model pattern for 2D and 3D; identical meshes twice; no user object mutated; a parametric sweep
  equals the separate runs; examples run.
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
  moved by 7e-12): phase E will fingerprint each block separately. SolidGlue (an epoxy) had a Rayleigh fit
  (α = 8e-5, β = 1.2e-7) giving η ≈ 1e-4 instead of its 0.03: replaced by the fit of η = 0.03 at
  100 Hz and 1 kHz (`lp.Rayleigh.from_loss_factor`, the rule the other fits follow). Only the
  damping-dependent references changed (C operators, damped modes and responses; K, M unchanged).
  Rubber and PhenolicCloth fits do not match their own η either (to be reviewed with the author).

## Phase C — done

### C1: names carry no physics; coupling normals from the mesh (done)

- `lp.Clamped(…, dofs="rz")` and `lp.Coil(…, direction="z")` are parameters with defaults;
  `lp.FluidStructure` and `lp.Radiation` have no `side`; air and PML regions have no orientation.
  `geometry.naming` is now only the label-matching rule (the `_r_`, `front`, … tokens are modifiers
  for matching, nothing else).
- The fluid–structure coupling uses, on every edge, the outward normal of the adjacent air, found
  from the mesh (`geometry.topology.EdgeSides`). With it, every air region has the same formulation.
- **Two errors in the former coupling, found while doing this:**
  1. One ±1 per boundary group, while the edge normal follows the direction in which each curve of
     the `.geo` is drawn: on the HPNEW speaker, 136 of 357 rear edges (the spider zigzag) and 4 of
     82 front edges coupled with the wrong sign. The volume swept by a unit rigid translation was
     −13.7 % (front) and −18.3 % (rear) off π r_max²; it is now exact (1e-9).
  2. The "front / rear" convention (rear air matrices × −1) left the front pressure with the wrong
     sign and made the rear air push back on the structure with the wrong sign: the power into the
     front air came out negative and the energy balance (coil power = structural losses + radiated
     power) was off by 0.2–4 %. Now each radiated power is positive, the balance closes to 1e-10,
     and front and rear pressures are in antiphase for the pistonic cone.
  Effect on the speaker (same mesh, 1 N on the coil): resonance level −1.1 dB (same frequency,
  115 Hz), cone displacement and front SPL ±2 dB over 50–600 Hz, rear SPL up to +6 dB near 200 Hz.
  New tests: `test_coupling_sweeps_the_right_volume`, `test_coupled_energy_balance`,
  `test_piston_front_and_rear_in_antiphase`. References regenerated on purpose: Cam, H, Q, the
  coupled responses (force, voltage) and the DtN system; the structural ones unchanged.
- Mesh signature version 2: the meshed groups (with kind, size, PML thickness), not the selector
  names; the committed test meshes were migrated after checking that their inputs match (the
  `.msh` files are unchanged), other saved meshes are rebuilt once. A test run that rewrites a
  committed mesh now fails.

### C2: selection objects (done)

- `lp.Group(*labels)`, `lp.Box(r=…, z=…)` (a number is a line: `lp.Box(r=0)` is the axis), `lp.Near(point)`,
  `lp.Interface(a, b)`, `lp.BoundaryOf(a)`, combined with `|`, `&`, `-`; a string stands for `lp.Group`, a
  model item for what it covers. Every region and boundary takes `on=` (default: its name as a label); the
  name is the label in the results. (`lp.Groups` of the first sketch is `lp.Group` with several labels or
  a glob.)
- Regions select whole domain groups **before meshing**, on the curve loops of the `.geo` (outlines
  sampled with gmsh; `lp.Near` = the closed loop containing the point). Boundaries select **edges of the
  mesh after meshing** (`geometry.selectors.Edges`: edges between two different domain groups, on the
  outside of the meshed domains, and of the kept named curves), so interfaces need no curve in the `.geo`
  and loaded meshes are selected like fresh ones. A boundary made of labels only keeps one part per named
  group (identical parts, identical numbers); any other selector gives one part named after the item.
- `Model.select(selector, kind=)` evaluates a selector on the meshed model; `Field.restrict` accepts
  selectors. `Model.add` refuses duplicate names. A saved mesh keeping more boundary groups than the model
  needs is reused (the kept curve groups do not change the triangulation).
- Checks (`tests/test_selections.py`, 9 tests): on the committed FSI mesh, `Interface(solids, front air)`
  gives exactly the 82 edges of `interface_acou_meca_front`; the rear one gives the 357 named edges plus
  the 3 of the clamped spider rim, which the `.geo` removed by hand; coil = `Interface("coil", "former")`,
  axis = `BoundaryOf("dustcap") & Box(r=0)`, both rims by `Near` / `Box`; plate rim, axis and faces. A
  speaker model whose six boundaries are all selectors (no named curve used) gives the coupled response
  of the named model on the same mesh to ≤ 7e-14 (u, p, 50 Hz – 5 kHz). No reference changed (123 tests
  pass with `--slow`).
- Limits: `lp.Near` on a boundary takes the chain of edges between corners (> 30°), junctions and
  changes of the adjacent domains; selectors are 2D (3D scenes come with phase F).

### C3: PML by selection (done)

- `lp.PML(name, on=…)` selects the **base curves** of the layers on the curves of the `.geo` before
  meshing (`geometry.selectors.Curves`: the same selectors, sides = the meshed curve loops;
  `GeometryIndex` reads the groups and samples every curve in one gmsh session). Default: the name as a
  label of the `PML_*` curve groups, which are now curve groups (`Model.groups("domain")` = the loops).
- The mesher grows each layer from a straight base line r = const or z = const along the **outward normal
  of the air surface it bounds** (`gmsh.model.isInside` on that surface) instead of away from the
  centroid of the drawing; corners where two perpendicular layers meet; errors for a curved / oblique base,
  a base not bounding exactly one air surface, two layers in opposite directions, or a curve claimed by
  two PMLs. The layer geometry goes to `pml_layers` in `<mesh>.msh.json`, nothing in the names
  (`PML_r+_1_front`, `PML_r+_1_front_Corner`; `absorber_1` … for a PML selected by position); the
  former names (`…__r0=…__t=…`) are still read when a mesh has no stored layers.
- Gate: the committed FSI mesh is reused and gives the same six PMLLayer values (test); H and Q
  unchanged (operator references pass). A deterministic fresh mesh of the FSI speaker by the new builder
  is node-for-node the one of the former builder (17 237 nodes, run with the former code in a git
  worktree), H and Q equal to 4e-14 and 5e-15 (the former names rounded r0 = 0.2999999999999985 of the
  brep to 0.3). The same layers come out of `lp.PML("PML")` and of
  `lp.PML("absorber", on=lp.Box(r=0.3) | lp.Box(z=0.532) | lp.Box(r=0.2) | lp.Box(z=-0.169))`, and are read
  back when the mesh is reused.
- **Bug of the former rule, found on the way:** on a drawing that extends beyond the PML base (a solid
  above a top PML), the centroid rule grew the layer *into* the air (z from 0.05 to 0.1 over an air
  ending at 0.1, overlapping it) while its name still said +z. The new rule grows it outward
  (`test_pml_grows_out_of_the_air`). The example models were not affected.

### C4: geometry sources (done)

- `geometry/sources.py`: `GeometrySource` (a `.geo`, a CAD file or a builder function) loads the geometry
  into gmsh the same way for the selectors (`GeometryIndex`: layout, domains, named curve groups, sides of
  every curve, sampled curves) and for the mesher, so the entity tags agree. Layouts: **loops** (a `.geo`
  without surfaces: every closed physical curve group is a domain, the `sub` prefix is no longer needed)
  and **surfaces** (physical surface groups, CAD faces, the closed regions of a sketch found by
  fragmenting a rectangle with its curves, builder parts). Surfaces are fragmented together (conforming);
  unnamed surfaces are domains `surface_<tag>`, their mesh group takes the region's name. STEP / IGES are
  read in metres (`Geometry.OCCTargetUnit`). A builder's signature hashes its code, defaults and simple
  constants. The mesher takes an explicit plan (`DomainPlan`): no name convention left in it.
- Regions and boundaries combine with `|`, `&`, `-`; a string naming a region stands for it in
  `Interface` / `BoundaryOf`.
- Checks (`tests/test_geometry_sources.py`): the plate as curve loops, as a Physical Surface and from a
  builder gives the same mesh node for node and the same frequencies (bit-identical); a loop `.geo`
  without the `sub` prefix; named / unnamed surfaces, a surface claimed through two groups, a mesh group
  name clash; builder mesh reuse. **The HPNEW speaker from its STEP sketch** (115 curves in mm, no
  names): 10 regions found, selected by points; areas of the curved P2 domains equal to the `.geo` model
  to ≤ 4e-8 (front air 2e-12), except the spider (6.4e-4: one element across each arc crest in one mesh,
  several in the other; the exact face area lies between the two) and the rear air next to it (7e-7, the
  same area), straight boundaries equal to 1e-12, the same six PML layers.
  In-vacuo modes STEP vs `.geo`: 4.3e-3 at 1 mm, 1.7e-6 at 0.5 mm, 4e-5 at 0.25 mm (the frequencies
  themselves move by 1.3 % from 1 to 0.25 mm): the same speaker, different meshes. The `.geo` loop path
  still gives the former deterministic mesh node for node. No reference changed.
- gmsh traps found: `removePhysicalGroups([])` removes every group; a group dropped by a boolean
  operation keeps its name, which then blocks a new group of that name (names removed explicitly).

### C5: driver wet profiles by selection (done)

- `lp.Profile(selector, start=None, end=None)` selects curves of the driver's 2D geometry (the `Curves`
  context of C3, all domains as sides): one chain (start: its end farthest from the axis) or, with `end`,
  the shortest way along the selected curves (Dijkstra over curve lengths); start / end snap to curve end
  points (the given start stays the first point, as before). `geometry=` of `lp.Driver.from_modes` may be
  a path, a builder or the driver's `lp.Model` (region names). Curve tags still accepted.
- The test scenes now use `lp.Profile(lp.BoundaryOf("subacou_rear"), start=…, end=cone foot)` and
  `lp.Profile(lp.Interface(solids, "subacou_front"), start=…)`: the same curves in the same order as the
  former tags, sampled bit for bit identically, so every 3D reference passes unchanged; the coil of the
  3D coupling tests is `lp.Interface("coil", "former")` (no curve tag left in the tests).
- The examples (`LoudPy_exemples/Coupling_3D`) still use curve tags: rewritten in phase F.

### Gate C (passed)

- **Same parts and PML layers on the existing models.** Models written with names reuse the committed
  meshes and build the same parts (one per named group); the committed FSI mesh gives the same six PML
  layers; a fresh deterministic mesh of the speaker is node for node the one of the former mesher (H, Q to
  4e-14, the rounding of r0 in the former names).
- **Regressions.** The only references regenerated in phase C are those of the coupling fix (C1, explained
  there: energy balance to 1e-10, swept volume to 1e-9, front / rear in antiphase). From C2 to C5 no
  reference changed: 134 tests pass with `--slow`, including every 3D scene with the selector profiles.
- **Nothing read from names.** Clamped components and the coil direction are parameters, coupling normals
  and PML directions come from the geometry / mesh, the PML layer geometry is stored with the mesh, curve
  loops are the closed curve groups (no `sub` prefix), driver profiles are selected (no curve tags in the
  tests). Names only label and select (`naming.matches`); the former PML names are still read for old meshes.
- **New, verified:** a speaker whose boundaries are all selectors gives the named model's response to
  7e-14; the plate from loops, a Physical Surface and a builder gives bit-identical meshes and modes; the
  speaker straight from its STEP sketch matches the `.geo` (areas, boundaries, PML; in-vacuo modes converge
  together with the mesh).
- Documentation: `docs/guide/models.rst` (geometry sources, selections, PML layers), `conventions.rst`,
  `scenes.rst`; CHANGELOG.

## Review after phase C (2026-11-20)

The plan holds; adjustments, decided with the user:

- **API style.** loudpy keeps compact items (`lp.Solid("cone", "Paper", damping=…, mesh_size=…)`), shorter
  than COMSOL's separate materials / physics / mesh nodes, and gets the COMSOL-like structure around
  them (phase F): `print(model)` as a model tree (geometry, selections, regions, boundaries, mesh,
  studies), `model.check()` (everything selected, materials complete, units plausible) before meshing,
  mesh controls by selection (`lp.MeshSize(on=…, size=…)` next to the per-item `mesh_size`), studies as
  steps, a preview of what a selector selects (`lp.plot.selection(model, selector)`), and in phase G the
  model description and parameters saved in the result file (like an .mph) and derived values.
- **Parameters and sweeps** (phase F, +1 session): `lp.Parameters`, builders `builder(occ, p)` and item
  values taken from parameters, `lp.Sweep(study, …)` with results stacked over the parameter.
- **Order.** D stays next: E's contribution protocol and F's model are designed on the generic spaces
  (orders, quads, 3D), so nothing is written twice.
- **Coupled examples.** Their stored results (FSI, FSI_elec, FSI_DtN) predate the coupling fix of C1 and
  the damping refits. A separate early re-run ("phase R") was considered and dropped by the user: phases
  D–L change the numerics again, so phase M recomputes them once, with the final code. The user's own
  coupled results computed before `a5800cd` are affected as well (sizes in the C1 notes).
- **Phase D, lessons of C**: the mesh data model and the 2D mesher know TRI6 / LINE3 (and TET10) only —
  D generalizes `geometry.mesh.CELL_TYPES`, `MeshManager.extract_mesh_data` (element types per order,
  quads via recombination) and the selectors' edge extraction (`selectors._tri_edges`: quads, P3 edges);
  the gate compares operators with the former code on the same machine (git worktree), as done in C3/C4.
- **Phase E**: fingerprint the coupled DtN system per block (the global sums barely see the air block).
- **Phase F**: the 3D scenes take the selectors and geometry sources; the examples drop the curve tags.

## Damping refit of Rubber and PhenolicCloth (after phase C, done)

- A material carries two damping descriptions: its loss factor η (hysteretic, constant in frequency) and
  a Rayleigh pair (α, β) for `lp.Rayleigh()` (viscous, needed in time studies), whose loss factor
  η(f) = α/ω + βω can only equal η at two frequencies (in between it dips to 2√(αβ), outside it grows).
  The library rule is η matched at 100 Hz and 1 kHz. Rubber (surround) was fitted at 50 Hz and 1 kHz
  (η 0.150 at 50 Hz, 0.086 at 100 Hz, 0.068 at 316 Hz for η = 0.15); PhenolicCloth (spider) matched its
  η = 0.07 nowhere (minimum 0.085, 0.156 at 100 Hz, 0.140 at 1 kHz: about twice too damped).
- Refitted with `lp.Rayleigh.from_loss_factor(η, band=(100, 1000))`: Rubber (45, 2.3e-5) → (85.6798,
  2.1703e-5); PhenolicCloth (90, 2.0e-5) → (39.9839, 1.0128e-5). A test now checks every library fit
  against its own η at 100 Hz and 1 kHz (within 10 %; Paper, Kapton … are rounded fits within 7 %).
- References regenerated on purpose, damping-dependent ones only (K, M, H, Q, F, KT, plate: back to the
  committed files, changes ≤ 2e-13): C operators (C, C_m1, C_nl), damped speaker modes (frequencies
  move by ≤ 0.03 %, damping ratios fall where the spider works, e.g. 212 Hz ζ 0.029 → 0.022, 423 Hz
  0.038 → 0.024), forced responses (meca at 2 kHz 17 %, FSI at 200 Hz near the 212 Hz mode up to 83 % of
  the peak pressure, 1500 Hz 7.5 %, voltage drive 4 %), transient 3 %, the DtN system's Cs, and the 3D
  tests built on the driver modes (box and two drivers 70 %: lightly damped resonances; open scenes
  0.4–0.5 %). 134 tests pass with `--slow`.
- Found while answering "Rayleigh or hysteretic?": `lp.Transient` silently ran hysteretic regions undamped
  (the nonlinear time integration only uses α, β). It now refuses hysteretic damping with an error
  pointing to `lp.Rayleigh` (test in `test_drive_checks`). Modes and frequency responses accept both.

## Phase D — done (gate report at the end of this section)

- **D1 (done)**: `fem/reference.py`: reference cells (line, triangle, quadrilateral, tetrahedron, gmsh
  conventions), Lagrange elements P1–P3 on lines / triangles / tetrahedra and Q1–Q2 on quadrilaterals on
  gmsh's node layouts (11 gmsh types: 1, 2, 3, 4, 8, 9, 10, 11, 21, 26, 29), basis from the inverse
  Vandermonde, `tabulate` (N, ∇N), facet node tables ordered as the facet element. `fem/quadrature.py`:
  `quadrature(cell, degree)` (Gauss–Legendre tensor rules, collapsed Gauss–Jacobi on simplices, positive
  weights, interior points); the former 7-point triangle rule (8-digit values: exact to degree 5 only to
  ~2e-7) and Gauss lines kept as `legacy_*` so that D3 can reproduce the former operators exactly.
  92 tests (`tests/test_fem_reference.py`): every layout equals `gmsh.model.mesh.getElementProperties`
  (1e-15), δ_ij at nodes, partition of unity, exact reproduction of each element's polynomials (values
  1e-12, gradients 1e-11), facet restriction = facet basis, monomials exact to degree 8 on every cell
  (1e-13), generic TRI6 / LINE3 equal the former hand-written shape functions (1e-13). 226 tests pass
  with `--slow`.
- **D2 (done)**: `fem/space.py` (`Space`: DOFs = sorted node tags of the cells, components interleaved
  as in the former assemblers; isoparametric cell coordinates; `Space.from_mesh`), `fem/kernels.py`
  (generic Numba kernels on tabulated bases: mass, Laplace, load, boundary mass / load on lines in 2D and
  triangles in 3D; real or complex coefficients per quadrature point; optional 2πr weight; `Assembler`:
  CSR pattern once, scatter map). 31 tests (`tests/test_fem_assembly.py`, in-memory gmsh meshes,
  `tests/support/fem_meshes.py`): textbook P1 matrices, measures (area, 2π∫r dA, perimeter, revolved
  area 4π), patch tests on distorted curved cells (P1–P3, Q1–Q2, axisymmetric, tetrahedra P1–P3:
  1e-10), manufactured solutions with L2 orders ≥ p + 1 − 0.15 for P1–P3 and Q1–Q2 (Poisson,
  Helmholtz k² = 4, axisymmetric with the natural axis condition) and ≥ p + 1 − 0.3 on unstructured
  tetrahedra (P1 1.80, P2 2.73, P3 3.73 with h_eff = n_cells^(−1/3)); curved cells: ∫r² dA on a
  quarter annulus converges as h² with straight sides and is 10³–10⁸ times more accurate curved.
  **Found**: gmsh places the interior node of curved TRI10 cells so that P3 converges at ≈ 3.5 instead
  of 4 on curved domains (ln r: 3.42 / 3.82 / 3.54); re-placing it by x₉ = ¼ Σ edge nodes − ⅙ Σ corners
  (`fem.reference.tri10_interior_nodes`) gives 3.89 / 4.12 / 4.01 and a 4× smaller error. To be applied
  when P3 meshes are read (D4). 257 tests pass with `--slow`.
- **D3 (done)**: `fem/solid.py` (generic axisymmetric elasticity, Fourier order m, total-Lagrangian
  kernels for any 2D Lagrange element), `kernels.diffusion` (diagonal tensor: PML), shared physical
  gradients (cofactor form in 2D) and Jacobian (`np.dot`, the former summation order). `MecaAssembler`,
  `MecaAssemblerFourier`, `MecaAssemblerNL` and `AcouAssembler` now call the generic kernels with the
  TRI6 basis at the former 7-point rule (`ElementAssembler.tabulation`); PML stretching vectorised in
  `physics.acoustic.pml_coefficients`. Former code in a worktree (`3f010fd`), same machine, full CSR
  matrices of the committed speaker mesh, max |A − B| / max |A|: M, M_m1, M_nl ≤ 1.3e-14; H 6.9e-15,
  Q 6.2e-15 (with PML); F_int 2.9e-14; Cam, F 0 (not switched yet); **K, C, K_m1, C_m1, KT, C_nl
  1.2e-11**. The 1.2e-11 is not a formula difference: on the worst cell, both kernels evaluated in plain
  IEEE Python agree to 4e-19; the JIT-compiled former kernel is 1.3e-11 from its own IEEE value (the new
  one 1.7e-12), amplified by one nearly inverted spider cell (element 236 of `submeca_spider`: curved P2
  edge, det J at one quadrature point 1.6e-10 vs 2.5e-8 … 1.3e-7 at the others). Every other region
  agrees per element at ≤ 1.1e-13. Committed references unchanged (1e-10). Assembly time, compiled,
  former → new: structure 3.5 → 2.8 ms, air with PML 12.5 → 5.2 ms, Fourier 5.3 → 4.5 ms, TL 2.8 → 2.2 ms
  (0.41–0.85×). 14 new tests (`tests/test_fem_solid.py`, P1–P3, Q1–Q2): rigid translation, uniform-strain
  patch with exact strain energy (1e-11), Fourier rigid modes (m = 1 translation and tilt, m = 0
  rotation), order 0 = axisymmetric, TL at u = 0 = linear element and K_T = ∂F_int/∂u (central
  differences, 1e-6), PML coefficients. 271 tests pass with `--slow`.
  **To check with the user**: the committed spider mesh has a nearly inverted P2 cell (see above).
- **D4a (done)**: boundary kernels on the generic core: `kernels.boundary_normal_mass` (∫ N N n_i ds, the
  normal from the Jacobian: lines in 2D, triangles in 3D), used by `FsiAssembler` (Cam); the coil force by
  `boundary_load`; the DtN vectors b_n = ∮ P_n(cos θ) N 2πr ds by `boundary_load` with the same 8-point
  Gauss rule. Former code, same machine: Cam 2.9e-16, F 2.5e-16, DtN system Ms 1.2e-14, Ks / Cs 1.16e-11
  (their structure block: the spider cell of D3). Test: divergence theorem ∮ x_i n_i = |Ω|, ∮ n = 0 on
  the unit square (P1–P3) and cube (P1–P2). 276 tests pass with `--slow`.
- **D4b, step 1 (done)**: the 3D toolkit (`fem/tet10.py`: `assemble_p2`, `surface_quadrature`) on the
  generic kernels (P2 tetrahedra, degree-4 rule; TRI6 faces with the former 6-point Dunavant rule), with
  `curved=False` (cells mapped from their corners, as before) as the default for this step: same matrices
  as the former exact integrals at ≤ 1.3e-15 (K 1.1e-15, M 7.1e-16, surface points / weights / W
  ≤ 1.3e-15 on a P2 ball), every reference unchanged. On that ball (h = 0.4) straight cells give the
  volume −3.5 % and the area −1.9 %; curved (isoparametric) cells −0.015 % and −0.010 %.
- **D4b, step 2 (done)**: curved 3D cells. `fem.tet10` is isoparametric by default (`CURVED = True`;
  folded cells — det J changing sign inside — raise a `MeshError`), and the 3D mesher
  (`geometry/meshing3d.py`, 3 places) now lets gmsh put the P2 mid-edge nodes on the CAD surfaces
  (`Mesh.SecondOrderLinear = 0`, was 1: straight edges, faceted spheres and cylinders). Pulsating
  hemisphere vs the exact field at 1 m (200 Hz / 1 kHz): h = 5 cm 2.88 % / 1.99 % → 0.033 % / 0.19 %;
  3.5 cm 1.84 % / 1.43 % → 0.011 % / 0.054 %; 2.5 cm 0.89 % / 0.69 % → 0.0027 % / 0.016 %. The 2.2 %
  of phase A was the faceted geometry, as suspected. `test_pulsating_hemisphere_radiation` now runs
  curved cells with a bound of 0.1 % (was 2.2 %). The committed 3D test meshes have straight mid-edge
  nodes and are still reused (the scene signature does not include the node placement), so no reference
  moves; meshes built from now on are curved. **For phase F** (one remeshing rule): put the node
  placement in the mesh signature so that old straight meshes are rebuilt.
- **D4c (done)**: every element order through the model. Mesh cell types line2/3/4, tri3/6/10, quad4/9,
  tet4/10/20 (`geometry.mesh.CELL_TYPES`, `fem.reference.cell_type_name`); the 2D mesher reads every
  order and quadrangles (one element type per group: a partial recombination is refused with a
  `MeshError`) and re-places the interior node of cubic triangles (`tri10_interior_nodes`, see D2);
  `Mesh.triangles` sub-triangulates TRI10 (9) and QUAD4 / QUAD9 (2 / 8); the selectors take the edges
  from the element's facet tables (`_tri_edges`: quads, P3); the assemblers (structure, Fourier, TL, air,
  PML, coupling, force, DtN) infer the element from the cells (`fem.assembly.cell_tabulation`: TRI6 /
  LINE3 keep the former rules, other elements a rule of degree 2·degree + 2). Through
  `lp.Model(mesh_order=p)` on the clamped plate (h = 2 mm), first two axisymmetric modes vs Kirchhoff
  508.96 / 1981.4 Hz: P1 620.9 / 2463.6 (shear locking of linear triangles, +22 %), P2 509.00 / 1976.8,
  P3 508.74 / 1975.5 (P2 and P3 within 0.1 %; at h = 1 mm P2 508.76, P3 508.67). The public API is
  unchanged (`mesh_order` existed; element orders are documented for users in phase F). Operators of
  the committed meshes unchanged. 283 tests pass with `--slow`.
- **D5 (done)**: the former kernels are removed (`fem/elements/tri6_axi/*` (10 files), `fem/batch.py`,
  `fem/shape.py`; `material_matrix_D` moved to `fem/solid.py`; the transient's coil force on the generic
  boundary kernel). Numba, all 2D operators of the FSI speaker, first call in a new process: empty cache
  (compilation) 15.0 s former → 15.3 s new; warm cache 0.12 → 0.21 s. Theory note for users:
  `docs/guide/elements.rst` (cells, layouts, quadrature, isoparametric map, axisymmetric and PML forms,
  boundary terms, what the tests check). 283 tests pass with `--slow`.

**Gate D report.**

| Gate item | Result |
|---|---|
| patch tests | linear fields exact on distorted curved cells, P1–P3, Q1–Q2, axisymmetric, tetrahedra P1–P3 (≤ 1e-10); uniform-strain elasticity patch with exact energy (1e-11) |
| exact integration | monomials to degree 8 on every cell (1e-13); gmsh layouts (1e-15) |
| manufactured orders ≈ p + 1 (L2) | structured 2D: P1–P3, Q1–Q2 ≥ p + 0.85 (Poisson, Helmholtz, axisymmetric); tetrahedra P1 1.80, P2 2.73, P3 3.73 (unstructured); curved P3 4.0 with the TRI10 node fix |
| former operators at 1e-12 (same machine, worktree) | M, H, Q (PML), F_int, Cam, F ≤ 3e-14; 3D toolkit (straight mode) ≤ 1.3e-15; **K-family 1.2e-11**: identical formulas (4e-19 in IEEE Python), JIT rounding of the former kernel amplified by one nearly inverted spider cell — to be accepted by the user |
| committed references at their tolerances | all unchanged (1e-10 / 1e-6 / 1e-4) |
| assembly ≤ 1.5× the former | 0.41–0.85× (faster); compilation unchanged |
| public API unchanged | yes (`mesh_order` existed; tests through `support/current_api.py`) |

Beyond the gate: curved 3D cells (hemisphere 1.8 % → 0.011 %); every element order through the model
(plate: P2 and P3 within 0.1 %, P1 locks as expected).

**Gate D accepted by the user (2026-11-20)**, the 1.2e-11 included: the nearly inverted spider cell was
judged negligible (estimated effect ≲ 0.1 % on spider-dominated modes, ≲ 0.01 % elsewhere; not measured).

## Phase E — done (gate report at the end of this section)

- **E1 (done)**: `physics/builder.py`: fields (unknowns with their size and expand), terms (a block times
  s⁰, s¹, s² or any function of s = jω), loads, `SystemBuilder` (`at(s)`, `rhs(s)`, `split(x)`,
  `polynomial()` for eigenproblems, which refuses non-polynomial terms such as a semi-inductance), and the
  contributions by role (solid, air, fluid_structure, coil: force or voltage with its circuit as a field
  "I"). `FrequencyResponse` builds no block any more: it asks the builder. Fields against the code before
  E (`c97bd73`, worktree, same machine, direct solves; FSI speaker with front and rear air + PML; 50, 212,
  800, 1500, 4000 Hz; force, and voltage with Re 6 Ω, Bl 5, Le 0.3 mH, n = 0.7): u 4.8e-15 / 1.5e-14,
  p 6.9e-14 / 9.7e-14, coil current 1.4e-14, Ze identical. 3 builder tests; 286 tests pass with `--slow`.
- **E2a (done)**: the DtN radiation on the builder. `physics.systems.dtn_system` computes the DtN pieces
  (b_n, auxiliaries) and builds the symmetric u–ψ–φ system with `symmetric_builder` (fields u, psi, phi;
  structure, air −H/ρ −Q/ρ, coupling Camᵀ / Cam in s¹, radiation Dk / Dc, G, Zaa, Caa) and
  `builder.polynomial()`; `Modes(physics="coupled")` goes through it unchanged. Ks, Ms, Cs **bit-identical**
  to the code before E (`c97bd73`). New test `test_dtn_system_blocks` (new reference
  `tests/reference/test_dtn_system_blocks.npz`, created on purpose, nothing else regenerated): size, nnz,
  Σ|a|, ‖·‖_F of every term of the system, so that the air block (Σ|a| ≈ 1.2e4 against 1.2e13 for the
  structure) is checked on its own.
- **E2b (done)**: one registry for both forms. Each role's contribution (`physics/builder.py`: solid, air,
  fluid_structure, radiation, coil) takes `form="frequency"` (fields u, p, I: forced response) or
  `form="symmetric"` (fields u, psi, phi: coupled modes); `coupled_builder` and `symmetric_builder` only
  declare fields and roles. The voice-coil circuit is the coil's contribution (field I, a non-polynomial
  term for the semi-inductance sⁿ Le). DtN system still bit-identical, fields ≤ 1e-13 against `c97bd73`.
  288 tests pass with `--slow`. Not done (a new feature, not in the gate): the exact DtN radiation in
  `FrequencyResponse` (today a frequency study ignores lp.Radiation, as before E): the frequency form of
  the radiation contribution raises an error, so the gap is explicit; candidate for phase F.
- **E3 (done)**: the 3D scenes on the same builder (the generic design chosen by the user). The builder
  gained frequency-dependent terms (a block given as a function of s), `LowRank` operator blocks (dense on a
  boundary, never assembled) and a `BlockSolver`: eliminated fields factorised exactly, GMRES on the Schur
  complement of the big field with a caller-supplied preconditioner (or a direct solve when there is no
  big field). `physics.modal.add_driver_modes` declares the driver modes, extra coordinates (ports) and the
  cavity eliminated through its modes (ρ s² (S D(s) Sᵀ + R)); `physics.exterior.scene_builder` adds the
  exterior FE (K + s² M/c²), the half-space DtN as a LowRank term −G γ(k) Gᵀ and the wet-surface couplings
  Bfᵀ / −ρ s² Bf. `coupled_sweep` (configurations cavity+exterior and merged), `solve_forced` (cavity) and
  `DriverCavityCoupling.coupled_modes` (polynomial form, fields q and a) all go through it. Against the
  code before E3 (worktree, same machine): open scenes q ≤ 6.7e-16, far field ≤ 6.3e-15, exterior p
  ≤ 2e-12 (stored complex64); cavity q 2.3e-18, cavity pressures 1.1e-20, two drivers 2.4e-16, coupled
  modes bit-identical. Same speed (scene script 31.5 → 30.3 s). 3D references unchanged. 3 new builder
  tests (BlockSolver = dense direct solve with a LowRank block, 1e-9; small-only multi-RHS). 290 tests
  pass with `--slow`.
- **E4 (done)**: clean-up and documentation. The exterior FE terms are one function
  (`physics.exterior.add_exterior`), shared by the scene builder and `ExteriorFE.solve` (now a builder +
  `BlockSolver` with no eliminated field; its unused `extra` argument and `ExteriorFE.operator`, the former
  matrix-free operator, are removed). `docs/guide/architecture.rst`: "Systems: fields, terms and
  contributions"; `physics/__init__` map; CHANGELOG. 290 tests pass with `--slow`.

**Gate E report.**

| Gate item | Result |
|---|---|
| same fields (rear air included) at 1e-10 as before E | 2D FSI speaker, front + rear air + PML, force and voltage (semi-inductance n = 0.7), 5 frequencies: u ≤ 1.5e-14, p ≤ 9.7e-14, coil current 1.4e-14 |
| coupled modes and impedance unchanged | coupled DtN system Ks, Ms, Cs bit-identical; Ze identical; 3D scene coupled modes bit-identical |
| energy balance and swept-volume checks exact | unchanged (1e-10, 1e-9) |
| per-block fingerprint of the coupled DtN system | `test_dtn_system_blocks` (new reference, created on purpose) |
| one contribution protocol, circuit as physics, 3D on the same protocol | `physics/builder.py` (fields, terms in s, LowRank, contributions by role, two forms, BlockSolver); the coil contributes the circuit; 3D scenes (cavity, cavity + exterior, merged, coupled modes) on the builder: q ≤ 6.7e-16, exterior p ≤ 2e-12 (complex64), cavity ≤ 2.3e-18, same speed |
| committed references | all unchanged |

Known gap (explicit, not in the gate): the exact DtN radiation in a 2D `FrequencyResponse` (its frequency
form raises an error; the 2D frequency study still ignores lp.Radiation, as before E).

## Phase F — done (gate report at the end of this section)

- **F1 (done)**: `print(model)` / `model.tree(selections=True)` (geometry, materials, regions with
  material / damping / size, boundaries, mesh controls, mesh, and what each item selects);
  `model.check()` (selections, materials complete and plausible, PML tuning, mesh controls; one error
  listing every problem; notes: no coil, air not coupled); `lp.MeshSize(on=…, size=…, name=…)` (element
  size by selection, overrides the regions' size on the selected domain groups, last control wins);
  `model.copy()`. Meshing deterministic by default; the thread count is recorded in the signature but no
  longer decides reuse (every committed mesh still reused). `FrequencyResponse(remesh=…)` meshes copies
  (it used to change the regions' sizes and restore them). Gate items covered: identical meshes twice
  (speaker meshed twice from scratch: same nodes, coordinates and cells), no user object mutated
  (remeshed sweep test). 6 tests (`tests/test_model_tree.py`); 296 tests pass with `--slow`.
- **F2 (done)**: `lp.Parameters` (immutable, `with_`), `lp.P(name)` references (scale and offset:
  `0.5 * lp.P("h") + 1e-3`) in any numeric field of an item, nested dataclasses included (damping, material),
  resolved and validated when the model is meshed or checked (`model.resolved()`, `model.with_parameters()`);
  builders `builder(occ, p)` (second argument named p / params) receive the parameters, whose values enter the
  mesh signature; `lp.Sweep(study, name, values).solve(…)` runs the study on copies of the model, meshed next
  to the model's mesh → `SweepResults` (values, results, `stack`, `save`). Gate item "a parametric sweep
  equals the separate runs": Young's modulus of the plate and the thickness of a builder disc swept, every
  run **bit-identical** to the model built with that value; the user's model unchanged. Found while doing
  it: `resolved()` must resolve the copy's items, or the remeshing study changes the user's model again
  (caught by the F1 test). 5 tests (`tests/test_parameters.py`); 301 tests pass with `--slow`.
- **F3 (done)**: one model pattern for 2D and 3D. `lp.Model(cabinet, dim=3, drivers=[…], air=…,
  parameters=…)` returns an `lp.Scene`; scenes have `tree()` / `print`, `check()` (configuration, drivers,
  parameters, bands; notes), `copy()`, `with_parameters()`, `lp.P` in their settings (the given settings are
  kept, resolved copies are used by meshing and solving), cabinet builders `cabinet(occ, p)`, and
  `lp.Sweep` runs on scenes (one mesh folder per value). One remeshing rule: the 3D signatures record the
  mid-edge node placement (`p2_nodes: curved`) and the parameters of a cabinet taking them; 3D meshing uses
  one thread. The committed scene meshes (straight edges) are now read with reuse="always" in
  `test_scene._small` (fixed test inputs). **Found**: meshing a cavity from scratch with the curved
  mid-edge nodes of D4b gave 126 folded tetrahedra around the driver cut-out (the D4b check raised); gmsh's
  high-order optimiser aborts in this build (HighOrderOptimize 1 and 2), so `fem.tet10.untangle_p2` moves the
  mid-edge nodes of folded cells back to their edge midpoints (globally, conforming) when a P2 mesh is read;
  committed meshes unchanged. Gate items: a scene sweep (cabinet depth 0.15 / 0.20 m, coupled cavity modes)
  bit-identical to the separate runs; one model pattern. 2 tests (scene pattern + sweep, untangle);
  `docs/guide/scenes.rst` "The model pattern in 3D".
  Repair criterion (both the reader and the `assemble_p2` check): det J changing sign inside a cell **or of
  the other sign than the straight cell** (a cell inverted as a whole by a curved face, which |det J|
  silently accepted). The repair moves the cell's mid-edge nodes half-way to the edge midpoints per round
  (all the way after 30 rounds) to keep the curvature. Hemisphere at h = 3.5 cm: one inverted cell; the
  D4b numbers (0.011 % / 0.054 %) were computed with it, the repaired mesh gives 0.040 % / 0.081 % (200 Hz /
  1 kHz; h = 5 cm unchanged 0.033 % / 0.19 %). The verification test keeps the 0.1 % bound and compares the
  worst error over the two frequencies (the 200 Hz error no longer falls between 5 and 3.5 cm).
- **F4 (done)**: the examples on the current API and run. The 3D examples (`enclosure.py`, `column.py`,
  `labyrinth.py`, `Build_vented.py`) select their driver profiles with selectors (lp.BoundaryOf /
  lp.Interface) instead of gmsh curve tags: the sampled profiles are identical (difference 0.0).
  `generate_meshes.py` fixed (an indentation error since phase A; the F1 signature of `_solve_block`).
  Static check of the 42 scripts (they compile, every lp.<name> and loudpy import exists): 0 problems.
  Smoke run (`tests/tools/smoke_examples.py`, untracked like the other tools): the 12 Running scripts run as
  written with the solves stubbed (2D meshing real, reused; 3D scenes checked, not meshed) and the 3
  validation scripts (clamped plate, rectangular cavity, DtN hemisphere) run in full: 15 / 15. The full
  computations are phase M.

**Gate F report.**

| Gate item | Result |
|---|---|
| one model pattern for 2D and 3D | `lp.Model(geometry)` / `lp.Model(cabinet, dim=3, drivers=…)`: both have parameters, print / tree, check(), copy(), with_parameters(), mesh(), studies, lp.Sweep |
| identical meshes twice | speaker meshed twice from scratch: same nodes, coordinates and cells (deterministic by default; 3D on one thread) |
| no user object mutated | remeshed frequency sweep and parametric sweeps work on copies (tests check the items and the mesh after) |
| a parametric sweep equals the separate runs | bit-identical: plate Young's modulus, builder disc thickness, scene cabinet depth |
| examples run | 15 / 15 in the smoke run (above); full re-runs in phase M |

Content of F not done, proposed for later (not in the gate): typed mesh / solver settings objects, studies
as steps (a study list on the model), `lp.plot.selection` (phase J, plotting), the exact DtN radiation in
a 2D FrequencyResponse (explicit error today).

## Phase G — done (gate report at the end of this section)

- **G1 (done)**: result format 3 (`loudpy_format = "3.0"`). `/model`: the model description
  (`results/description.py`: loudpy version, geometry name and file digests, parameters, regions with
  material / damping / mesh size, boundaries, mesh controls, mesh path and signature, the model tree, and
  per mesh group its resolved physical data — region, role, ρ, E, ν, c, Rayleigh / loss factor, PML layer)
  plus the geometry files (the .geo and the files it merges, a CAD file) or the builder's code, embedded up
  to 2 MB; scenes: drivers (frames, modal sources), air, configuration and settings, the cabinet's code.
  `/study`: the settings (type, drive, options; a transient's drive signal as an array). Every study
  attaches them (`Study._finish`), including the streamed 3D sweeps (`StreamWriter.close(described=…)`).
  `res.model` (`.tree`, `.parameters`, `.groups`, `.files`, `.extract(folder)`), `res.study`. Sweeps: one
  file of kind "sweep" with `/runs/<i>` (`SweepResults` moved to `results/sweep.py`; a folder path keeps
  the former one-file-per-value layout). Format 2 and 1 still read. Gate item "format-3 round trip":
  modes, frequency (voltage drive, coupled), transient, sweep (2 values) and the streamed 3D scene saved
  and read back — fields, meshes, extra, meta and boundaries bit-identical, description and settings
  equal; the geometry files extracted from a file give the same digests as the originals. 4 tests
  (`tests/test_results_format3.py`) + format checks in the scene tests; 307 tests pass with `--slow`.
- **G2 (done)**: `Mesh.interpolation(points, gradient=False)` exact for the mesh's own elements: the cell
  found by a KD-tree on the cell centres and Newton on the isoparametric map (curved cells included),
  weights = the cell's Lagrange basis (P1–P3 triangles, Q1–Q2, P1–P3 tetrahedra), gradients through the
  inverse Jacobian; a point on a node takes that node exactly, a point in no cell the nearest node. Tested
  exact (≤ 1e-12) for x^p (P1–P3 triangles and tetrahedra), x^p y^p (Q1–Q2), the isoparametric
  coordinates on curved P2 cells (values and gradients). No regression reference moved (the references are
  nodal). `Results.arrays()` (plain arrays, what `export` writes). **Far field from the file alone**: 2D
  axisymmetric Kirchhoff–Helmholtz (`results/farfield.py`, `FrequencyResults.far_field`): p and ∂p/∂n on a
  sphere (meridian arc, Gauss–Legendre) with the ring Green's function by the trapezoidal rule in φ, rigid
  infinite baffle by the image Green's function. Analytical monopoles from a saved file, 3 m: free field
  7.2e-6 (200 Hz) / 3.3e-4 (1 kHz), baffled 4.5e-5 / 5.5e-4 (h = 1 cm; 2.4e-3 at 1 kHz with h = 2 cm,
  converging). Speaker (committed FSI mesh, baffle z = 0.032): the far field of spheres R = 0.12 and 0.2 m
  reproduces the FE pressure at 0.26 m to ≤ 1.8e-3 at 500 Hz and 1.5 kHz. **Found**: at 100 Hz both radii
  agree with each other but differ from the FE pressure by 26 %: the integral keeps the outgoing wave only,
  so the FE field of this test model (PML 0.1 m thick tuned to 1 kHz, λ = 3.4 m) holds a reflected part at
  low frequency — a property of the coarse test setup, to keep in mind for low-frequency SPL in the
  examples (phase M). 3D scenes: the far field computed by a separate Python process from the stored file
  equals the live one bit for bit (slow scene test). 13 tests (`tests/test_interpolation_farfield.py`).
- **G3 (done)**: derived fields (`results/derived.py`), reachable as `res.field(name, …)`: strain (ε_rr, ε_zz,
  γ_rz, ε_θθ; on the axis ε_θθ = ∂u_r/∂r), stress σ = D(E, ν) ε of each element's group (E, ν from the
  file's group data), von Mises (for a harmonic amplitude, the maximum over the cycle, closed form), v = jω u,
  a = −ω² u, particle velocity −∇p/(jωρ), active intensity ½ Re(p v*). Gradients of the isoparametric field
  at each element's nodes (any order), averaged at the shared nodes. Gate item "derived fields vs analytical
  cases": uniform strain exact for P1–P3 and a quadratic displacement for P2–P3 (strain ≤ 6e-14, stress
  ≤ 4e-11 relative), rigid motion zero, von Mises cycle maximum = brute force (1e-7), quadratic pressure
  → exact particle velocity, plane wave intensity within 0.2 % (P2, kh = 0.08); from a reloaded speaker
  file the stress equals the live one bit for bit, and a file without the description is refused with a
  clear error. 6 tests (`tests/test_derived_fields.py`).
- **G4 (done)**: derived values (`results/values.py`): `volume_velocity(on)`, `radiated_power(on=… | radius=…)`
  and `thiele_small(on=…)` on `FrequencyResults`. Boundary integrals over the curved edges (their Lagrange
  basis, 4-point Gauss, dA = 2πr ds), the normal out of the solid cell next to each edge. Thiele–Small: Mms,
  Cms from the least squares of Im(Bl² / Z_mot) over fs/1.5 … 1.5 fs, Rms = Re(Bl² / Z_mot) at fs (a FE
  model's mechanical resistance varies with frequency: the band mean gave Rms 3× too high, i.e. a predicted
  peak of 28 Ω for a measured 64 Ω; at fs the prediction is 72.2 Ω for 70.4 Ω on a 2.5 Hz grid), Sd = |Q| /
  |v_coil| at fs, Vas = ρc² Sd² Cms. Speaker (Bl = 7 T·m, Re = 6 Ω): fs 112.2 Hz, Mms 17.7 g, Cms 0.113 mm/N,
  Qms 16.9, Qes 1.53, Qts 1.40, Sd 78.2 cm², Vas 0.98 L. ParaView series: `res.export("x.pvd")` (one .vtu
  per entry and mesh, the coordinate as the time step). 5 tests (`tests/test_derived_values.py`).

**Gate G report.**

| Gate item | Result |
|---|---|
| format-3 round trip | modes, frequency (voltage drive, coupled), transient (with its drive signal), a 2-value sweep in one file and the streamed 3D scene: saved and read back with fields, meshes, per-entry values, metadata and boundaries bit-identical, model description and study settings equal; the embedded geometry files give the original digests |
| far field from the file alone | 3D scene: computed by a separate process that only has the file, equal bit for bit to the live result; 2D: Kirchhoff–Helmholtz from a saved file vs analytical monopoles at 3 m: free field 7.2e-6 / 3.3e-4 (200 Hz / 1 kHz), baffled 4.5e-5 / 5.5e-4 (P2, h = 1 cm); speaker: FE pressure inside the mesh reproduced to ≤ 1.8e-3 at 500 Hz and 1.5 kHz |
| derived fields vs analytical cases | strain / stress of uniform (P1–P3) and quadratic (P2–P3) displacements exact (≤ 6e-14 / 4e-11), rigid motion zero, von Mises cycle maximum (1e-7), quadratic pressure → exact particle velocity, plane-wave intensity within 0.2 % (kh = 0.08) |
| derived values | rigid translation: swept volume ±π r_max² (1e-12); power through the front interface = the coupling operator's power of the C1 energy check (1.3e-12 / 1.4e-11) = the flux through a sphere in the air (1.4e-4 / 7.4e-5); a lumped driver's Thiele–Small parameters recovered to 7e-16 |
| interpolation in tetrahedra | exact for the element spaces (P1–P3 tetrahedra, values and gradients; also P1–P3 triangles, Q1–Q2, curved cells) |

Not done in G, proposed for later: probes of the derived names over all entries (`res.probe("stress", …)`;
`res.field(…).at(…)` works per entry), stress for the azimuthal orders m ≥ 1, the far field of 3D results
other than the scenes. Found on the way (recorded in G2): the coarse test model's PML (0.1 m, tuned to
1 kHz) leaves a reflected part of ~26 % at 100 Hz — relevant for low-frequency SPL in phase M.

## Phase H — done (gate report at the end of this section)

- **H1 (done)**: estimators (`fem/estimate.py`). Gradient recovery by **polynomial-preserving recovery**
  (least-squares fit of degree p + 1 — 2p on quadrilaterals — to the nodal values of a vertex patch around
  each node; chosen after measuring: plain nodal averaging gave effectivity 0.43–0.52 for P2). Recovery
  indicators: scalar ∫|G*u_h − ∇u_h|² (axisymmetric 2πr) and elastic energy ∫(ε* − ε_h)ᵀD(ε* − ε_h)2πr;
  explicit residual indicator for −Δu − k²u = f (interior residual with straight-cell Hessians, normal-
  derivative jumps on interior edges). Gate item "effectivity 0.8–1.5", manufactured solution (Poisson), n = 16
  (n = 8): triangles P1 1.034 (1.086), P2 1.007 (1.045), P3 1.293 (1.719); quadrilaterals Q1 1.012, Q2 1.005;
  curved unstructured P1 1.028, P2 1.004, P3 1.194 (1.586); axisymmetric P1 1.08, P2 1.01, P3 1.02; elastic
  energy of interpolated smooth displacements P1 1.003, P2 1.001, a quadratic P2 field exactly 0. The
  residual indicator keeps a constant ratio to the error (≈ 5, 10, 18 for P1–P3: the error's rate, constant
  not calibrated) except on curved P3 cells (the straight-cell Hessians: ratio 140 → 333), noted as a limit.
  12 tests (`tests/test_estimate.py`).
- **H2 (done)**: remeshing from the indicators (`fem/adapt.py`). New cell sizes by equidistribution: a cell's
  squared indicator scales as h^(2p + dim), so h_new = h (T² / (N η²))^(1/(2p + dim)) for the target estimate
  T over N cells, the factor kept within [0.3, 2] per step; nodal sizes (smallest of the adjacent cells) go to
  a gmsh list-format view (`.pos`), read through a PostView field and combined (Min) with the regions'
  size fields; `MeshManager(size_field=…)` / `model.mesh(size_field=…)`, the view's sha256 in the mesh
  signature (reused while unchanged, rebuilt when it changes). Gate item "the adaptive loop beats uniform
  refinement": L-shaped Laplace, exact r^(2/3) sin(2θ/3), P2, energy-norm error, target 0.5 × current
  estimate per step. Uniform h = 0.2 → 0.025: 421 DOFs 6.34e-2, 1533 3.99e-2, 5781 2.51e-2, 22541 1.59e-2
  (rate N^(-1/3), the singularity). Adaptive from h = 0.2: 338 DOFs 3.55e-2, 422 2.09e-2, 562 1.27e-2, 820
  7.27e-3, 1204 4.29e-3 (rate N^(-1.43) over the last two steps; N^(-1) asymptotically): at 1204 DOFs the
  error is 0.10 × the uniform error at the same DOFs, and 0.27 × the uniform error with 19× more DOFs. The
  recovery effectivity on this singular solution is 1.75 → 1.56 along the loop (pre-asymptotic at the
  corner; the gate's 0.8–1.5 is on smooth solutions, H1). 3 tests (`tests/test_adapt.py`, incl. a plate
  meshed with a size view: refined near the rim, reused / rebuilt by the signature).
- **H3 (done)**: model level (`study/adaptive.py`). `lp.Adaptive(study, target, max_iterations)`: after each
  solve the relative error is estimated per cell from the results (solids: elastic-energy recovery indicator /
  ∫ε_hᴴDε_h 2πr; air outside the PML: pressure-gradient recovery / ∫|∇p|² 2πr; per frequency or mode, a cell
  keeps its largest value; `estimate_errors(res)`), new sizes for 0.5 × target (the cell count grows after
  the update: aiming at the target itself made the loop creep), the model copy remeshed with the size view
  (`<stem>_adapt<i>.msh / .pos`), history in `res.meta["adaptive"]`. `lp.convergence(study, factors,
  quantity)`: every mesh size × factor; orders in h = N^(-1/2): local (three meshes) and a least-squares fit
  q ≈ q∞ + C h^r over all meshes (the local orders are noisy on unstructured meshes: 0.2–6.6 on the plate),
  Richardson value q∞, expected orders for the element order (eigenfrequencies 2p, energy p, L2 p + 1).
  Two estimator fixes found on the models: (1) the PPR fit of degree p + 1 is undetermined on a patch one cell
  across a thin layer (3 levels of nodes: z(z−1)(z−2) is in the null space), the minimum-norm fit then mixed
  that null polynomial into ∂/∂z and the plate's estimate jumped (0.0137 → 0.229 between two meshes); the fit
  now falls back to the highest degree the patch determines (exact on quadratics, test added); (2) the size
  view had the solid and the air nodes of an interface twice (fine and coarse sizes), gmsh read the coarse one
  on the shared curves and the speaker's solid estimate stalled at 0.10 (long interface edges at the corners);
  nodal sizes are now per node tag over every group (the finer wins). The recovery is a sparse operator built
  once per space (many frequencies / modes: one product each). 4 tests (`tests/test_adaptive_study.py`, one
  `--slow`), 1 in `tests/test_estimate.py`.

**Gate H report.**

| Gate item | Result |
|---|---|
| estimator effectivity 0.8–1.5 | recovery estimate / true energy error on manufactured solutions at n = 16: triangles P1 1.034, P2 1.007, P3 1.293; Q1 1.012, Q2 1.005; curved unstructured P1 1.028, P2 1.004, P3 1.194; axisymmetric 1.08 / 1.01 / 1.02; elastic energy 1.003 / 1.001 (all → 1 under refinement); on the singular L-shape 1.75 → 1.56 along the adaptive loop (pre-asymptotic at the corner) |
| the adaptive loop beats uniform refinement | L-shape (r^(2/3) sin(2θ/3), P2): adaptive 1204 DOFs energy error 4.29e-3 = 0.10 × uniform at the same DOFs, 0.27 × uniform at 22 541 DOFs; rate N^(-1.43) vs N^(-1/3). Clamped plate modes (P2): adaptive 1006 → 1948 → 3492 → 4554 DOFs, estimate 0.0525 → 0.0185 → 0.0089 → 0.0053 (target 0.008); uniform 5170 DOFs 0.0207, 15 562 DOFs 0.0119; f1 adaptive 508.634 Hz, uniform 508.705 (5170 DOFs) and 508.657 (15 562), reference 508.633 (fit of uniform 5170–29 310 DOFs, itself uncertain by ~1e-5): f1 error 2e-6 at 4554 DOFs (below the reference's uncertainty) vs 1.4e-4 uniform at 5170 and 4.8e-5 at 15 562. Coupled speaker (structure + air + PML, 200 Hz and 1 kHz, target 0.02): 20 109 → 51 719 → 82 114 → 98 395 DOFs, solid 0.294 → 0.109 → 0.053 → 0.028, air 0.0435 → 0.0107 → 0.0058 → 0.0044 |
| expected orders reported | `lp.convergence` tables them (eigenfrequencies 2p, energy p, L2 / point values p + 1) next to the observed ones. Plate P2 (expected 4 for f): fit 3.39 / 4.08 (f1 / f2) over 1006–5170 DOFs, 1.88 / 1.89 over 5170–29 310 DOFs (the clamped edge's corners lower the order asymptotically; the coarse meshes are pre-asymptotic) |

Not done in H, proposed for later: adaptation of 3D scenes and of the azimuthal orders m ≥ 1, coarsening
beyond the regions' mesh sizes (they stay the coarsest sizes), a goal-oriented estimate (error in one output,
e.g. the SPL at a point, by an adjoint solve), the residual indicator's constant (it ranks cells, it does not
measure; curved P3 cells need curved-cell Hessians). The speaker's solid error sits at three corners
(surround–cone junction, surround rim, former foot: half of it in 8 cells on the first mesh), worth knowing for
phase M's mesh choices.

## Phase I — in progress

- **I1 (done)**: shells (`fem/shell.py`). Flat Reissner–Mindlin cells in 3D, six DOFs per node: MITC4
  (Bathe–Dvorkin covariant shear tied at the edge mid-points, 2 × 2 Gauss) and MITC3 (Lee–Bathe, tying points
  (½, 0), (0, ½), (½, ½)); membrane Q1 / CST; drilling rotation tied to the in-plane rotation by a Hughes–Brezzi
  penalty α G h ∫(θ₃ − ω)², α = 1e-3, **fully integrated** (one point per cell left 9 hourglass mechanisms on a
  4 × 4 plate) and with an inertia α ρh³/12 (with ρh³/12 the drilling modes fell at √(12αG/ρ)/h, 550 Hz for a
  10 cm aluminium plate, inside the band); warped quadrilaterals linked rigidly to their mean plane (without it
  a rigid rotation of a quad-meshed cylinder cost energy). Checks (`tests/test_shell.py`, 10 tests): six rigid
  motions with zero energy and exactly six zero eigenvalues (box, cylinder tri / quad, disk); membrane and
  bending patch tests on distorted cells (1e-10); hard simply supported square plate vs the exact Mindlin
  frequencies (3 × 3 eigenproblem per (m, n)), first four modes, MITC4 n = 8 / 16 / 32: 7.2 / 1.74 / 0.43 %
  (h/a = 0.01; the same at 1e-3: no shear locking; 0.10 % for the (1,1) mode at h/a = 0.1), order 2.0;
  MITC3 unstructured h = a/32: 0.18 / 0.45 / 0.46 / 0.73 % (h/a = 0.01), but 0.76–3.1 % at h/a = 1e-3 (MITC3
  keeps some locking as h/L → 0, known; cabinet walls have h/L ≈ 0.02–0.05); clamped disk (h/R = 0.005) vs
  Kirchhoff λ² = 10.2158 / 39.771 at R/50: MITC4 0.02 / 0.14 %, MITC3 0.03 / 0.13 %. **Found**: SuperLU's
  default partial pivoting on shell matrices runs for minutes on 20k DOFs (row swaps on the small drilling and
  rotary terms); `fem.shell.symmetric_lu` (symmetric mode, minimum degree on A + Aᵀ) takes 0.2 s. Library:
  MDF, Plywood, Aluminium, Steel.
- **I2 (done)**: direct shell–air coupling (`physics/shell.py`). `wall_shells` lays one MITC3 cell on each TRI6
  face of a P2 air mesh (normal into the air, from the owning tetrahedron), so the coupling A = ∫ N_a n N_j dS
  (P1 shell × P2 pressure) is exact face by face; the unsymmetric u–p system (K + sC + s²M) d + A p = F,
  (H + s²Q) p − ρ s² Aᵀ d = 0 on the builder (`direct_builder`), coupled modes by shift-invert on the standard
  form (ARPACK's generalized mode needs a symmetric M; the static ω = 0 solution of the u–p form is dropped),
  responses by sparse LU. Checks (`tests/test_shell_air.py`): ∫ x·n = −3V and rigid translations 0 (1e-12);
  a 2 mm steel panel (0.3 × 0.3 m, simply supported) on a 0.1 m rigid box: first coupled mode 117.51 Hz vs
  √(f_vacuo² + f_air²) = 117.69 Hz (f_air = 38.78 Hz from the uniform-pressure air spring; −0.16 %: the air's
  added mass); **rigid-wall limit**: all six walls MDF 18 mm (first panel mode 1.08 kHz), the four lowest
  non-uniform cavity modes (572–894 Hz) move by ≤ 3.3e-3 and with E × 10 / × 100 / × 1000 by 1.4e-4 / 1.3e-5 /
  1.3e-6 (as 1/E).
