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
| C | Selections and geometry: selection objects, three geometry sources, nothing read from names, PML by selection, driver profiles from selections | 3 | ◐ |
| D | Own generic FE core: Lagrange P1–P3 on lines / triangles / quads / tetrahedra, quadrature, spaces, generic kernels and assembly, curved (isoparametric) elements; old kernels removed | 5 | ○ |
| E | Physics interfaces and couplings: one contribution protocol, generic system builder, electrical circuit as physics, 3D on the same protocol (the sign convention was fixed in C) | 4 | ○ |
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
- **C** existing models give the same parts and PML layers; P2 regressions unchanged except the coupled
  ones, which change by the coupling fix below, checked by physical invariants (energy balance, swept volume).
- **D** patch tests; exact integration; manufactured-solution orders ≈ p + 1 for P1 / P2 / P3 and Q1 / Q2;
  old operators reproduced at 1e-12.
- **E** same fields (rear air included) at 1e-10 as after phase C; coupled modes and impedance unchanged;
  energy balance and swept-volume checks still exact.
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
  moved by 7e-12): phase E will fingerprint each block separately. SolidGlue (an epoxy) had a Rayleigh fit
  (α = 8e-5, β = 1.2e-7) giving η ≈ 1e-4 instead of its 0.03: replaced by the fit of η = 0.03 at
  100 Hz and 1 kHz (`lp.Rayleigh.from_loss_factor`, the rule the other fits follow). Only the
  damping-dependent references changed (C operators, damped modes and responses; K, M unchanged).
  Rubber and PhenolicCloth fits do not match their own η either (to be reviewed with the author).

## Phase C — in progress

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

### Still to do in C

- Driver wet profiles from the solid–air interface of the 2D driver model (no curve IDs).
