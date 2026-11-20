# loudpy — architecture refresh (branch `api-refresh`)

This is the design reference for the refactor: what is wrong with the current structure, the target architecture, the public API it gives the user, and the order of the work.

---

## 1. Current state

About 9.8k lines in 12 top-level packages. The package-level import graph:

```
SimObjects ──► SubDomains, Interfaces ──► ObjectsGeo
PhysicalEntitiesManager ──► Meshing (MeshManager)
assembly ──► Finite_Elements_formulations, SubDomains, Interfaces
Studies ──► everything (assembly, FE, Interfaces, ObjectsGeo, Files_Saver, Files_Loader, root)
Files_Saver ──► Files_Loader
Plotter ──► Files_Loader
```

### 1.1 Architectural issues

| # | Issue | Where | Consequence |
|---|---|---|---|
| A1 | **No layering.** `Studies` mixes problem definition, assembly orchestration, numerical algorithms, I/O and printing. The 3D modules (`Cavity_coupling`, 1024 lines) hold gmsh geometry, P2-tet FE, modal solvers, coupling and pyvista plots in one file. | `Studies/*` | Nothing can be reused or tested on its own. A change in one concern touches the others. |
| A2 | **Studies inherit from `Problem`** and copy its fields (`FreqStudy(Problem, DofMapMixin)`). | `Freq_study`, `Eigen_study`, `Time_study` | A study is not a kind of problem. The state is duplicated, and mixins hide dependencies. |
| A3 | **Mesh and physics are fused by inheritance**: `SubDomainMeca(TR6_Surfs)`, `InterfaceForced(L3_Edges)`. | `SubDomains`, `Interfaces`, `ObjectsGeo` | The physics classes are data bags tied to one element type (TR6 axisymmetric). 3D cannot reuse them. |
| A4 | **Dispatch by `isinstance` filters and substring matching on names.** Each assembler filters `fem_objects` by class. Regions match when `spec.name in domain.name` ("coil" also matches "coil_former"). Orientation comes from "front"/"rear" in names, clamped directions from "_rz_" in names, PML parameters are parsed from names. | `PhysicalEntitiesManager`, `assembly_tri._parse_pml`, `MeshManager` | Fragile and implicit. A new physics means editing several assemblers (not open/closed). |
| A5 | **Materials are resolved by `inspect.signature`** of the physics constructor, merged with spec overrides; units are implicit. Air in 3D comes from constants (`RHO_AIR`, `C_AIR`) instead. | `ObjectsGeo.get_params`, `Cavity_coupling` | Two sources of truth, and errors appear late. |
| A6 | **Two parallel FE stacks.** 2D: `MeshManager` → `fem_objects` → assemblers. 3D: hand-written gmsh code, its own mesh reader (`read_p2_mesh`), its own P2-tet assembly. | `Meshing`, `Cavity_coupling`, `Exterior_dtn` | Duplicated infrastructure (meshing, reading, quadrature, sparse patterns). |
| A7 | **Algorithms locked inside studies.** `EigenStudy` (1369 lines) contains ARPACK, the GEP/QEP, multishift, covering, the DtN system, normalisation and summaries. `fast_linalg` lives on the side. | `Eigen_study`, `Time_study` | Solvers cannot be reused across physics (2D, Fourier, 3D) or benchmarked alone. |
| A8 | **The excitation is not a concept.** `force=1.0`, `force=True` (a boolean) or `Re, Le, n, u, Bl` keywords, `force_amplitude` + `force_direction="_z_"` + `force_signal`, or a hand-built `F` in 3D. The electrical circuit is coded inside `FreqStudy.solve_fsi`. | studies | Each study is driven differently, so there is no multi-driver drive and no reuse. |
| A9 | **No result model.** Studies keep `_results` lists of ad-hoc dicts. The HDF5 layout is implicit, split between `StudySaver` and three different readers (`snapshots()`, `load(kind)` → tuple, `load_run`). 3D bypasses it all (npz). The snapshot coordinate is `f`, `freq` or `t`. Multi-block sweeps are stitched by the user. | `Files_Saver`, `Files_Loader` | Every loading script is different. The file format has no version. |
| A10 | **The public/private boundary is blurred.** `Studies.__all__` is broken (missing commas). Examples import private helpers (`_gmsh`, `_h2`). Package names are not standard Python style (`Files_Loader`, `Finite_Elements_formulations`). | package roots | No stable API surface. |
| A11 | **Side effects.** `print` everywhere instead of `logging`. Global gmsh state. Scripts need environment hacks (`KMP_DUPLICATE_LIB_OK`, `NUMBA_CACHE_DIR`). | everywhere | Hard to embed in other code or run quietly. |
| A12 | **No tests.** pytest is a declared dependency, but there is no test suite. | — | Any refactor is unverified. |

### 1.2 User-facing issues (from the 47 example scripts)

- Five import roots, plus private helpers.
- Solver names describe the algorithm, not the question: `solve_meca_eigen_ARPAC`, `solve_meca_eigen_multishift`, `solve_meca_eigen_fourier`, `solve_fsi_eigen`, `solve_time_domain_rayleigh_nl`. The setup step has three names: `assemble_domains`, `assemble_meca`, `assemble_init`.
- Long spec names with cryptic fields: `DomainSpecMecaRayleigh(size=)`, `DomainSpecPML(t=, f_pml=, n=, alpha=)`.
- Boilerplate every user rewrites: the remesh-per-band loop with block saving (~60 lines), and in 3D the cut-outs, the `classify()` function and coil nodes found from curve ids.

---

## 2. Target architecture

### 2.1 Layers (dependencies only point down)

```
                ┌──────────────────────────────────────────────┐
  public API    │  loudpy  (import loudpy as lp)               │  re-exports only
                ├──────────────────────────────────────────────┤
  studies       │  Modes · FrequencyResponse · Transient       │  orchestration
                ├───────────────┬──────────────────────────────┤
  results/plot  │  results      │  plot                        │  Results, HDF5 schema, load(); plots read Results
                ├───────────────┴──────────────────────────────┤
  physics       │  structural · acoustic · fsi · radiation     │  turns a Model into
                │  electrical · modal (driver modes, cavity)   │  operator blocks (K, C, M, coupling, F)
                ├──────────────────────────────────────────────┤
  fem           │  elements (tri6_axi, tet10) · quadrature     │  discretisation, no physics names
                │  assembly (sparse patterns) · dofs · constraints │
                ├──────────────────────────────────────────────┤
  geometry      │  Mesh (dim-agnostic) · gmsh I/O · meshing2d  │
                │  meshing3d (cut-outs, classify) · Frame      │
                ├──────────────────────────────────────────────┤
  model         │  Model · Scene · regions · boundaries        │  declarative, no numerics
                │  materials · damping · drives                │
                └──────────────────────────────────────────────┘
  solvers  (side layer, numpy/scipy only)
           linear (SweepSolver) · eigen (arpack, qep, multishift, covering) · time (generalised-α, Newton)
```

Rules:
- `model` imports nothing numerical.
- `solvers` know nothing about physics (matrices in, vectors out).
- `physics` never touches files.
- `studies` is the only layer that combines them.
- `plot` only reads `Results`.

Each layer can be tested on its own.

### 2.2 Package layout

```
src/loudpy/
  __init__.py            public API (explicit __all__)
  model/
    model.py             Model  (2D axisymmetric: .geo + regions + boundaries)
    scene.py             Scene  (3D: enclosure + placed drivers + ground/radiation)
    regions.py           Solid, Air, PML
    boundaries.py        Clamped, Coil, FluidStructure, Radiation, Ground
    materials.py         Material, MaterialLibrary (materials.json), Air default
    damping.py           Rayleigh, Hysteretic, Undamped
    drives.py            Force, Voltage(circuit), Signal, Drive set for several drivers
  geometry/
    mesh.py              Mesh: nodes (n, dim), cell blocks by type, named groups
    gmsh_session.py      context manager around gmsh (no global state leaks)
    meshing2d.py         today's MeshManager (PML, grading, reuse signature)
    meshing3d.py         cut-outs (add_driver_cutout/front, carve_drivers), classify
    frame.py             Frame (origin + rotation): today's DriverFrame
  fem/
    elements/tri6_axi.py K, M, H, Q, FSI, PML, Fourier-m, NL kernels (today's FE_functions)
    elements/tet10.py    P2 tetrahedron (today's assemble_p2)
    quadrature.py
    assembly.py          sparse pattern + accumulation (today's assembly/core, kernels)
    dofs.py              DofMap (node tags ↔ dofs, free/blocked), constraints
  physics/
    base.py              Contribution protocol + registry
    structural.py        solid regions → K, M, C (damping model plugged in)
    acoustic.py          air/PML → H, Q
    fsi.py               fluid-structure coupling blocks
    radiation.py         DtN on an arc (2D) and on a half-sphere (3D)
    electrical.py        voice-coil circuit block (Re, Le, n, Bl)
    modal.py             DriverModes, CavityModes, modal coupling (today's Cavity_coupling core)
  solvers/
    linear.py  eigen.py  time.py
  studies/
    base.py              Study (has a Model, a Drive, a Solver), common solve/save
    modes.py             Modes
    frequency.py         FrequencyResponse (with built-in remesh bands)
    transient.py         Transient
  results/
    results.py           Results, Snapshot (.frequency / .time / .mode), probes
    schema.py            HDF5 layout, version attribute, readers for old files
    io.py                save(), load() (single file, block files or a directory)
  plot/
    field2d.py  field3d.py  curves.py  directivity.py
  _internal/             logging setup, utilities
tests/                   pytest: fast regression cases per layer
docs/ARCHITECTURE.md
```

### 2.3 Key design choices

1. **Composition instead of inheritance.** A study holds a `Model`, a `Drive` and a solver configuration. Regions and boundaries hold a reference to a mesh group; they don't inherit from mesh objects.
2. **Explicit selectors instead of substring matching.** Regions and boundaries name their mesh group exactly, or with an explicit glob (`group="glue_*"`). Orientation and clamped directions are parameters (`Clamped(..., dofs="rz")`, `FluidStructure(..., side="rear")`), not encoded in names. Unknown or unmatched names are an error at `mesh()` time, with the list of available groups.
3. **A registry of contributions (open/closed).** Every region or boundary type declares what it contributes to the system (`contribute(system, dofs)`). Assemblers stop filtering by `isinstance`. A new physics is one new class, with no edits elsewhere.
4. **Materials and damping are values.** `Material(rho, E, nu)` is a frozen dataclass, loaded once from `materials.json` with validation and SI units. Damping is a separate object (`Rayleigh(alpha, beta)`, `Hysteretic(eta)`), so `Solid` is one class instead of three. 2D and 3D share the same `Air`.
5. **Drives are objects.** `Force(1.0)`, `Voltage(2.83, Re=8, Le=4e-3, Bl=7)`, `Force(signal, fs=48_000)`. A dict `{driver_name: drive}` drives several drivers. Each drive knows how to build its right-hand side (and an extra electrical block for `Voltage`) for a given study.
6. **One result model.** Every study returns a `Results` with `.frequencies` / `.times` / `.modes`, `.field(name, at=…)`, `.probe(points)`, `.spl(at=…)`, `.mesh`. One versioned HDF5 schema is written by `save()` and read by `lp.load()`, which also stitches multi-block sweeps and reads the old files.
7. **One mesh model for 2D and 3D.** `Mesh` holds `nodes (n, dim)`, cell blocks per element type, and named groups. Both the 2D meshing and the 3D builders produce it, and both element families assemble from it.
8. **Quiet by default.** `logging` replaces `print` (`lp.set_verbosity("info")`). gmsh runs inside a context manager. Required environment settings (numba cache, OpenMP) are set by the package, not by the scripts.
9. **Tests first.** Regression tests freeze today's numbers before code moves: plate modes, Fourier clamped plate, rectangular cavity, single-driver coupling baseline, multi-driver symmetry and reciprocity, one FSI frequency point, one short transient. Each migration step must keep them green.

---

## 3. Public API (user point of view)

One import. Nouns for the things you model, the same verbs everywhere (`add`, `mesh`, `solve`, `save`, `load`), and studies named after the question they answer.

```python
import loudpy as lp

model = lp.Model("HPNEW-Sketch.geo", materials="materials.json")
model.add(
    lp.Solid("membranne", "Paper",  damping=lp.Rayleigh(), mesh_size=0.5e-3),
    lp.Solid("surround",  "Rubber", damping=lp.Hysteretic(eta=0.1)),
    lp.Air("subacou", mesh_size=5e-3),
    lp.PML("PML"),
    lp.Clamped("interface_constrained", dofs="rz"),
    lp.Coil("interface_forced"),
    lp.FluidStructure("interface_acou_meca_front"),
)
model.mesh(reuse=True)

modes = lp.Modes(model).solve(f_max=6000, orders=[0, 1, 2, 3])
resp  = lp.FrequencyResponse(model, drive=lp.Voltage(2.83, Re=8, Le=4e-3, Bl=7))
resp.solve(freqs, remesh=lp.PerWavelength(6)).save("sweep.h5")
trans = lp.Transient(model, drive=lp.Force(signal, fs=48_000), nonlinear=True).solve(duration=1.5)

res = lp.load("sweep.h5")
res.spl(at=(0.0, 0.3))
res.field("p", f=1000)
lp.plot.field(res, "p", f=1000)
```

3D, same pattern:

```python
hp = lp.Driver.from_modes("modes.h5", orders=[0, 1, 2, 3], f_max=6000)   # coil + wet faces from stored interfaces
scene = lp.Scene(materials="materials.json")
scene.add(lp.Cabinet(build_cabinet),          # OCC builder function or "cabinet.step"
          hp.place("top", at=(0.07, 0.58, 0)), hp.place("middle", at=(0.07, 0.45, 0)),
          lp.Ground(), lp.Radiation(margin=0.05))
scene.mesh(remesh=lp.PerWavelength(4), reuse=True)
res = lp.FrequencyResponse(scene, drive={"top": lp.Force(1), "middle": lp.Force(1)}).solve(freqs)
res.driver("top").excursion()
```

### 3.1 Rename map

| today | new |
|---|---|
| `Problem`, `add_sub_domain`, `add_interface` | `Model`, `add` |
| `DomainSpecMecaRayleigh / Hysteretic / Meca (size=)` | `Solid(material, damping=Rayleigh() / Hysteretic(eta) / None, mesh_size=)` |
| `DomainSpecAcou`, `DomainSpecPML(t=, f_pml=, n=, alpha=)` | `Air`, `PML(thickness=, tuned_to=, order=, strength=)` |
| `InterfaceSpecClamped("…_rz_…")` | `Clamped("…", dofs="rz")` |
| `InterfaceSpecForced`, `InterfaceSpecAcouMeca`, `InterfaceSpecDtN` | `Coil`, `FluidStructure(side=)`, `Radiation` |
| `EigenStudy.solve_meca_eigen_{ARPAC, multishift, fourier}` | `Modes(model).solve(method=, orders=)` |
| `solve_fsi_eigen`, `solve_acoustic_eigen_ARPAC` | `Modes(model, physics="coupled" / "acoustic")` |
| `FreqStudy.solve_meca / solve_fsi` + manual loop + remesh blocks | `FrequencyResponse(model, drive=).solve(freqs, remesh=)` |
| `TimeStudy.solve_time_domain_rayleigh_nl` | `Transient(model, drive=, nonlinear=True).solve()` |
| `FreqReader`, `EigenReader`, `TimeReader`, `Domain` | `lp.load(path)` → `Results` |
| `snap.f / snap.freq / snap.t` | `.frequency / .time` |
| `DriverModes`, `MultiDriver`, `DriverFrame` | `Driver`, `driver.place()`, `Frame` |
| `DriverCavityCoupling`, `ExteriorFE`, `coupled_sweep`, `solve_forced` | internal (`physics.modal`, `physics.radiation`), used by `FrequencyResponse` |
| `RHO_AIR`, `C_AIR` | `lp.materials["Air"]` |

---

## 4. Migration plan

The refactor proceeds by strangling the old code: build the new layers next to the old ones, move the code into them (moved, not rewritten), and delete each old module once nothing uses it. The numbers stay identical at each step.

| Phase | Content | Done when |
|---|---|---|
| 0 ✅ | Test harness: pytest + regression fixtures from the current code (`tests/`, see below) | the tests reproduce today's numbers |
| 1 ✅ | `model/` (Model, regions, boundaries, materials, damping, drives) + `geometry/` (Mesh, gmsh session, naming, meshing2d moved, meshing3d, Frame) | the tests build their models through `lp.Model` |
| 2 ✅ | `fem/` (move kernels, assembly, dofs) + `physics/` with the role registry | the assembled matrices are identical to the old ones |
| 3 ✅ | `solvers/` (eigen, linear, time moved out of the studies) | regression tests green |
| 4 ✅ | `results/` (schema, save, load, old-file readers) + `plot/` on `Results` | `lp.load` reads every former file to the same numbers as the former readers |
| 5 ✅ | `study/` (Modes, FrequencyResponse with remesh bands, Transient) returning `Results` | 2D examples (Running and Loading scripts) migrated |
| 6 ✅ | 3D: `Scene`, `Driver`, `physics.modal`, `physics.radiation` | Coupling_3D examples migrated, results unchanged |
| 7 ✅ | Delete the old packages, docs, ruff + CI | only the new API remains |
| 8 ✅ | Evaluated data (Data, Field, selections, quantities), plot package, exports | every figure and export goes through the result objects |

### Phase 0 — done

`tests/` freezes today's numbers in 24 cases. `pytest` runs the 19 fast ones (~10 s); `pytest --slow` runs all 24, with the 3D ones (~30 s).

- **Adapter:** the tests call loudpy only through `tests/support/current_api.py`. Later phases rewrite that file for the new API; the tests and references stay as they are.
- **Comparisons survive reordering:** nodal fields are matched by node coordinates, and mode shapes are compared with the MAC. A new DOF or node ordering doesn't break them.
- **Fixed inputs:** the meshes are committed (`tests/data/meshes/`).
- **Backward compatibility:** old-format result files are kept in `tests/data/legacy/`. The new `lp.load()` must still read them.
- **Sensitivity checked:** E + 1e-6 fails every 2D numerical test, and c + 1e-5 fails every cavity-mode test.

Reproducibility issues found (to fix in the phase named):

| Issue | Where | Effect | Fix in |
|---|---|---|---|
| gmsh meshes with `General.NumThreads = cpu_count()` | `Meshing/MeshManager.py` | The same `.geo` gives different triangulations from run to run (~1e-6 on frequencies). The 3D HXT meshing (`Mesh.Algorithm3D = 10`) is multithreaded too. | 1 (a `deterministic=True` meshing option, used by the tests) |
| `eigsh` without a start vector | `CavityModes.solve` (and ARPACK calls generally) | Cavity-mode results vary ~1e-6 between runs, so the 3D tests use rtol 1e-5 | 3 ✅ fixed: every ARPACK call starts from `solvers.eigen.start_vector`; `RTOL_3D` is 1e-9 |

### Phase 1 — done

New packages: `loudpy.geometry` and `loudpy.model`. The new names are exported from `import loudpy as lp`: `Model`, `Solid`, `Air`, `PML`, `Clamped`, `Coil`, `FluidStructure`, `Radiation`, `Rayleigh`, `Hysteretic`, `Undamped`, `Force`, `Voltage`, `Material`, `materials`, `load_materials`, `Frame`, `Mesh`.

- **Exact selection:** names select physical groups by whole tokens, not substrings (`geometry/naming.py`): "coil" no longer matches "coil_former". Explicit globs work with or without the physics prefix (`"glue_*"`). `Model.selection()` (called by `mesh()`) fails on a name that selects nothing, or a group claimed by two regions, and lists the available groups.
- **Materials:** a validated `MaterialLibrary` of frozen `Material` values (SI units). Missing properties give an explicit error. A material can be defined in the script (`lp.Material("Steel", …)`). Constructor-signature inspection is gone from the new path.
- **Damping** is a separate object, so there is one `Solid` instead of three spec classes.
- **Deterministic meshing:** `Model.mesh(deterministic=True)` meshes on one thread and gives identical meshes (test).
- **Moves (history kept with `git mv`):** `MeshManager` is now `geometry/meshing2d.py`, with an injectable name matcher and the deterministic option. The 3D builders and `DriverFrame` (now `Frame`) moved from `Cavity_coupling` / `Exterior_dtn` to `geometry/meshing3d.py` and `geometry/frame.py`. All old import paths are re-exported; the 74 loudpy import statements of the examples all resolve.
- **Bridge:** `model/_legacy.py`. Until the studies are rebuilt (phases 2–5), `Model.as_problem()` returns the equivalent `Problem` for `EigenStudy` / `FreqStudy` / `TimeStudy`. A test checks that `Model` builds the same physics objects (class, parameters, cells, nodes) as `Problem` for the plate, the speaker and the speaker with air + PML.
- **Small fixes:** the broken `__all__` of `loudpy.Studies` and `loudpy.assembly`. `from loudpy import *` works.
- **Scene (3D):** moved to phase 6, together with `Driver`, so that the 3D declarative API and its physics are designed together.

Tests: 51 (24 regression through `lp.Model`, 27 for the model layer), all passing. The committed meshes and references are unchanged.

### Phase 2 — done

Two new layers. `fem` holds no physics names. `physics` holds the physics objects and their operators. All the code was moved with `git mv`, and the numerics weren't rewritten.

- **`loudpy.fem`:**
  - `quadrature` and `shape` (former `Gauss_Quadratures`, `FE_kernels`).
  - `elements/tri6_axi/`: the element kernels, renamed by what they compute (`stiffness`, `mass`, `acoustic_stiffness`, `acoustic_mass`, `pml_stiffness`, `pml_mass`, `coupling`, `edge_force`, `total_lagrangian`, `fourier`).
  - `batch`: element loops; `sparse` / `sparse_kernels` / `indexing`: the COO → CSR scatter map.
  - `assembly.ElementAssembler`: node numbering and CSR pattern over T6 regions.
  - `dofs.DirichletElimination`: blocked DOFs. It was duplicated in the linear and nonlinear assemblers and is now shared.
- **`loudpy.physics`:**
  - `registry`: roles `solid`, `air`, `pml`, `clamped`, `coil`, `fluid_structure`, `radiation`, and `select(objects, *roles)`.
  - `parts`: the physics objects hold their mesh group by composition (`SolidPart`, `AirPart`, `PMLPart`, `ClampedEdge`, `CoilEdge`, `CouplingEdge`, `RadiationEdge`). They no longer inherit from mesh classes.
  - `structural`, `acoustic`, `coupling`, `forcing`, `radiation`: the assemblers. Every `isinstance` filter in the assemblers and studies became a role selection.
- **PML metadata:** a PML part carries its layer geometry (`geometry.naming.PMLLayer`, parsed once when the part is built). The acoustic assembler no longer parses surface names.
- **Former classes:** `SubDomain*` / `Interface*` are thin subclasses of the parts, with their former constructors. Their names stay, because the result files store the class name as the physics key (until phase 4). `material_parameters` (formerly `get_params`) lives in `physics.parts`.
- **Shims:** `loudpy.assembly`, `loudpy.assembly.assemblers.*`, `loudpy.assembly.dtn` and `loudpy.Finite_Elements_formulations(.FE_kernels, .Gauss_Quadratures)` re-export the new modules. The 74 example imports still resolve.
- **Removed dead code:** `FsiAssembler._build_fluid_neighbor_map_csr` (never called).

Gate: `tests/test_operators.py`, 18 operators on the committed meshes. It covers K, M, C, F, H, Q (with PML), Cam, Fourier m = 1 (K, M, C, T), the nonlinear tangent (K_T, M, C, F_int), the reduced system, and the full structure + air + DtN system. The references were written by the phase-1 code. The phase-2 code reproduces them at rtol 1e-12, and a 1e-9 change of E fails exactly the stiffness-dependent ones.

Tests: 69 passing.

### Phase 3 — done

New layer `loudpy.solvers`: algebra on assembled matrices, with no physics and no files.

- **`solvers.linear`** (former `Studies/fast_linalg.py`, moved with `git mv`): `factorize`, `SweepSolver` and `QEPShiftInvert`.
- **`solvers.eigen`:** the algorithms moved out of `EigenStudy` (1369 → 851 lines) as plain functions with clear names. They cover the GEP / QEP shift-invert solves (`solve_gep`, `solve_qep`, `solve_qep_at`), several shifts over a band (`multishift`, `covering`, `default_shifts`), merging shifts (`mac_deduplicate`, `weighted_mac`), the single physical filter (`filter_modes`), normalisation (`normalise_modes`, `normalise_symmetric`) and the summaries. `EigenStudy` keeps its former private method names as aliases.
- **`solvers.time`:** the generalised-α / Newton–Raphson integrator moved out of `TimeStudy` (317 → 197 lines). It is `generalized_alpha_newton(tangent, …)` with the structural tangent as a callback, plus `GeneralizedAlpha` (parameters from ρ∞) and `free_dof_reducer`.
- **Reproducible eigen solves:** every ARPACK call starts from a fixed vector (`start_vector`, seed 0): `arpack_eigs`, the acoustic `eigsh` and `CavityModes.solve`. Identical runs give identical modes. The 3D references were regenerated on purpose; they changed only by the former noise (≤ 5e-6), and `RTOL_3D` went from 1e-5 to 1e-9. The 2D eigen tolerances stay at 1e-7, ARPACK's convergence tolerance: a 1e-15 change of a matrix can move an eigenvalue by up to that much.
- **Bug fixed:** `EigenStudy._print_acou_summary` lacked `self` / `@staticmethod`, so `solve_acoustic_eigen_ARPAC` failed when printing its summary. It is now a staticmethod alias of `solvers.eigen.print_acoustic_summary` (not covered by a test yet).
- **Shim:** `loudpy.Studies.fast_linalg`.

Tests: 77 passing. That includes `tests/test_solvers.py`, which checks the solvers layer alone on problems with known answers: undamped frequencies, Rayleigh-damped ζ and the normalisation, gap-free covering, MAC deduplication with degenerate pairs, the half-plane filter, reproducible ARPACK, the sweep solver, and second-order accuracy of generalised-α against an exact solution.

### Phase 4 — done

New layer `loudpy.results`, plus `loudpy.plot`. Both are exported as `lp.load`, `lp.Results` and `lp.plot`; `lp.plot` is imported on first use, so `import loudpy` doesn't load matplotlib.

- **One result model:** `FrequencyResults`, `ModeResults` and `TimeResults` share one interface:
  - `.frequencies` / `.times`, `.names` (`"u"`, `"p"`, `"v"`, `"a"`; the former `u_meca` / `p_acou` still work);
  - `.field(name, f= / t= / index=)` returns a `FieldView` (the mesh and values at one entry);
  - `.probe(point, name)` gives the values at the nearest node for every entry, and `.spl(at=)` (RMS by default, `rms=False` for the former peak convention);
  - `.extra("zeta" | "I_coil" | …)` and `.save(path)`.
  `ModeResults` adds `.zeta`, `.sets`, `.orders`, `.order(m)`, `.set(label)` and `.mode(k)`.
- **Blocks:** a result is a list of blocks, one per mesh or per mode set. That's how a sweep remeshed per band, or several azimuthal orders, stay one object: `probe`, `spl` and `frequencies` run across the blocks, and `field` finds the right one.
- **Format 2** (`results/schema.py`, `loudpy_format = "2.0"`): self-describing. The physics are named by role (`meshes/solid`, `meshes/air`), never by Python class name, and the written order of fields and mesh groups is kept.
- **Format 1** (`results/legacy.py`) is read by `lp.load` (`lp.load(file | folder | glob | list)`). Several files merge into one result sorted by frequency or time, with an explicit list keeping its order.
- **Checked on real data:** the 14-block FSI_elec sweep (347 frequencies on 14 meshes) gives probe pressures bit-identical to the former per-snapshot reader loop, about 3× faster. The coupled DtN modes (both fields, both meshes) and both in-vacuo sweeps are bit-identical too.
- **`loudpy.plot`:** `field`, `mode`, `spl`, `response`, `history`, taking a `Results` or a `FieldView` (components `mag`, `r`, `z`, `theta`, `real`, `imag`, `phase`, `spl`). Built on the former `Plotter` functions, which stay.
- **`geometry.Mesh`:** added the `tri3` cell type, `triangles()`, `nearest()` and `rows_of()`.
- **Kept for now:** the studies still write format 1 (`Files_Saver`), and the former readers (`Files_Loader`) stay. Phase 5 rebuilds the studies so that they return `Results` and write format 2, and migrates the example Loading scripts with their Running scripts (most of their result files are not in the repository, so they are migrated where they can be run).

Tests: 87 passing. `tests/test_results.py` covers `lp.load` vs the former readers on the legacy files (exact), format-2 round trips, no class names in format 2, multi-file sweeps (folder, glob, list), the `Results` interface, and the plots.

### Phase 5a — the new studies

New package `loudpy.study`, exported as `lp.Modes`, `lp.FrequencyResponse`, `lp.PerWavelength` and `lp.Transient`. It's named `study`, not `studies`: on macOS's case-insensitive file system, `studies` is the same folder as the former `Studies`. It can be renamed when the former package goes in phase 7.

- **`lp.Modes(model, physics="structure" | "coupled" | "acoustic").solve(f_max, f_min, orders=, method="multishift" | "arpack", …)`** returns a `ModeResults`. Structural azimuthal orders, coupled structure + air + exact DtN (options passed to the covering solver), and acoustic modes.
- **`lp.FrequencyResponse(model, drive=lp.Force(…) | lp.Voltage(…)).solve(freqs, remesh=lp.PerWavelength(…))`** returns a `FrequencyResults`. It solves the structure alone, or structure + air when the model has air. With `PerWavelength` (bands= or every=, air elements per λ with a cap, PML thickness in λ and elements across it), it builds one mesh per band and one result block per band, then restores the model afterwards. That replaces the ~60-line remesh / save-block loop of the FSI scripts.
- **`lp.Transient(model, drive=lp.Force(signal, fs=…)).solve(n_steps= | duration=)`** returns a `TimeResults`. The coil direction comes from the `Coil` boundary instead of a `"_z_"` string.
- **`study.save(path)`** writes result format 2.
- **Underneath:** results are built in memory (`study/records.py`: DOF vectors → node-indexed fields on per-physics meshes, the same arrays the former saver wrote). The former `EigenStudy` / `FreqStudy` / `TimeStudy` remain the engines for system building and the solve loops, through `model.as_problem()`, until they are folded in.
- **Gate:** the regression tests (eigen, frequency, transient, files) now run through the new studies against unchanged references, and all pass. New `tests/test_study.py` covers band splitting, a remeshed coupled sweep (meshes per band, wavelengths, model restored), drive checks, the ARPACK method against multishift, and acoustic modes (the phase-3 fix, now tested).
- **Legacy files:** `tests/data/legacy/` (format 1) is fixed data now. The update run no longer rewrites it, because the new studies write format 2.

Tests: 92 passing.

### Phase 5b — the 2D examples

Every 2D example now uses the new API only: `Eigen_study/Meca`, `Fourier` (incl. the plate validation and the 3D views), `FSI_DtN`, `Frequency_study/Meca`, `FSI`, `FSI_elec`, `Time_study/Multitone`, and `generate_meshes.py`. They went from about 2,000 to 1,190 lines. No `Problem`, `DomainSpec*`, `InterfaceSpec*` or `Files_Loader` remains outside `Coupling_3D` (phase 6).

- **The remesh loops** of FSI and FSI_elec became `lp.PerWavelength(every=30 | bands=[…], …)`. Proof that the rule is the same: `generate_meshes.py` reuses all 14 existing FSI_elec band meshes built by the former loop, because their signatures match exactly.
- **Each Running script writes one file in format 2** (`modes.h5`, `eigen_fourier.h5`, `sweep.h5`, `time.h5`). `study.save(path, **metadata)` keeps extra parameters, such as the multitone `t_ramp` and tone frequencies.
- **`DriverModes.from_eigen_h5`** (3D coupling) reads format-2 eigen files too, through `lp.load`, and gives identical driver modes. Format-1 files still go through the former reader.
- **`import loudpy` sets `KMP_DUPLICATE_LIB_OK`** (the OpenMP clash between scipy and numba), so the scripts no longer do it.
- **Added for the scripts:** `FieldView.restrict(region)`, `FieldView.at_nodes(tags)`, `Results.boundary(name)`, `lp.plot.modes_grid`, `lp.plot.deformed_boundary` and `lp.plot.coupled_modes_grid`.

Verification, per example:
- Every Running script ran in a reduced copy (fewer frequencies, modes or steps), output to a scratch folder.
- Every Loading script ran on the existing full results, converted to the file the new Running script writes: 81 + 373 + 876 modes, sweeps of 3000, 307 and 347 frequencies.
- The multitone pair ran end to end on a reduced signal, animation included.
- The plate validation prints the same table as before (worst 0.575 %).

Issues in the former examples, fixed by the migration:
- **FSI_elec** plotted the impedance with `U_drive=1.0` while the drive is √8 V, so |Z| was 2.83× too low (below Re at low frequency). It now uses the stored drive voltage.
- **Frequency_study/Meca** unpacked `extract_interface(...)` into 2 values (it returns 3), so it failed at the interface figure.
- **Time_study/Multitone** Loading hard-coded a `sys.path` entry to a home folder.

Tests: 94 passing.

### Phase 6a — 3D scenes (library)

New package `loudpy.scene`, exported as `lp.Driver`, `lp.Profile`, `lp.Scene` and `lp.Cabinet`. The scope was decided with the user: configurations A, B and C; the cabinet geometry from a builder function; the 3D fields streamed to the result file.

- **`lp.Driver.from_modes(file, orders, f_max, geometry=, rear=lp.Profile(curves, start), front=…)`:** the modes are read from an eigen file (format 2 or 1). The coil is the boundary `interface_forced` of that file, which gives the same 13 nodes as the former curve-37 lookup. The wet profiles come from the driver's 2D geometry. `driver.place(name, at=, axis=, ref=)` gives a `Frame`.
- **`lp.Scene(cabinet, drivers)`:** `cabinet(occ)` returns `lp.Cabinet(inner=, cuts=, outer=, openings=)`. `scene.cavity(...)` is configuration A, adding `scene.exterior(...)` gives B, and `scene.merged(...)` is C. `scene.mesh(folder, reuse=True | "always" | False)` does the following:
  - adds the driver cut-outs and front plugs;
  - sizes the half-space DtN hemisphere from the cabinet vertices and the drivers;
  - classifies the wet surfaces per driver (both baffle planes for C);
  - meshes per band;
  - writes a signature next to each mesh (cabinet source, drivers, mesh settings only).
- **`lp.FrequencyResponse(scene, drive=lp.Force | {name: lp.Force}).solve(freqs, store=path)`** returns `SceneResults`. It keeps the stacked modal amplitudes `q`, the DtN coefficients, and the cavity modal pressures (A, B). The FE pressure field is streamed to `store` (format 2, one block per band mesh) and read back lazily. `lp.Modes(scene)` gives the coupled driver + cavity modes (A).
- **`SceneResults`:** `drivers`, `driver(name)` (`excursion()`, `displacement(at, theta)`, `coil_excursion()`, `surface(f)`), `far_field(points)` (exact DtN), `cavity_pressure(point)`, and `spl(at)` (far field outside the hemisphere, FE field inside, cavity otherwise). The driver modal bases and cavity modes are stored as attachments, so a result file is self-contained.
- **Format 2 additions:** attachments, `results_class`, lazy datasets above 5·10⁶ values (`LazyArray`), and `StreamWriter`.
- **Gate (`tests/test_scene.py`):**
  - Configuration A on the committed test boxes reproduces the existing references at 1e-9 (one driver; the two-driver transfer matrix; the coupled modes).
  - B and C on small committed scenes match a direct computation with the low-level code (`ExteriorFE` + `coupled_sweep`): amplitudes at 1e-10, the streamed field, and the far field.

Tests: 99 passing.

### Phase 6b — the 3D examples

All three `Coupling_3D` examples run on `lp.Scene`; the `Build_*.py` scripts are gone (a scene meshes itself on the first run and reuses the meshes afterwards). Each example has one geometry module (`enclosure.py`, `labyrinth.py`, `column.py`) holding the dimensions, the cabinet builders and `driver()`; `enclosure.py` also has `sealed_scene()`, `exterior_scene()` and `vented_scene()`, shared by the running and plotting scripts.

- `Driver_in_cavity`: A (`Running_script.py` → `coupled.h5`, `coupled_modes.h5`), B (`Running_exterior.py` → `exterior.h5`) and the merged vented box (`Running_vented.py merged` → `vented_merged.h5`). The hybrid vented model (modal inside + 1D port + exterior FE) stays on the low-level toolkit (`Build_vented.py`, `Running_vented.py hybrid`). `Recover_vented.py` is gone (it rebuilt results from the former field file; the pressure field is now streamed to the result file).
- `lp.Cabinet` gained `opening_size` / `opening_margin`: the walls around the openings (ports) are the group `openings`, refined like the former `port_walls`.
- `SceneResults`: the modal force of the drive is stored (attachment `drive`), so `driver(name).in_vacuo()` and `kinetic_energy(by_order=True)` need no recomputation; `cavity_frequencies`.

### Phase 7 — public API, plotting, documentation

- **Top-level namespace:** `import loudpy as lp` exports the new API only (model, geometry, studies, scenes, results, `lp.spl`, lazy `lp.plot`), with `__version__`. The former names (`DomainSpec*`, `InterfaceSpec*`, `MeshManager`, `PhysicalEntitiesManager`, FE kernels) are no longer re-exported; they stay importable from their modules because the studies' internal engines still use them.
- **Plotting:** `loudpy.plot.style` is the one place for the figure style (serif STIX fonts, Okabe–Ito colours, a colormap per quantity, labels with units, standard sizes, log frequency axes, safe saving). `lp.plot` and the former `loudpy.Plotter` apply it (the conflicting rcParams blocks were removed); every example calls `lp.plot.style.use()`. SPL is RMS everywhere (`lp.spl`). New `lp.plot.directivity`.
- **Documentation:** Sphinx in `docs/` (Furo theme): getting started, user guide (models, studies, results, plotting, 3D scenes, examples), reference (API via autodoc, file format, architecture, conventions). Build: `python3 -m sphinx -b html docs docs/_build/html`.
- **Consistency:** `Running_scrip.py` → `Running_script.py` (Frequency_study); docstrings on every public class.
- **Old packages deleted (phase 7b).** The engines were ported into the layers, then `Studies`,
  `Files_Loader`, `Files_Saver`, `SubDomains`, `Interfaces`, `Meshing`, `assembly`,
  `Finite_Elements_formulations`, `Plotter`, `SimObjects`, `PhysicalEntitiesManager` and
  `ObjectsGeo` were removed:
  - `geometry.meshing2d` takes `DomainSizing` records; the mesh signatures are unchanged, so every
    saved mesh stays reusable;
  - `Model.mesh()` builds the physics parts itself (`model.parts`, `model.boundary_nodes()`);
  - `physics.systems`: the assembled structural, coupled, Fourier, acoustic and DtN systems (the
    part of the former studies that built K, M, C, H, Q, Cam and the symmetric DtN form);
  - `study.frequency / modes / transient` solve on `physics.systems` and build `Results` directly;
  - 3D: `physics.modal` (cavity modes, driver modes, coupling), `physics.exterior` (half-space
    DtN, exterior FE, coupled sweep), `fem.tet10` (P2 tetrahedra), `results.surfaces` (pyvista
    datasets); `Materials_Bank` became `loudpy/data`.
  All regression references are unchanged (2D at 1e-12 / 1e-7, 3D at 1e-9).
- **Messages:** `logging` (logger `loudpy`), `lp.set_verbosity("info" | "debug" | "warning")`;
  progress at INFO, solver details and mode tables at DEBUG.
- **Tooling:** ruff configuration in `pyproject.toml`; `.github/workflows/tests.yml` (ruff
  errors, pytest, docs with warnings as errors).

### Phase 8 — evaluated data, plotting, exports

The plotting was rebuilt around what is plotted, in the spirit of COMSOL's results tree (where →
what → how) with the brevity of xarray / pandas.

- **Quantities** (`results.quantities`): one name, label, symbol, unit and colormap family per
  quantity (`p`, `spl`, `u_z`, `v_r`, `Z`, `zeta`, coordinates `f`, `t`, `angle`, `s` …). Every axis
  label, colour bar and export header comes from there.
- **Selections** (`lp.Point`, `lp.Points`, `lp.Polar`, `lp.Arc`, `lp.Line`): where to evaluate; 2D
  meshes are interpolated linearly on the P2 triangles split through their mid-side nodes
  (`Mesh.interpolation`), 3D meshes use the nearest node.
- **`lp.Data`**: values over named coordinates with a quantity — `sel` / `isel` (values, lists,
  ranges), `abs`, `phase`, `db`, `spl`, `normalize`, `velocity`, `acceleration`, `spectrum`,
  `impulse_response`, arithmetic; `to_pandas`, `to_xarray`, `export(.csv | .mat | .npz)`, `plot()`.
- **`lp.Field`**: values on a mesh at one entry — `component`, `magnitude`, `spl`, `at_phase`,
  `restrict`, `at(selection)`, `revolve` (3D surface of an azimuthal order); `to_pandas`,
  `to_pyvista`, `export(.vtu | .csv | .mat | .npz)`, `plot()`.
- **Results** return them: `spl`, `pressure`, `displacement`, `velocity`, `acceleration`,
  `probe(name, at)`, `scalar`, `impedance`, `field`; `ModeResults.table()`; `res.export(.mat | .npz)`.
  Scene results: `driver(name)` returns Data (`excursion`, `kinetic_energy(by_order=True)`,
  `displacement`), `in_vacuo()` a DriverResponse, `driver_surfaces`, `cavity_field`, `cavity_mode`.
- **`loudpy.plot`**: `curves` (line, map, polar), `fields` (2D sections with outlines, deformed
  shapes, mirror, mode grids, mode chart, coupled modes, boundary profiles), `view3d` (pyvista
  scenes in matplotlib axes with matplotlib colour bars), `scene3d` (`scene_view`: cut-away SPL
  views of scene results), `animate`; `style` (one style; labels from the quantities).
- **Examples**: every Loading / Plot script uses only this API (no raw matplotlib for standard
  plots); the four 3D scene scripts share `scene_view`; each writes CSV / VTU exports next to its
  figures' data. The `Plotter` package is gone.
