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
| 1 | `model/` (Model, Scene, regions, boundaries, materials, damping, drives) + `geometry/` (Mesh, gmsh session, meshing2d moved, meshing3d, Frame) | the 2D examples mesh through `Model` |
| 2 | `fem/` (move kernels, assembly, dofs) + `physics/` with the contribution registry | the assembled matrices are bit-identical to the old ones |
| 3 | `solvers/` (eigen, linear, time moved out of the studies) | regression tests green |
| 4 | `results/` (schema, save, load, old-file readers) + `plot/` on `Results` | all Loading scripts use `lp.load` |
| 5 | `studies/` (Modes, FrequencyResponse with remesh bands, Transient) | 2D examples migrated |
| 6 | 3D: `Driver`, `Scene`, `physics.modal`, `physics.radiation` | Coupling_3D examples migrated, results unchanged |
| 7 | Delete the old packages, docs, ruff + CI | only the new API remains |

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
| `eigsh` without a start vector | `CavityModes.solve` (and ARPACK calls generally) | Cavity-mode results vary ~1e-6 between runs, so the 3D tests use rtol 1e-5 | 3 (fixed `v0` in `solvers.eigen`, then tighten `RTOL_3D`) |
