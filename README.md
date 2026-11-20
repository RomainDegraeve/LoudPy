
![description](LoudPy_exemples/Field_Response_8970Hz_Symmetric_Clean.png)

# LoudPy Beta

LoudPy is an open-source Python API dedicated to loudspeaker finite element simulation, released under the [GPL-3.0 license](https://www.gnu.org/licenses/gpl-3.0.html).

> ⚠️ **Beta version** — The API is still under active development. Documentation: `docs/` (Sphinx, see [Documentation](#documentation)).
## Overview

Designing a loudspeaker with FEM means choosing between complex general-purpose tools that
require deep expertise, or expensive commercial software. LoudPy is a free, Python-based
alternative built specifically for loudspeaker/transducer engineers and researchers.

Starting from a CAD file of your loudspeaker, LoudPy lets you:
- run **harmonic frequency-domain simulations** (structural, acoustic, or fully coupled FSI)
- extract **eigenfrequencies** and mode shapes of the mechanical structure
- simulate **nonlinear behaviour** 
- evaluate results at points, arcs and lines, plot them as consistent scientific figures, and
  export them (NumPy, pandas, CSV, MATLAB, ParaView)

The finite element formulations are entirely custom — this project does not depend on any 
external FEM library. Numerical computations rely exclusively on **NumPy**, **SciPy**, and 
**Numba**, while meshing is handled by **Gmsh**.

This project was developed by Romain Degraeve as part of the
[IMDEA Master's programme in Acoustics](https://iags.univ-lemans.fr/en/education-programs/master-s-degrees-in-acoustics/parcours-en-anglais/imdea.html#Generalinformation1-1)
(International Master Degree in Acoustics).

## Installation

The easiest way to install LoudPy is via pip:

```bash
pip install "loudpy[all]"        # with plotting (matplotlib, pandas) and 3D views (pyvista)
```

The core (`pip install loudpy`: NumPy, SciPy, Numba, h5py, gmsh) runs every study; the extras add
`loudpy[plot]` and `loudpy[3d]`.

For development installation, see the [Developer Setup](#developer-setup) section below.

## Quick start

```python
import numpy as np
import loudpy as lp

model = lp.Model("LoudPy_exemples/Geometries/Loudspeaker/HPNEW-Sketch.geo")
model.add(lp.Solid("membranne", "Paper"), lp.Solid("surround", "Rubber"), lp.Solid("coil", "Copper"),
          lp.Clamped("interface_constrained"), lp.Coil("interface_forced"))
model.mesh("mesh.msh")

lp.set_verbosity("info")                              # progress messages (silent by default)
modes = lp.Modes(model).solve(f_max=20000)            # eigenmodes
modes.table()                                         # f, ζ (pandas)
modes.mode(0).plot(deformed=True)                     # the first mode shape

res = lp.FrequencyResponse(model, drive=lp.Force(1.0)).solve(np.geomspace(20, 20000, 200))
u = res.displacement(at=(0.0, 0.0), component="z")    # labelled data: u_z over f, with units
u.plot()                                              # scientific figure, labels from the quantity
u.export("cone_tip.csv")                              # for Excel / MATLAB / Origin (or .mat, .npz)
np.asarray(u)                                         # plain NumPy for your own post-processing
```

A model (or a 3D `lp.Scene`) goes into a study (`Modes`, `FrequencyResponse`, `Transient`); the
study returns results, saved with `study.save(path)` and read back with `lp.load(path)`. Evaluating
results (`res.spl(at=…)`, `res.field("p", f=…)`, at points, `lp.Arc`s or `lp.Line`s) gives labelled
data (`lp.Data`, `lp.Field`) that plot themselves and export to NumPy, pandas, CSV, MATLAB and
ParaView.

## Examples

The `LoudPy_exemples/` folder contains ready-to-run scripts (run them from the repository root).
Each example has a `Running_*.py` (writes `Results/Files/`) and `Loading_*.py` / `Plot_*.py`
scripts (write `Results/Figures/`).

| Example | Description |
|---|---|
| `Eigen_study/Meca` | Structural modes of the driver in vacuo |
| `Eigen_study/Fourier` | Azimuthal-order (non-axisymmetric) modes, 3D views, plate validation |
| `Eigen_study/FSI_DtN` | Coupled structure + air modes with an exact radiation condition |
| `Frequency_study/Meca` | Mechanical-only frequency sweep |
| `Frequency_study/FSI` | Coupled structural–acoustic sweep, air remeshed per wavelength |
| `Frequency_study/FSI_elec` | The same, voltage-driven (impedance, coil current) |
| `Time_study/Multitone` | Nonlinear time-domain simulation, multi-tone |
| `Coupling_3D/Driver_in_cavity` | Driver in a 3D enclosure: cavity modes, radiation on a rigid ground, vented box |
| `Coupling_3D/Labyrinth` | Folded transmission line on a rigid ground |
| `Coupling_3D/Column_labyrinth` | Three drivers in a column with a folded line (multi-driver) |

## Documentation

```bash
pip install sphinx furo
python3 -m sphinx -b html docs docs/_build/html      # then open docs/_build/html/index.html
```

## Publication

LoudPy will be presented at Acousticum 2026 — (ref. 226).

## Contributing

If you want to contribute to this open-source project, feel free to reach out:

- Email: [rom2graeve@gmail.com](mailto:rom2graeve@gmail.com)
- LinkedIn: [Romain Degraeve](https://www.linkedin.com/in/romain-degraeve-558073290/)

---

## Technical Details

### Workflow

LoudPy handles 2D axisymmetric geometries. The full pipeline is:

1. **Import** — a STEP file is loaded and converted to BREP format by LoudPy
2. **Physical entity assignment** — the [Gmsh standalone app](https://gmsh.info) is used to
   assign named physical groups: closed curve loops define subdomains, individual curves define
   boundaries. This produces two paired files: a `.brep` (geometry) and a `.geo` (named entities)
3. **Meshing** — LoudPy loads both files, links named entities to materials and boundary
   conditions, and generates the mesh via the Gmsh Python API (bundled — no separate install needed)
4. **Solve** — the chosen study assembles and solves the FEM system
5. **Results** — saved to a self-describing **HDF5** file (`.h5`, loudpy result format 2) and read
   back with `lp.load`; evaluated, plotted and exported from the result object

### Studies

| Study | Description |
|---|---|
| `lp.Modes(model)` | Structural modes (axisymmetric, or azimuthal orders m = 0, 1, 2…), complex with damping |
| `lp.Modes(model, physics="coupled")` | Coupled structure + air modes with an exact (DtN) radiation condition |
| `lp.Modes(model, physics="acoustic")` | Acoustic modes of the air (rigid structure) |
| `lp.FrequencyResponse(model, drive=…)` | Harmonic response: structure, or structure + air (force or voltage drive), air remeshed per wavelength |
| `lp.Transient(model, drive=lp.Force(signal, fs))` | Geometrically nonlinear time response |
| `lp.FrequencyResponse(scene)` | 3D enclosures with one or several drivers (cavity modes, exterior air on a rigid ground) |

### Physics

- **Structural domain** — 2D axisymmetric linear elasticity, T6 elements
- **Acoustic domain** — pressure-based formulation, T6 elements
- **Fluid-structure coupling** — `lp.FluidStructure` boundaries, assembled automatically
- **Damping** — Rayleigh (`α`, `β`) and hysteretic (`η`), both supporting non-proportional formulations
- **Radiation** — Perfectly Matched Layer (`lp.PML`) or exact Dirichlet-to-Neumann condition (`lp.Radiation`)

### Material Library

Built-in materials (`materials.json`): `Polypropylene`, `Paper`, `Copper`, `Rubber`, `Kapton`,
`PhenolicCloth`, `Air`, `SolidGlue`. Custom materials can be added to the JSON file or defined
directly in the study script.

### Post-processing

- `res.spl(at=…)`, `res.pressure(at=…)`, `res.displacement(at=…, component="z")`, `res.probe(name, at=…)`
  at points, `lp.Points`, `lp.Arc`, `lp.Line` → `lp.Data` (values with coordinates and units)
- `res.field(name, f=…)` → `lp.Field` (values on the mesh): components, SPL, regions, interpolation
- `.plot()` on any of them, and `lp.plot` (mode grids, directivity, 3D views, animations), one
  scientific style for every figure
- `.to_pandas()`, `np.asarray(…)`, `.export("x.csv" | "x.mat" | "x.npz" | "x.vtu")`, `res.export("all.mat")`

---

## Developer Setup

LoudPy uses [PDM](https://pdm-project.org) for package management.

**Install PDM via pipx** (recommended — keeps PDM isolated):
```bash
pipx install pdm
```

**Or via uv:**
```bash
uv pip install pdm
```

**Verify:**
```bash
pdm --version
```

**Install LoudPy in editable mode:**
```bash
pdm install -e .
```

LoudPy is developed under Python 3.13. To install it with uv:
```bash
uv python install 3.13
```
