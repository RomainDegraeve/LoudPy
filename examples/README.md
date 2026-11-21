# loudpy examples

One folder per example, numbered in reading order. Every folder has the same layout:

```
NN_name/
  README.md        what it shows, how to run it, the key results
  run.py           builds the model, solves, writes Results/Files/  (the result file, the meshes)
  plot.py          reads the results, writes Results/Figures/ (PDF) and the key curves (CSV, netCDF)
  plot_3d.py       3D views (pyvista), where there are some
```

Run the scripts from anywhere (paths are relative to each script), e.g.

```bash
python3 examples/01_structure_modes/run.py
python3 examples/01_structure_modes/plot.py
```

A `run.py` is computed once (meshes are built on the first run and reused); `plot.py` can be re-run as often as
needed on the stored results.

| # | Example | What it shows | run.py |
|---|---|---|---|
| 01 | `01_structure_modes` | modes of the driver in vacuo | < 1 min |
| 02 | `02_azimuthal_modes` | non-axisymmetric modes (orders m = 0…4), 3D views — **needed by 10–15** (driver modes) | ~1 min |
| 03 | `03_coupled_modes_radiation` | structure + air modes with an exact radiation condition | ~7 min |
| 04 | `04_response_structure` | frequency response of the structure, 1 N on the coil | ~1 min |
| 05 | `05_response_coupled` | structure + air response 20 Hz – 20 kHz, air remeshed per wavelength, directivity | ~2 min |
| 06 | `06_response_voltage` | the same driven by a voltage: impedance, coil current | ~2 min |
| 07 | `07_transient` | nonlinear time response to a multitone force; output control (`lp.Output`) | ~40 min |
| 08 | `08_sweep_geometry` | parametric sweep of the **geometry** (plate radius × thickness, `.geo` parameters) | ~5 s |
| 09 | `09_sweep_materials` | parametric sweep of **materials** (surround × spider stiffness of the driver) | ~1 min |
| 10 | `10_box_cavity` | the driver in a 3D box: cavity modes, coupled modes | ~5 min |
| 11 | `11_box_radiation` | the box on a rigid ground radiating (exact half-space radiation) | ~14 min |
| 12 | `12_box_vented` | a vented (bass-reflex) box as one air domain | ~40 min |
| 13 | `13_box_elastic_walls` | the box with elastic MDF walls (shells) | ~8 min |
| 14 | `14_labyrinth` | a folded transmission line on a rigid ground | ~12 min |
| 15 | `15_column_multi_driver` | three drivers in a column with a folded line (multi-driver scene) | ~30 min |

Other folders:

- `validation/` — checks against analytical solutions (clamped plate, rectangular cavity, radiating hemisphere);
- `shared/` — the box of examples 10–13 (`enclosure.py`) and its meshes;
- `Geometries/` — the driver (`HPNEW-Sketch.geo` + `.brep`, the STEP sketch, the FreeCAD source `HPNEW.FCStd`) and
  the plates (`clamped_plate.geo`, `clamped_plate_param.geo` with its dimensions as parameters);
- `legacy/` — former scripts kept for reference (solver benchmarks, a mesh generator, the low-level hybrid model of the
  vented box, a STEP preparation script).

Times are on an Apple M2 (8 GB).
