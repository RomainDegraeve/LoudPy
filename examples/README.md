# loudpy examples

One folder per example, numbered in reading order. Every folder has the same layout:

```
NN_name/
  README.md        what it shows, how to run it, the key results
  run.py           builds the model, solves, saves its results the loudpy way: results/<example>/<study>_<stamp>.h5
                   (study.save(HERE, name=…), stamped with what made them); meshes and caches in Results/Files/
  plot.py          reads the newest result (shared/names.stored(); else the file an earlier run wrote in
                   Results/Files/), writes Results/Figures/ (PDF) and the key curves (CSV, netCDF) in Results/Files/
  plot_3d.py       3D views (pyvista), where there are some
```

Run the scripts from anywhere (paths are relative to each script), e.g.

```bash
python3 examples/01_structure_modes/run.py
python3 examples/01_structure_modes/plot.py
```

A `run.py` is computed once (meshes are built on the first run and reused); `plot.py` can be re-run as often as
needed on the stored results.

| # | Example | What it shows | run.py | re-run 2026-11-25 (run + plots) |
|---|---|---|---|---|
| 01 | `01_structure_modes` | modes of the driver in vacuo | < 1 min | yes, 10 s + 62 s |
| 02 | `02_azimuthal_modes` | non-axisymmetric modes (orders m = 0…4), 3D views — **needed by 10–15** (driver modes) | ~1 min | yes, 53 s + 13 min |
| 03 | `03_coupled_modes_radiation` | structure + air modes with an exact radiation condition | ~7 min | yes, 6.2 min + 47 s |
| 04 | `04_response_structure` | frequency response of the structure, 1 N on the coil | ~1 min | yes, 57 s + 4 s |
| 05 | `05_response_coupled` | structure + air response 20 Hz – 20 kHz, air remeshed per wavelength, directivity | ~2 min | yes, 2.0 min + 13 s |
| 06 | `06_response_voltage` | the same driven by a voltage: impedance, coil current | ~2 min | yes, 2.9 min + 15 s |
| 07 | `07_transient` | nonlinear time response to a multitone force; output control (`lp.Output`) | ~40 min | no (~40 min): built and checked |
| 08 | `08_sweep_geometry` | parametric sweep of the **geometry** (plate radius × thickness, `.geo` parameters) | ~5 s | yes, 4 s + 1 s |
| 09 | `09_sweep_materials` | parametric sweep of **materials** (surround × spider stiffness of the driver) | ~1 min | yes, 47 s + 3 s |
| 10 | `10_box_cavity` | the driver in a 3D box: cavity modes, coupled modes | ~5 min | no (long 3D): built, scene checked, plots compiled; its stored results still plot |
| 11 | `11_box_radiation` | the box on a rigid ground radiating (exact half-space radiation) | ~14 min | no (long 3D): built, scene checked, plots compiled; its stored results still plot |
| 12 | `12_box_vented` | a vented (bass-reflex) box as one air domain | ~40 min | no (long 3D): built, scene checked, plots compiled; its stored results still plot |
| 13 | `13_box_elastic_walls` | the box with elastic MDF walls (shells) | ~8 min | no (long 3D): built, scene checked, plots compiled; its stored results still plot |
| 14 | `14_labyrinth` | a folded transmission line on a rigid ground | ~12 min | no (long 3D): built, scene checked, plots compiled; its stored results still plot |
| 15 | `15_column_multi_driver` | three drivers in a column with a folded line (multi-driver scene) | ~30 min | no (long 3D): built, scene checked, plots compiled; its stored results still plot |
| 18 | `18_walls_radiating` | elastic walls (a STEP of panels and labyrinth boards, thicknesses measured) in one air domain with an opening, radiating | not run yet (~1 h) | no (long 3D): built, scene checked, plots compiled |

Other folders:

- `validation/` — checks against analytical solutions (clamped plate, rectangular cavity, radiating hemisphere);
- `shared/` — the box of examples 10–13 (`enclosure.py`) and its meshes;
- `Geometries/` — the driver (`HPNEW-Sketch.geo` + `.brep`, the STEP sketch, the FreeCAD source `HPNEW.FCStd`) and
  the plates (`clamped_plate.geo`, `clamped_plate_param.geo` with its dimensions as parameters);
- `legacy/` — former scripts kept for reference (solver benchmarks, a mesh generator, the low-level hybrid model of the
  vented box, a STEP preparation script).

Times are on an Apple M2 (8 GB). The work order of 2026-11-25 re-ran the examples under ~10 min (the 3D ones,
10–15 and 18, and the transient 07 are the long ones: rewritten, built and checked, not re-run).
