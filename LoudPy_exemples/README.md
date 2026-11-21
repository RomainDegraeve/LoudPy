# loudpy examples

Ready-to-run scripts on the HPNEW driver (a 2D axisymmetric model) and on drivers in 3D enclosures. Run every script
**from the repository root**, e.g.

```bash
python3 LoudPy_exemples/Eigen_study/Meca/Running_script.py
python3 LoudPy_exemples/Eigen_study/Meca/Loading_script.py
```

Each example separates the computation from the figures:

- `Running_*.py` builds the model, solves, writes `Results/Files/` (the result file, format 3, and the meshes, built
  on the first run and reused afterwards);
- `Loading_*.py` / `Plot_*.py` read the results (`lp.load`) and write `Results/Figures/` (PDF) and the key curves
  in `Results/Files/` (CSV, netCDF).

| Folder | What it shows | Time |
|---|---|---|
| `Eigen_study/Meca` | structural modes of the driver in vacuo, 20 Hz – 40 kHz | < 1 min |
| `Eigen_study/Fourier` | non-axisymmetric (azimuthal order m = 0…4) modes, 3D views; Kirchhoff plate validation. **Needed by the 3D examples** (driver modes) | ~1 min |
| `Eigen_study/FSI_DtN` | coupled structure + air modes with the exact radiation condition | ~7 min |
| `Frequency_study/Meca` | structural frequency response, 1 N on the voice coil | ~1 min |
| `Frequency_study/FSI` | coupled structure + air response 20 Hz – 20 kHz, the air remeshed per wavelength | ~2 min |
| `Frequency_study/FSI_elec` | the same driven by a voltage: impedance, coil current | ~2 min |
| `Time_study/Multitone` | nonlinear transient response to a multitone force; tip signal at full rate (`lp.Output` probe) | ~40 min |
| `Sweep_study` | parametric sweeps: a **geometry** sweep (plate radius × thickness from a `.geo` with `DefineConstant`) and a **material** sweep (surround × spider stiffness of the driver); results as labelled data, netCDF exports | ~1 min |
| `Coupling_3D/Driver_in_cavity` | the driver in a 3D box: cavity modes, radiation on a rigid ground, vented box, elastic MDF walls; validations of the 3D cavity and of the exact half-space radiation | 5–40 min each |
| `Coupling_3D/Labyrinth` | folded transmission line on a rigid ground | ~12 min |
| `Coupling_3D/Column_labyrinth` | three drivers in a column with a folded line (multi-driver scene) | ~30 min |

`Geometries/` holds the driver (`HPNEW-Sketch.geo` + `.brep`, the STEP sketch, the FreeCAD source `HPNEW.FCStd`) and the
plates. `legacy/` keeps scripts that are no longer examples (solver benchmarks, a mesh generator for every example, the
low-level hybrid model of the vented box, a STEP preparation script, an unused geometry variant).
