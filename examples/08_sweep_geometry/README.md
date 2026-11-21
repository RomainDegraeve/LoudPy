# 08_sweep_geometry — a parametric sweep of the geometry

A clamped steel plate whose radius `a` and thickness `h` are `DefineConstant` entries of
`Geometries/Plate/clamped_plate_param.geo`; the model's parameters of the same name replace them, so each run
rebuilds and remeshes the geometry. 4 radii × 3 thicknesses = 12 runs.

```bash
python3 examples/08_sweep_geometry/run.py    # ~5 s → Results/Files/plate_sweep.h5
python3 examples/08_sweep_geometry/plot.py   # f1 against a and h with Kirchhoff; netCDF export
```

**Shows**: `lp.Parameters` and `lp.P` (the mesh size follows h), geometry parameters in a `.geo`,
`lp.Sweep(study, {"a": …, "h": …})` (all combinations), `sweep.table()`, `sweep.evaluate(…)` → data over a × h,
`.export("x.nc")`.
**Result**: every f1 within 0.5 % of Kirchhoff (slightly lower for thick plates: shear).
