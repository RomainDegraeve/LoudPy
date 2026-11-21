# 09_sweep_materials — a parametric sweep of materials

The driver of `04_response_structure` with the Young's moduli of its surround and spider as parameters, each at
0.5, 1 and 2 × the library value: 9 runs, 20 Hz – 2 kHz, 1 N on the coil. The mesh does not change, so every run
reuses it.

```bash
python3 LoudPy_exemples/09_sweep_materials/run.py    # ~1 min → Results/Files/speaker_sweep.h5
python3 LoudPy_exemples/09_sweep_materials/plot.py   # coil excursion per combination, fs map; netCDF + CSV
```

**Shows**: `lp.P` in a material property, `lp.Sweep` on a frequency response, evaluation on boundary nodes,
`sweep.evaluate` → data over E_surround × E_spider × f, `sweep.table()`.
**Result**: fs from 81 Hz (soft) to 162 Hz (stiff). With a stiff surround and a soft spider the fundamental comes
close to the spider's own mode near 150 Hz (the two share their motion).
