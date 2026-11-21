# 01_structure_modes — modes of the driver in vacuo

The HPNEW driver as an axisymmetric structure (cone, surround, spider, former, coil, glue, dust cap), clamped at the
rim and the spider: its modes from 20 Hz to 40 kHz.

```bash
python3 examples/01_structure_modes/run.py     # < 1 min → Results/Files/modes.h5
python3 examples/01_structure_modes/plot.py    # → Results/Figures/ (mode chart, mode shapes), modes.csv
```

**Shows**: `lp.Model` on a `.geo`, `lp.Solid` with library materials, `lp.Clamped`, `lp.Coil`, `lp.Modes`, the mode
table (`modes.table()`), mode shapes on the deformed structure, `lp.plot.modes`, `lp.plot.mode_chart`.
**Result**: pistonic mode 115 Hz (ζ 0.029).
