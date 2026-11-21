# 04_response_structure — frequency response of the structure

1 N on the voice coil, 10 Hz – 20 kHz, structure only (no air).

```bash
python3 LoudPy_exemples/04_response_structure/run.py    # ~1 min → Results/Files/sweep.h5
python3 LoudPy_exemples/04_response_structure/plot.py   # tip u, v, a; deformed fields; impulse response
```

**Shows**: `lp.FrequencyResponse` with `lp.Force`, evaluations at points (`res.displacement(at=…)`), velocity and
acceleration, fields on the deformed structure, regions of a field (`.restrict`), CSV export.
**Result**: tip displacement 2.0 mm/N at the 115 Hz resonance.
