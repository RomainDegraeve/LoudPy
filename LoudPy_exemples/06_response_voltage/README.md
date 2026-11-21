# 06_response_voltage — the same, driven by a voltage

The coupled model of `05_response_coupled` driven by 2.83 V through the voice-coil circuit (Re, Bl, semi-inductance
Le · (jω)ⁿ): electrical impedance, coil current, SPL.

```bash
python3 LoudPy_exemples/06_response_voltage/run.py    # ~2 min → Results/Files/sweep.h5
python3 LoudPy_exemples/06_response_voltage/plot.py   # impedance, SPL, directivity, fields
```

**Shows**: `lp.Voltage` (Re, Bl, Le, semi-inductance exponent), the electrical impedance `Z`, the coil current, SPL and
directivity as in 05; the Thiele–Small parameters (`res.thiele_small(on=…)`, written to `thiele_small.csv`); the
deformed structure in 3D as interactive HTML files (`field.view3d().save("u.html")`).
**Result**: impedance peak 75.8 Ω at 112 Hz; fitted fs = 111.5 Hz, Qms = 17.1, Qes = 2.02, Qts = 1.81,
Sd = 78 cm², Vas = 1.0 L.
