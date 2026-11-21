# 07_transient — nonlinear time response to a multitone force

A multitone force (2 Hz grid, 20 Hz – 1 kHz, zero-leakage block) after a 0.5 s ramp, geometrically nonlinear
structure, 48 kHz, 1 s (48 000 steps).

```bash
python3 LoudPy_exemples/07_transient/run.py    # ~40 min → Results/Files/time.h5 (76 MB)
python3 LoudPy_exemples/07_transient/plot.py   # tip u, v, a over time and their spectra; field snapshots; animation
```

**Shows**: `lp.Transient` with `lp.Force(signal, fs)`, **output control** `lp.Output(fields=["u"], every=40,
probes={"tip": (0, 0)})` — the tip at full rate, the field every 40 steps (8.7 GB → 76 MB), `res.probe_signal`,
spectra (`.spectrum()`), `lp.plot.animate`.
