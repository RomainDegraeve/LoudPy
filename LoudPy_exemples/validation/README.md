# validation — checks against analytical solutions

```bash
python3 LoudPy_exemples/validation/clamped_plate.py     # clamped circular plate vs Kirchhoff, m = 0…4 (< 1 min)
python3 LoudPy_exemples/validation/rect_cavity.py       # 3D rigid cavity vs the exact modes (< 1 min)
python3 LoudPy_exemples/validation/dtn_hemisphere.py    # exterior FE + exact DtN vs a pulsating / oscillating hemisphere
```

| check | result |
|---|---|
| clamped plate (a = 0.1 m, h = 2 mm), all (m, n) modes | within 0.6 % of thin-plate theory (slightly lower: shear, rotary inertia) |
| rectangular cavity, 43 modes ≤ 3 kHz | max error 0.021 % |
| hemisphere on a rigid ground, 100 Hz – 3 kHz | 0.6 % at 1 kHz, ≤ 1.4 % at 3 kHz |

The test suite checks many more cases; their numbers are in `docs/guide/verification_report.rst`.
