# 10_box_cavity — the driver in a 3D box (cavity modes)

The driver's azimuthal-order modes (m = 0…3, from `02_azimuthal_modes`) coupled to the rigid-wall acoustic modes of
a non-axisymmetric 8.4 L box (slanted back, internal brace, off-centre driver), by modal superposition with a static
correction. The driver front is in vacuo. The box is defined once in `../shared/enclosure.py`.

```bash
python3 LoudPy_exemples/02_azimuthal_modes/run.py   # driver modes (once)
python3 LoudPy_exemples/10_box_cavity/run.py        # ~5 min → Results/Files/coupled.h5, coupled_modes.h5
python3 LoudPy_exemples/10_box_cavity/plot.py       # response, coupled modes, cavity modes, 3D deformation
python3 LoudPy_exemples/10_box_cavity/plot_3d.py    # 3D views every 10 Hz (slow, pyvista)
```

**Shows**: `lp.Driver.from_modes`, `lp.Profile` selections, `lp.Scene(…).cavity()`, `lp.FrequencyResponse` and
`lp.Modes` on a scene, `res.driver(…)` (excursion, kinetic energy per azimuthal order, the same driver in vacuo),
the SPL at a point inside the box, the coupled modes and their azimuthal content.
**Result**: pistonic mode 115.1 Hz in vacuo → 120.6 Hz in the box (air-spring estimate 121.6 Hz); the rocking orders
m ≥ 1 reach 13 % of the kinetic energy at the first box mode (466 Hz), from the box's asymmetry alone.
