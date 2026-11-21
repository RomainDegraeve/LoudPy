# 13_box_elastic_walls — the box with elastic MDF walls (shells)

The box of `10_box_cavity` with 12 mm MDF walls: MITC3 shells on the walls of the cavity mesh, panel edges clamped,
coupled to the cavity air through their in-vacuo modes.

```bash
python3 LoudPy_exemples/13_box_elastic_walls/run.py     # ~8 min → Results/Files/coupled_walls.h5, walls_modes.csv
python3 LoudPy_exemples/13_box_elastic_walls/plot.py    # rigid vs elastic walls (uses 10_box_cavity's result)
```

**Shows**: `scene.walls(material, thickness)`, `lp.Modes(scene, physics="walls")`, `res.wall_volume_velocity()`,
`res.driver_volume_velocity()`.
**Result**: 50 wall modes to 5.9 kHz (first at 1.1 kHz); the walls' volume velocity reaches −10 dB of the driver's
at 1.38 kHz; the coil excursion changes by up to 4 %.
