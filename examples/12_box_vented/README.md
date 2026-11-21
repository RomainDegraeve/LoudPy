# 12_box_vented — a vented (bass-reflex) box as one air domain

Walls 15 mm thick, a port of radius 20 mm and length 62 mm through the front face (its tube protrudes into the
box). ONE air domain: the inside of the box, the port and the outside (FE + half-space DtN); the driver is wetted on
both faces and the walls around the port are refined.

```bash
python3 examples/12_box_vented/run.py              # ~40 min → Results/Files/vented_merged.h5
python3 examples/12_box_vented/run.py 50,500       # quick test (two frequencies)
python3 examples/12_box_vented/plot.py             # vented vs sealed (11_box_radiation): SPL, excursion
python3 examples/12_box_vented/plot_3d.py          # the box cut through the driver and port axes
```

**Shows**: `scene.merged()`, `lp.Cabinet(openings=…, opening_size=…)`, the port tuning (Helmholtz estimate against the
simulated excursion minimum).
