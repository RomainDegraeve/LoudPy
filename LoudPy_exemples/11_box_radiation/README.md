# 11_box_radiation — the box on a rigid ground, radiating

The box of `10_box_cavity` standing on a perfectly reflecting ground: the front radiates into 3D P2 FE air inside a
hemisphere with the **exact** half-space radiation condition (DtN); the rear is loaded by the cavity modes.

```bash
python3 LoudPy_exemples/11_box_radiation/run.py            # ~14 min → Results/Files/exterior.h5
python3 LoudPy_exemples/11_box_radiation/plot.py           # SPL at 1 m, polars, response change
python3 LoudPy_exemples/11_box_radiation/plot_3d.py        # cuts every 10 Hz (slow)
python3 LoudPy_exemples/11_box_radiation/plot_3d_views.py  # full 3D scenes from 6 angles
```

**Shows**: `scene.exterior()`, band meshes per wavelength, the exact far field (`res.far_field`, `res.spl(at=lp.Arc…)`),
directivity, `lp.plot.scene_view`.
**Check**: at 50 Hz the far field is omnidirectional and p·r is within 1.4 % of a monopole on a rigid ground.
