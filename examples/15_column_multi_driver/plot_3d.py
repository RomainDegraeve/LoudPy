"""
plot_3d.py — 3D scenes of the three-driver column with a folded transmission line, on a
rigid ground (run.py first): one air domain inside the line, at the mouth and outside,
and the three deformed drivers (res.plot / res.view).

The vertical plane through the driver axes (x = X0) cuts the column: it shows the three drivers and
the whole folded line in section, the front radiation and the mouth at the back. SPL [dB re 20 µPa]
on the ground, the walls and the cut, one colour scale (RANGE dB below the maximum beyond 8 cm of
the drivers), iso-lines every 3 dB; drivers at one instant and one scale (largest displacement =
10 % of the diameter).

What this produces (in Results/Figures/)
-----------------------------------------
  - column_views_3d.pdf : one page per frequency of FREQS, four views
"""
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

import loudpy as lp
from column import X0, Z_F, Z_BACK, W_IN as W, H_IN as H, WALL_T, quarter_wave
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from names import stored  # noqa: E402  (results where run.py writes them, or stored earlier)
HERE = Path(__file__).resolve().parent               # this example's folder
EXAMPLES = HERE.parent                               # examples/

lp.plot.style.use()                       # the loudpy figure style for this script

here = HERE / "Results"
FREQS = [40, 70, 120, 250, 500, 1000]         # [Hz] (nearest computed frequency)
RANGE = 50.0

res = stored(HERE, "column")
normal = np.array([-1.0, 0.0, 0.0])           # cut x = X0, cameras on the −x side
ey, ez = np.array([0, 1.0, 0]), np.array([0, 0, 1.0])
mid = np.array([X0, 0.36, 0.5 * (Z_F + Z_BACK)])
cameras = {
    "section through the driver axes": lp.plot.Camera(mid + 1.8 * normal, mid, ey, parallel=0.62),
    "three-quarter front, cut side": lp.plot.Camera(mid + 1.3 * normal + 0.5 * ey + 1.2 * ez, mid, ey),
    "front (whole column)": lp.plot.Camera(mid + 2.2 * ez, mid, ey, parallel=0.45, cut=False),
    "three-quarter rear: the mouth": lp.plot.Camera(mid + [-1.5, 0.45, -1.9], mid - [0, 0.05, 0], ey, cut=False),
}
t = WALL_T
ground = pv.Disc(center=(X0, -5e-4, -0.08), inner=0.0, outer=0.8, normal=(0, 1, 0), c_res=90)

with lp.plot.pdf_pages(here / "Figures/column_views_3d.pdf") as pdf:
    for f in FREQS:
        if np.min(np.abs(res.frequencies - f)) > 15.0:
            continue
        fig = res.plot(f, cameras, range_db=RANGE, ground=ground, size=(900, 900),   # cut through the driver axes
                       title=f"Column on a rigid ground (quarter-wave ≈ {quarter_wave():.0f} Hz)")
        pdf.savefig(fig)

        # one view as a View3D → an interactive HTML file (and a .glb 3D model)
        v = res.view(f, cameras["three-quarter front, cut side"], range_db=RANGE, ground=ground, size=(900, 900))
        v.save(here / f"Files/column_{f}Hz.html", title=f"Column on a rigid ground, {f} Hz")
        v.save(here / f"Files/column_{f}Hz.glb")          # optional: Blender / PowerPoint / Windows 3D viewer
        print(f"  {f} Hz rendered")
print(f"✓ {here / 'Figures/column_views_3d.pdf'}")
