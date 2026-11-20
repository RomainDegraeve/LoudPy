"""
Plot_3D_column.py — 3D scenes of the three-driver column with a folded transmission line, on a
rigid ground (Running_column.py first): one air domain inside the line, at the mouth and outside,
and the three deformed drivers (lp.plot.scene_view).

The vertical plane through the driver axes (x = X0) cuts the column: it shows the three drivers and
the whole folded line in section, the front radiation and the mouth at the back. SPL [dB re 20 µPa]
on the ground, the walls and the cut, one colour scale (RANGE dB below the maximum beyond 8 cm of
the drivers), iso-lines every 3 dB; drivers at one instant and one scale (largest displacement =
10 % of the diameter).

What this produces (in Results/Figures/)
-----------------------------------------
  - column_views_3d.pdf : one page per frequency of FREQS, four views
"""
from pathlib import Path

import numpy as np
import pyvista as pv

import loudpy as lp
from loudpy.plot import view3d
from column import X0, Y0S, Z_F, Z_BACK, W_IN as W, H_IN as H, WALL_T, quarter_wave

here = Path("LoudPy_exemples/Coupling_3D/Column_labyrinth/Results")
FREQS = [40, 70, 120, 250, 500, 1000]         # [Hz] (nearest computed frequency)
RANGE = 50.0

res = lp.load(here / "Files/column.h5")
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
yz = [(0.0, Z_F), (0.0, Z_BACK - t), (H + t, Z_BACK - t), (H + t, Z_F)]
segments = [((x, y0, z0), (x, y1, z1)) for x in (-t, W + t) for (y0, z0), (y1, z1) in zip(yz, yz[1:] + yz[:1])]
outline = pv.merge([view3d.dashed(a, b) for a, b in segments + [((-t, y, z), (W + t, y, z)) for y, z in yz]])
ground = pv.Disc(center=(X0, -5e-4, -0.08), inner=0.0, outer=0.8, normal=(0, 1, 0), c_res=90)

with lp.plot.pdf_pages(here / "Figures/column_views_3d.pdf") as pdf:
    for f in FREQS:
        if np.min(np.abs(res.frequencies - f)) > 15.0:
            continue
        fig = lp.plot.scene_view(res, f, cameras, cut=(normal, (X0, Y0S[1], 0.0)), range_db=RANGE, outline=outline,
                                 ground=ground, size=(900, 900),
                                 title=f"Column on a rigid ground (quarter-wave ≈ {quarter_wave():.0f} Hz)")
        pdf.savefig(fig)
        print(f"  {f} Hz rendered")
print(f"✓ {here / 'Figures/column_views_3d.pdf'}")
