"""
plot_3d.py — 3D scenes of the labyrinth cabinet on a rigid ground (run.py
first): one air domain inside the folded line, at the mouth and outside, and the deformed driver,
1 N on the coil (res.plot: cut through the driver axis, dashed outline of the removed half).

Each scene is cut by the vertical plane through the driver axis (x = X0), which shows the whole
folded line in section (dashed: the cabinet outline). SPL [dB re 20 µPa] on the ground, the walls
and the cut (inside the line, through the mouth and in front of the cabinet), one colour scale
(RANGE dB below the maximum beyond 8 cm of the driver), iso-lines every 3 dB; driver deformation
10 % of its diameter.

What this produces (in Results/Figures/)
-----------------------------------------
  - labyrinth_views_3d.pdf : one page per frequency of FREQS, six views
"""
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

import loudpy as lp
from labyrinth import X0, Z_F, W_IN as W, H_IN as H, WALL_T, Z_BACK, quarter_wave
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from names import stored  # noqa: E402  (results where run.py writes them, or stored earlier)
HERE = Path(__file__).resolve().parent               # this example's folder
EXAMPLES = HERE.parent                               # examples/

lp.plot.style.use()                       # the loudpy figure style for this script

here = HERE / "Results"
FREQS = [40, 70, 100, 150, 300, 1000]         # [Hz] (nearest computed frequency)
RANGE = 60.0

res = stored(HERE, "labyrinth")
normal = np.array([-1.0, 0.0, 0.0])           # cut x = X0, cameras on the −x side
ey, ez = np.array([0, 1.0, 0]), np.array([0, 0, 1.0])
F0 = np.array([X0, 0.20, -0.12])
cameras = {
    "three-quarter front, cut side": lp.plot.Camera(F0 + 0.9 * normal + 0.55 * ey + 0.95 * ez, F0, ey, view_angle=24),
    "section of the line": lp.plot.Camera([X0 - 1.5, 0.20, -0.12], [X0, 0.20, -0.12], ey, parallel=0.27),
    "section close-up: mouth": lp.plot.Camera([X0 - 1.0, 0.06, 0.0], [X0, 0.06, 0.0], ey, parallel=0.10),
    "front face (whole cabinet)": lp.plot.Camera([X0, 0.21, 1.6], [X0, 0.21, 0.0], ey, parallel=0.30, cut=False),
    "ground and section from above": lp.plot.Camera([X0, 2.0, -0.12], [X0, 0.0, -0.12], ez, parallel=0.52),
    "three-quarter rear (whole box)": lp.plot.Camera(F0 + [0.95, 0.65, -1.05], F0, ey, view_angle=24, cut=False),
}
t = WALL_T
ground = pv.Disc(center=(0.10, -5e-4, -0.06), inner=0.0, outer=0.45, normal=(0, 1, 0), c_res=90)


def keep(points):
    """Cut plane: inside the cabinet (and the mouth) everywhere, outside only in front of it."""
    x, y, z = points.T
    inside = (x >= -t) & (x <= W + t) & (y <= H + t) & (z >= Z_BACK - t)
    return (z > Z_F) | inside


with lp.plot.pdf_pages(here / "Figures/labyrinth_views_3d.pdf") as pdf:
    for f in FREQS:
        if np.min(np.abs(res.frequencies - f)) > 15.0:
            continue
        fig = res.plot(f, cameras, keep=keep, range_db=RANGE, ground=ground,      # cut through the driver axis
                       title=f"Labyrinth on a rigid ground, 1 N (quarter-wave ≈ {quarter_wave():.0f} Hz)")
        pdf.savefig(fig)
        print(f"  {f} Hz rendered")
print(f"✓ {here / 'Figures/labyrinth_views_3d.pdf'}")
