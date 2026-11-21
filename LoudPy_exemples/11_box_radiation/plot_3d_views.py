"""
plot_3d_views.py — full 3D scenes of the driver in its enclosure on a rigid ground, from several
angles, at a few frequencies (run.py first). 1 N on the coil (lp.plot.scene_view).

The box is cut open by the vertical plane through the driver axis (its half x < X0 removed; dashed:
its outline). SPL [dB re 20 µPa] outside (ground, outer walls, the cut in front of the box) and inside
(inner walls, the cut: cavity modes), one colour scale (RANGE dB below the larger of the inside
maximum and the outside maximum beyond 8 cm of the driver), iso-lines every 3 dB; the deformed
driver, whole revolution, at the instant of its largest displacement (10 % of its diameter).

What this produces (in Results/Figures/)
-----------------------------------------
  - views_3d.pdf : one page per frequency of FREQS, six views
"""
import sys
from pathlib import Path

import pyvista as pv

import loudpy as lp
from loudpy.physics.modal import CavityModes
from loudpy.plot import view3d
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))   # the shared box
from enclosure import X0, Y0, Z_F, H, back_z  # noqa: E402

HERE = Path(__file__).resolve().parent               # this example's folder
EXAMPLES = HERE.parent                               # LoudPy_exemples/

lp.plot.style.use()                       # the loudpy figure style for this script

here = HERE / "Results"
FREQS = [100, 470, 1000, 1680, 2400]          # [Hz] (nearest computed frequency)
RANGE = 50.0
focus, up = (0.10, 0.13, -0.06), (0, 1, 0)
cameras = {
    "front left (cut side)": lp.plot.Camera((-0.80, 0.70, 0.95), focus, up, view_angle=25),
    "front right (outer walls)": lp.plot.Camera((1.05, 0.65, 0.95), focus, up, view_angle=25),
    "side, facing the cut": lp.plot.Camera((-1.30, 0.35, -0.06), focus, up, view_angle=25),
    "from above": lp.plot.Camera((-0.35, 1.50, 0.45), focus, up, view_angle=25),
    "rear right (back and side walls)": lp.plot.Camera((1.00, 0.60, -1.05), focus, up, view_angle=25),
    "close-up on the driver (cut side)": lp.plot.Camera((-0.30, 0.36, 0.30), (X0, Y0, 0.0), up, view_angle=30),
}
yz = [(0.0, Z_F), (0.0, back_z(0.0)), (H, back_z(H)), (H, Z_F)]                 # removed half of the box
segments = [((0.0, y0, z0), (0.0, y1, z1)) for (y0, z0), (y1, z1) in zip(yz, yz[1:] + yz[:1])]
outline = pv.merge([view3d.dashed(a, b) for a, b in segments + [((0.0, y, z), (X0, y, z)) for y, z in yz]])

res = lp.load(here / "Files/exterior.h5")
cav = CavityModes.load(EXAMPLES / "shared/meshes/cavity_modes.h5")
with lp.plot.pdf_pages(here / "Figures/views_3d.pdf") as pdf:
    for f in FREQS:
        fig = lp.plot.scene_view(res, f, cameras, cut=((-1, 0, 0), (X0, 0, 0)), keep=lambda x: x[:, 2] > Z_F,
                                 cavity=cav, range_db=RANGE, outline=outline, ncols=3,
                                 title="Driver in its enclosure on a rigid ground, 1 N")
        pdf.savefig(fig)
        print(f"  {f} Hz rendered")
print(f"✓ {here / 'Figures/views_3d.pdf'}")
