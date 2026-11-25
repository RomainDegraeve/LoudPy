"""
plot_3d.py — 3D scenes of the vented box on a rigid ground (run.py merged first):
one air domain inside the box, through the port and outside, the deformed driver, 1 N on the coil
(res.plot; the dashed outline of the removed part is automatic).

Each scene is cut by the vertical plane through the driver axis and the port axis (dashed: the box
outline). SPL [dB re 20 µPa] on the ground, the walls and the cut (inside the box, through the port
and in front of the box), one colour scale (RANGE dB below the maximum beyond 8 cm of the driver),
iso-lines every 3 dB; driver deformation 10 % of its diameter.

What this produces (in Results/Figures/)
-----------------------------------------
  - vented_views_3d.pdf : one page per frequency of FREQS, six views
"""
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

import loudpy as lp
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))   # the shared box
from enclosure import X0, Y0, Z_F, W, H, WALL_T, back_z, PORT_X, PORT_Y, port_tuning, vented_volume  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from names import stored  # noqa: E402  (results where run.py writes them, or stored earlier)
HERE = Path(__file__).resolve().parent               # this example's folder
EXAMPLES = HERE.parent                               # examples/

lp.plot.style.use()                       # the loudpy figure style for this script

here = HERE / "Results"
FREQS = [50, 70, 100, 300, 1000, 1900]        # [Hz] (nearest computed frequency)
RANGE = 60.0

res = stored(HERE, "vented_merged")
# cut plane through the driver axis and the port axis; cameras on the removed side (+normal)
d = np.array([X0 - PORT_X, Y0 - PORT_Y, 0.0])
d /= np.linalg.norm(d)
normal = np.array([-d[1], d[0], 0.0]) * (1 if d[0] >= 0 else -1)
origin = np.array([X0, Y0, 0.0])
on_plane = lambda x: x - np.dot(x - origin, normal) * normal
F0 = np.array([0.10, 0.13, -0.06])
F_sec = on_plane(F0) + [0.0, 0.03, 0.0]
F_det = on_plane(np.array([0.5 * (X0 + PORT_X), 0.5 * (Y0 + PORT_Y), 0.0]))
ey, ez = np.array([0, 1.0, 0]), np.array([0, 0, 1.0])
cameras = {
    "three-quarter front, cut side": lp.plot.Camera(F0 + 0.9 * normal + 0.55 * ey + 0.95 * ez, F0, ey, view_angle=24),
    "section, square to the cut": lp.plot.Camera(F_sec + 1.5 * normal, F_sec, ey, parallel=0.27),
    "section close-up: port and driver": lp.plot.Camera(F_det + 1.0 * normal, F_det, ey, parallel=0.12),
    "front face (whole box)": lp.plot.Camera((0.10, 0.18, 1.6), (0.10, 0.18, 0.0), ey, parallel=0.30, cut=False),
    "ground and section from above": lp.plot.Camera((0.10, 2.0, -0.06), (0.10, 0.0, -0.06), ez, parallel=0.42),
    "three-quarter rear (whole box)": lp.plot.Camera(F0 + [0.95, 0.65, -1.05], F0, ey, view_angle=24, cut=False),
}
t = WALL_T
ground = pv.Disc(center=(0.10, -5e-4, -0.06), inner=0.0, outer=0.45, normal=(0, 1, 0), c_res=90)


def keep(points):
    """Cut plane: inside the box (and the port) everywhere, outside only in front of the box."""
    x, y, z = points.T
    return (z > Z_F) | ((x >= -t) & (x <= W + t) & (y <= H + t) & (z >= back_z(y) - t))


with lp.plot.pdf_pages(here / "Figures/vented_views_3d.pdf") as pdf:
    for f in FREQS:
        if np.min(np.abs(res.frequencies - f)) > 15.0:
            continue
        fig = res.plot(f, cameras, cut=(normal, origin), keep=keep, range_db=RANGE, ground=ground,
                       title=f"Vented box on a rigid ground, 1 N (port tuning ≈ {port_tuning(vented_volume()):.0f} Hz)")
        pdf.savefig(fig)
        print(f"  {f} Hz rendered")
print(f"✓ {here / 'Figures/vented_views_3d.pdf'}")
