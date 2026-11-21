"""
plot_3d_views.py — full 3D scenes of the driver in its enclosure on a rigid ground, from several
angles, at a few frequencies (run.py first). 1 N on the coil (res.plot / res.view, the default cameras).

The box is cut open by the vertical plane through the driver axis (the default cut: its half x < X0
removed; dashed: its outline). SPL [dB re 20 µPa] outside (ground, outer walls, the cut in front of the box) and inside
(inner walls, the cut: cavity modes), one colour scale (RANGE dB below the larger of the inside
maximum and the outside maximum beyond 8 cm of the driver), iso-lines every 3 dB; the deformed
driver, whole revolution, at the instant of its largest displacement (10 % of its diameter).

What this produces
------------------
Results/Figures/
  - views_3d.pdf       : one page per frequency of FREQS, six views
Results/Files/
  - scene_<f>Hz.html   : the cut-away scene, interactive in a browser (one self-contained file)
  - scene_<f>Hz.glb    : the same as a 3D model with its colours (Blender, PowerPoint, Windows 3D viewer, the web)

    python3 plot_3d_views.py --interactive      also opens each scene in a window to turn it
"""
import sys
from pathlib import Path

import loudpy as lp
from loudpy.physics.modal import CavityModes
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))   # the shared box
from enclosure import Z_F  # noqa: E402

HERE = Path(__file__).resolve().parent               # this example's folder
lp.plot.style.use()                                  # the loudpy figure style for this script

here = HERE / "Results"
FREQS = [100, 470, 1000, 1680, 2400]          # [Hz] (nearest computed frequency)
RANGE = 50.0

res = lp.load(here / "Files/exterior.h5")
cav = CavityModes.load(HERE.parent / "shared/meshes/cavity_modes.h5")
front = lambda x: x[:, 2] > Z_F                      # the cut plane: inside the box, outside only in front of it
with lp.plot.pdf_pages(here / "Figures/views_3d.pdf") as pdf:
    for f in FREQS:
        fig = res.plot(f, keep=front, cavity=cav, range_db=RANGE, ncols=3,
                       title="Driver in its enclosure on a rigid ground, 1 N")
        pdf.savefig(fig)
        v = res.view(f, keep=front, cavity=cav, range_db=RANGE)        # one view: interactive / 3D-model files
        title = f"Driver in its enclosure on a rigid ground, 1 N, {f} Hz"
        v.save(here / f"Files/scene_{f}Hz.html", title=title)
        v.save(here / f"Files/scene_{f}Hz.glb")
        if "--interactive" in sys.argv:
            v.interactive(title=title)
        print(f"  {f} Hz rendered")
print(f"✓ {here / 'Figures/views_3d.pdf'}")
