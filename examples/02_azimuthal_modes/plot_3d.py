"""
plot_3d.py — 3D views of the azimuthal-order modes (run.py first).

Every mode of the orders ORDERS is drawn on the whole structure: the boundary of the meridian mesh
revolved around the axis and deformed by the 3D field of order m,

    u_r = U_r cos mθ,   u_z = U_z cos mθ,   u_θ = U_θ sin mθ,

coloured by |u| (Field.revolve). Two views per mode: revolved over 270° (the cut-away shows the
section, spider and former) and a full revolution seen from the front (the circumferential
pattern on the membrane and the surround). Largest displacement: DEFORMATION × the diameter.

What this produces
------------------
Results/Figures/
  - modes_3d_m<m>.pdf  : PER_PAGE modes per page, one row per mode (3D view | front view)
Results/Files/
  - mode_m<m>_1.html   : the first mode of each order, interactive in a browser (Field.view().save)

    python3 plot_3d.py --interactive      also opens those modes in a window to turn them
"""
import sys
from pathlib import Path

import numpy as np

import loudpy as lp
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from names import OLD_NAMES  # noqa: E402  (results stored before the 0.3 renaming)

HERE = Path(__file__).resolve().parent               # this example's folder
EXAMPLES = HERE.parent                               # examples/

lp.plot.style.use()                       # the loudpy figure style for this script

here = HERE / "Results"
ORDERS = [0, 1, 2, 3]
DEFORMATION = 0.10
PER_PAGE = 3
SIZE = (620, 500)                                          # rendered image [px]

modes = lp.load(here / "Files/eigen_fourier.h5", rename=OLD_NAMES)
for m in [m for m in ORDERS if m in modes.orders]:
    om = modes.order(m)
    with lp.plot.pdf_pages(here / f"Figures/modes_3d_m{m}.pdf") as pdf:
        for start in range(0, len(om), PER_PAGE):
            page = range(start, min(start + PER_PAGE, len(om)))
            fig, axs = lp.plot.figure((7.2, 2.7 * PER_PAGE), PER_PAGE, 2, squeeze=False)
            for a in axs.ravel():
                a.set_axis_off()
            for row, k in zip(axs, page):
                mode = om.mode(k)
                for ax, sweep, view in ((row[0], 1.5 * np.pi, ("xz", 35, 25)), (row[1], 2 * np.pi, ("xy", 0, 0))):
                    v = lp.plot.View3D(SIZE)
                    v.add(mode.revolve(sweep=sweep, deformation=DEFORMATION), scalars="|u|", clim=(0, 1),
                          colorbar=False, smooth_shading=True)
                    v.view(view[0], azimuth=view[1], elevation=view[2], zoom=1.3).show(ax)
                row[0].set_title(rf"$m$ = {m}, mode {k + 1}: {mode.coordinate:.0f} Hz, $\zeta$ = "
                                 f"{np.real(mode.extra['zeta']):.3f}", fontsize=9)
                row[1].set_title("front view", fontsize=9)
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize
            fig.colorbar(ScalarMappable(Normalize(0, 1), lp.plot.style.CMAPS["displacement"]), ax=axs, shrink=0.5,
                         label=r"$|u|/|u|_{max}$")
            pdf.savefig(fig)
    first = om.mode(0)
    v = first.view(sweep=1.5 * np.pi, deformation=DEFORMATION)
    title = f"m = {m}, mode 1: {first.coordinate:.0f} Hz"
    v.save(here / f"Files/mode_m{m}_1.html", title=title)
    if "--interactive" in sys.argv:
        v.interactive(title=title)
    print(f"✓ m = {m}: {len(om)} modes → {here / f'Figures/modes_3d_m{m}.pdf'}")
