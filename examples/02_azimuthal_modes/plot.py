"""
plot.py — the modes of every azimuthal order m (run.py first).

What this produces
------------------
Results/Figures/
  - mode_chart.pdf   : frequency of every mode against its azimuthal order m (colour: ζ)
  - modes_m<m>.pdf   : the modes of each order on the meridian section (deformed by (U_r, U_z),
                       colour |U| including U_θ), 16 per page
Results/Files/
  - modes.csv        : the mode table (order, f, ζ)
(3D views of the modes: plot_3d.py)
"""
from pathlib import Path

import loudpy as lp
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from names import OLD_NAMES  # noqa: E402  (results stored before the 0.3 renaming)

HERE = Path(__file__).resolve().parent               # this example's folder
EXAMPLES = HERE.parent                               # examples/

lp.plot.style.use()                       # the loudpy figure style for this script

here = HERE / "Results"
modes = lp.load(here / "Files/eigen_fourier.h5", rename=OLD_NAMES)
print(modes, {m: len(modes.order(m)) for m in modes.orders})
modes.to_csv(here / "Files/modes.csv")

lp.plot.save(lp.plot.mode_chart(modes, y="order").figure, here / "Figures/mode_chart.pdf")
for m in modes.orders:
    om = modes.order(m)
    with lp.plot.pdf_pages(here / f"Figures/modes_m{m}.pdf") as pdf:
        for start in range(0, len(om), 16):
            fig = lp.plot.modes(om, indices=range(start, min(start + 16, len(om))))
            fig.suptitle(f"Azimuthal order $m$ = {m}: modes {start + 1}–{min(start + 16, len(om))} of {len(om)}")
            pdf.savefig(fig)
print(f"✓ figures → {here / 'Figures'}")
