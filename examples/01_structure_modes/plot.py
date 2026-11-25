"""
plot.py — the modes of the loudspeaker in vacuo (run.py first).

What this produces
------------------
Results/Figures/
  - modes_all.pdf         : every mode shape, 16 per page (deformed structure, |u| normalised)
  - modes_selection.pdf   : a hand-picked subset
  - mode_chart.pdf        : damping ratio against frequency
  - mode_3_*.pdf          : one mode on the whole structure, on the surround, interface profile
Results/Files/
  - modes.csv             : the mode table (f, ζ) for any other software
  - mode_3.vtu            : that mode shape for ParaView
"""
from pathlib import Path

import loudpy as lp
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from names import stored  # noqa: E402  (results where run.py writes them, or stored earlier)
HERE = Path(__file__).resolve().parent               # this example's folder
EXAMPLES = HERE.parent                               # examples/

lp.plot.style.use()                       # the loudpy figure style for this script

here = HERE / "Results"
modes = stored(HERE, "modes")
print(modes)
print(modes.table().head(10))
modes.to_csv(here / "Files/modes.csv")

with lp.plot.pdf_pages(here / "Figures/modes_all.pdf") as pdf:
    for start in range(0, len(modes), 16):
        pdf.savefig(lp.plot.modes(modes, indices=range(start, min(start + 16, len(modes)))))
lp.plot.save(lp.plot.modes(modes, indices=[0, 1, 2, 4, 7, 16, 17, 20, 32, 34], ncols=5),
             here / "Figures/modes_selection.pdf")
lp.plot.save(lp.plot.mode_chart(modes).figure, here / "Figures/mode_chart.pdf")

# ── one mode: whole structure, surround, interface profile ────────────────────
k = 2                                                   # 0-based: the third mode
mode = modes.mode(k)
lp.plot.save(mode.plot(deformed=True).figure, here / f"Figures/mode_{k + 1}_full.pdf")
lp.plot.save(mode.restrict("surround").plot(deformed=True).figure, here / f"Figures/mode_{k + 1}_surround.pdf")
lp.plot.save(lp.plot.boundary(mode, modes.boundary("wet_front")).figure,
             here / f"Figures/mode_{k + 1}_interface.pdf")
mode.export(here / f"Files/mode_{k + 1}.vtu")
print(f"✓ figures → {here / 'Figures'}")
