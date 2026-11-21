"""
plot.py — the geometry sweep (run.py first): f1 of the clamped plate for every radius and thickness, with
Kirchhoff's thin-plate law f1 = 10.2158 / a² · √(D / ρh) / 2π.

What this produces
------------------
  Results/Figures/plate_f1.pdf   f1 against the radius, one colour per thickness (FE dots, Kirchhoff lines)
  Results/Files/plate_f.nc       f of the first two modes over (a, h, mode): netCDF (xarray, MATLAB ncread, …)
"""
from pathlib import Path

import numpy as np

import loudpy as lp

HERE = Path(__file__).resolve().parent               # this example's folder
EXAMPLES = HERE.parent                               # examples/

lp.plot.style.use()
here = HERE / "Results"

# ── plate: geometry sweep ──────────────────────────────────────────────────────
plate = lp.load(here / "Files/plate_sweep.h5")
f = plate.evaluate(lambda r: r.frequencies[:2], name="f")              # lp.Data over a × h × mode
f.export(here / "Files/plate_f.nc")
steel = dict(E=2.1e11, nu=0.3, rho=7850.0)
fig, ax = lp.plot.figure("wide")
a_fine = np.linspace(f.coords["a"].min(), f.coords["a"].max(), 100)
for h in f.coords["h"]:
    D = steel["E"] * h ** 3 / (12 * (1 - steel["nu"] ** 2))
    kirchhoff = 10.2158 / a_fine ** 2 * np.sqrt(D / (steel["rho"] * h)) / (2 * np.pi)
    line, = ax.plot(f.coords["a"] * 1e3, f.sel(h=h).values[:, 0], "o", label=f"h = {h * 1e3:g} mm")
    ax.plot(a_fine * 1e3, kirchhoff, "-", color=line.get_color(), lw=0.8)
ax.set(xlabel="radius $a$ [mm]", ylabel="first frequency $f_1$ [Hz]", yscale="log")
ax.legend(title="FE (dots), Kirchhoff (lines)")
lp.plot.save(fig, here / "Figures/plate_f1.pdf")
print(f"✓ figure → {here / 'Figures'}, export → {here / 'Files'}")
