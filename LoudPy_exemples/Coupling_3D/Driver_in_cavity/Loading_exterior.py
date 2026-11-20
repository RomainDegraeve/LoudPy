"""
Loading_exterior.py — the driver in its enclosure, standing on a rigid ground, radiating into the
exterior air (Running_exterior.py first). 1 N on the coil; pressures outside the hemisphere from the
exact DtN expansion (ground included).

What this produces
------------------
Results/Figures/
  - spl_1m.pdf            : SPL at 1 m from the driver for a few horizontal angles
  - polars.pdf            : directivity maps re on axis (horizontal, vertical) and polar curves
  - response_change.pdf   : effect of the exterior air on the driver (kinetic energy per order,
                            dust-cap velocity), against the same enclosure with the front in vacuo
Results/Files/
  - spl_1m.csv, directivity_horizontal.csv
"""
from pathlib import Path

import numpy as np

import loudpy as lp
from enclosure import X0, Y0, Z_F, sealed_scene

here = Path("LoudPy_exemples/Coupling_3D/Driver_in_cavity/Results")
DIST = 1.0                                   # measurement distance from the driver centre [m]
H_ANGLES = [0, 30, 60, 90, 180]              # horizontal angles of spl_1m.pdf [°]
POLAR_F = [100, 300, 1000, 2000, 3000]       # frequencies of the polar curves [Hz]

res = lp.load(here / "Files/exterior.h5")
hp = res.driver("main")
src = np.array([X0, Y0, Z_F])
# the same driver and enclosure with the front in vacuo (cavity modes only: fast)
encl = lp.FrequencyResponse(sealed_scene(), drive=lp.Force(1.0)).solve(res.frequencies, log=lambda *a, **k: None)
hp_encl = encl.driver("main")

spl = res.spl(at=lp.Arc(DIST, H_ANGLES, center=src, axis="z", toward="x"))
spl.export(here / "Files/spl_1m.csv")
lp.plot.save(spl.plot().figure, here / "Figures/spl_1m.pdf")

horizontal = res.spl(at=lp.Arc(DIST, np.arange(-180, 181, 2), center=src, axis="z", toward="x")).normalize(angle=0)
vertical = res.spl(at=lp.Arc(DIST, np.arange(-10, 191, 2), center=src, axis="z", toward="y")).normalize(angle=0)
horizontal.export(here / "Files/directivity_horizontal.csv")
with lp.plot.pdf_pages(here / "Figures/polars.pdf") as pdf:
    fig, ax = lp.plot.figure("wide", 1, 2)
    horizontal.plot(kind="map", ax=ax[0], contours=[-6, -3], title="horizontal")
    vertical.plot(kind="map", ax=ax[1], contours=[-6, -3], title="vertical (90° up, −10° near the ground)")
    pdf.savefig(fig)
    fig = lp.plot.figure("wide", 1, 2, subplot_kw=dict(projection="polar"))[0]
    lp.plot.polar(horizontal.sel(f=POLAR_F), ax=fig.axes[0], title="horizontal, dB re on axis")
    lp.plot.polar(vertical.sel(f=POLAR_F), ax=fig.axes[1], zero="E", clockwise=False, title="vertical, dB re on axis")
    pdf.savefig(fig)

fig, ax = lp.plot.figure("wide", 1, 2)
E_ext, E_encl = hp.kinetic_energy(by_order=True), hp_encl.kinetic_energy(by_order=True)
for k, m in enumerate(E_ext.coords["order"]):
    c = lp.plot.style.COLORS[k]
    E_ext.sel(order=m).rename(label=f"$m$ = {m}, with exterior").plot(ax=ax[0], color=c)
    E_encl.sel(order=m).rename(label=f"$m$ = {m}, front in vacuo").plot(ax=ax[0], color=c, ls="--", lw=0.8)
hp.excursion().velocity().rename(label="with exterior").plot(ax=ax[1])
hp_encl.excursion().velocity().rename(label="front in vacuo").plot(ax=ax[1], ls="--")
ax[0].legend(fontsize=6.5, ncol=2)
lp.plot.save(fig, here / "Figures/response_change.pdf")
print(f"✓ figures → {here / 'Figures'}")
