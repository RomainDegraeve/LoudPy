"""
Loading_script.py — figures and exports of the two sweeps (Running_plate.py and Running_speaker.py first).

What this produces
------------------
Results/Figures/
  - plate_f1.pdf            f1 of the plate against the radius, one curve per thickness, with Kirchhoff's f ∝ h / a²
  - speaker_excursion.pdf   voice-coil excursion per N over frequency, one panel per spider stiffness, one curve per surround
  - speaker_fs.pdf          the first resonance fs over the 3 × 3 grid
Results/Files/
  - plate_f.nc              f of the first two modes over (a, h, mode): netCDF (xarray, MATLAB ncread, …)
  - speaker_excursion.nc    coil excursion over (E_surround, E_spider, f): netCDF
  - speaker_fs.csv          fs per run (a table)
"""
from pathlib import Path

import numpy as np

import loudpy as lp

lp.plot.style.use()
here = Path("LoudPy_exemples/Sweep_study/Results")

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

# ── speaker: material sweep ────────────────────────────────────────────────────
speaker = lp.load(here / "Files/speaker_sweep.h5")


def coil_excursion(r):
    """|u_z| of the voice coil over f: the mean over the nodes of its boundary (where the force acts)."""
    tags = r.boundary("interface_forced")
    u = r.displacement(at=lp.Points(r.mesh("solid").coords_of(tags)), component="z")
    return lp.Data(np.abs(u.values.mean(axis=1)), {"f": r.frequencies}, "u_z")


x = speaker.evaluate(coil_excursion)                                    # lp.Data over E_surround × E_spider × f
x.export(here / "Files/speaker_excursion.nc")
Es, Ep = x.coords["E_surround"], x.coords["E_spider"]
fig, axes = lp.plot.figure("wide", 1, len(Ep), sharey=True)
for j, (ax, ep) in enumerate(zip(axes, Ep)):
    for es in Es:
        ax.loglog(x.coords["f"], x.sel(E_surround=es, E_spider=ep).values * 1e3, label=f"{es / 1e6:g} MPa")
    ax.set(title=f"spider E = {ep / 1e6:g} MPa", xlabel="f [Hz]")
axes[0].set_ylabel("|coil excursion| [mm/N]")
axes[-1].legend(title="surround E")
lp.plot.save(fig, here / "Figures/speaker_excursion.pdf")

def first_resonance(r):
    """fs: the first peak of the coil excursion. (Not the largest: with a stiff surround and a soft spider the
    fundamental comes close to the spider's own mode near 150 Hz, the two share their motion, and the higher one
    can have the larger peak.)"""
    x = coil_excursion(r).values
    peaks = np.flatnonzero((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:])) + 1
    return r.frequencies[peaks[0]]


fs = speaker.evaluate(first_resonance, name="fs")
table = speaker.table()
table["fs [Hz]"] = [first_resonance(r) for r in speaker]
table.to_csv(here / "Files/speaker_fs.csv")
print(table)
fig, ax = lp.plot.figure("square")
im = ax.pcolormesh(Ep / 1e6, Es / 1e6, fs.values, shading="nearest")
ax.figure.colorbar(im, ax=ax, label="fs [Hz]")
ax.set(xlabel="spider E [MPa]", ylabel="surround E [MPa]", xscale="log", yscale="log")
lp.plot.save(fig, here / "Figures/speaker_fs.pdf")
print(f"✓ figures → {here / 'Figures'}, exports → {here / 'Files'}")
