"""
plot.py — the material sweep (run.py first): the voice-coil excursion and the first resonance fs for every
surround × spider stiffness.

What this produces
------------------
  Results/Figures/speaker_excursion.pdf   coil excursion per N over f, one panel per spider, one curve per surround
  Results/Figures/speaker_fs.pdf          fs over the 3 × 3 grid
  Results/Files/speaker_excursion.nc      coil excursion over (E_surround, E_spider, f): netCDF
  Results/Files/speaker_fs.csv            fs per run (a table)
"""
from pathlib import Path

import numpy as np

import loudpy as lp
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from names import OLD_NAMES  # noqa: E402  (results stored before the 0.3 renaming)

HERE = Path(__file__).resolve().parent               # this example's folder
EXAMPLES = HERE.parent                               # examples/

lp.plot.style.use()
here = HERE / "Results"

# ── speaker: material sweep ────────────────────────────────────────────────────
speaker = lp.load(here / "Files/speaker_sweep.h5", rename=OLD_NAMES)


def coil_excursion(r):
    """|u_z| of the voice coil over f: the mean over the nodes of its boundary (where the force acts)."""
    tags = r.boundary("coil_drive")
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
