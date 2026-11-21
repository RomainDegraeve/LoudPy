"""
plot.py — response of the driver loaded by a folded transmission line (labyrinth) on a
rigid ground (run.py first), compared with the sealed and vented boxes of
../Driver_in_cavity when their results are there. 1 N on the coil.

What this produces
------------------
Results/Figures/labyrinth_response.pdf
  1. SPL at 1 m (driver axis, in front of the mouth, 30° and 90° horizontal) and dust-cap
     excursion, with the sealed / vented boxes on the driver axis
  2. directivity maps re the driver axis, horizontal and vertical
  3. polar curves at a few frequencies
Results/Files/spl_1m.csv : the SPL curves as a table
"""
from pathlib import Path

import numpy as np

import loudpy as lp
from labyrinth import X0, Y0, Z_F, mouth, quarter_wave

HERE = Path(__file__).resolve().parent               # this example's folder
EXAMPLES = HERE.parent                               # LoudPy_exemples/

lp.plot.style.use()                       # the loudpy figure style for this script

here = HERE / "Results"
other = {"exterior.h5": EXAMPLES / "11_box_radiation/Results/Files/exterior.h5",
         "vented_merged.h5": EXAMPLES / "12_box_vented/Results/Files/vented_merged.h5"}
DIST = 1.0
POLAR_F = [50, 70, 100, 200, 500, 1000, 2000]

res = lp.load(here / "Files/labyrinth.h5")
f_qw = quarter_wave()
boxes = {k: lp.load(other[n]) for k, n in (("sealed box", "exterior.h5"), ("vented box", "vented_merged.h5"))
         if other[n].exists()}
src = np.array([X0, Y0, Z_F])
yb, yt = mouth()
probes = lp.Points([src + [0, 0, DIST], [X0, 0.5 * (yb + yt), Z_F + DIST],
                    src + DIST * np.array([np.sin(np.pi / 6), 0, np.cos(np.pi / 6)]), src + [DIST, 0, 0]],
                   labels=["driver axis", "in front of the mouth", "30° horizontal", "90° horizontal"])
spl = res.spl(at=probes)
spl.export(here / "Files/spl_1m.csv")
x = res.driver("main").excursion().rename(label="labyrinth")
low = res.frequencies < 200
f_min = res.frequencies[low][np.argmin(np.abs(x.values[low]))]

with lp.plot.pdf_pages(here / "Figures/labyrinth_response.pdf") as pdf:
    fig, ax = lp.plot.figure("tall", 2, 1, sharex=True)
    spl.plot(ax=ax[0])
    x.plot(ax=ax[1])
    for k, (name, b) in enumerate(boxes.items()):
        b.spl(at=src + [0, 0, DIST]).rename(label=f"{name}, driver axis").plot(ax=ax[0], color="0.45",
                                                                                ls=["--", ":"][k], lw=0.9)
        b.driver(b.drivers[0]).excursion().rename(label=name).plot(ax=ax[1], color="0.45", ls=["--", ":"][k], lw=0.9)
    for a in ax:
        a.axvline(f_qw, color="0.6", lw=0.7, ls=":")
        a.legend(fontsize=7, ncol=2)
    ax[0].set_xlabel("")
    ax[1].set_title(f"excursion minimum {f_min:.0f} Hz (line quarter-wave estimate {f_qw:.0f} Hz)", fontsize=9)
    pdf.savefig(fig)

    horizontal = res.spl(at=lp.Arc(DIST, np.arange(-180, 181, 2), center=src, axis="z", toward="x")).normalize(angle=0)
    vertical = res.spl(at=lp.Arc(DIST, np.arange(-10, 191, 2), center=src, axis="z", toward="y")).normalize(angle=0)
    fig, ax = lp.plot.figure("wide", 1, 2)
    horizontal.plot(kind="map", ax=ax[0], contours=[-6, -3], title="horizontal")
    vertical.plot(kind="map", ax=ax[1], contours=[-6, -3], title="vertical (90° up, −10° near the ground)")
    pdf.savefig(fig)

    fig = lp.plot.figure("wide", 1, 2, subplot_kw=dict(projection="polar"))[0]
    lp.plot.polar(horizontal.sel(f=POLAR_F), ax=fig.axes[0], title="horizontal, dB re driver axis")
    lp.plot.polar(vertical.sel(f=POLAR_F), ax=fig.axes[1], zero="E", clockwise=False,
                  title="vertical, dB re driver axis")
    pdf.savefig(fig)
print(f"✓ {here / 'Figures/labyrinth_response.pdf'}   (excursion minimum {f_min:.0f} Hz, "
      f"quarter-wave estimate {f_qw:.0f} Hz)")
