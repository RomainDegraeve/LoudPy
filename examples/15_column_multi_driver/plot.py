"""
plot.py — response of the three-driver column with a folded transmission line, on a
rigid ground (run.py first). Pressures outside the hemisphere come from the exact DtN
expansion (ground included); all drivers driven with the DRIVE amplitudes of run.py.

What this produces
------------------
Results/Figures/column_response.pdf
  1. SPL at 1 m (front axis of the middle driver, 30° up and down, behind at the mouth) and the
     dust-cap excursion of each driver
  2. directivity maps re the front axis, vertical (the column effect) and horizontal
  3. vertical and horizontal polar curves at a few frequencies
  4. per driver: share of the kinetic energy in the orders m ≥ 1 (rocking, ovalisation: driven only
     by the non-uniform air load, the coil drives m = 0) and excursion phase re the middle driver
Results/Files/
  - spl_1m.csv, directivity_vertical.csv : the curves and the vertical map as tables
"""
import sys
from pathlib import Path

import numpy as np

import loudpy as lp
from column import X0, Y0S, Z_F, Z_BACK, WALL_T, MOUTH_H, NAMES, quarter_wave
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from names import stored  # noqa: E402  (results where run.py writes them, or stored earlier)
HERE = Path(__file__).resolve().parent               # this example's folder
EXAMPLES = HERE.parent                               # examples/

lp.plot.style.use()                       # the loudpy figure style for this script

here = HERE / "Results"
DIST = 1.0                                      # distance of the probes from the middle driver [m]
POLAR_F = [50, 100, 200, 400, 700, 1000]        # [Hz] (nearest computed frequency)

res = stored(HERE, "column")
print(res)
f_qw = quarter_wave()
src = np.array([X0, Y0S[1], Z_F])                                       # middle driver, front face
up30 = np.array([0, np.sin(np.pi / 6), np.cos(np.pi / 6)])
probes = lp.Points([src + [0, 0, DIST], src + DIST * up30, src + DIST * up30 * [1, -1, 1],
                    [X0, 0.5 * MOUTH_H, Z_BACK - WALL_T - DIST]],
                   labels=["front axis", "30° up", "30° down", "behind, at the mouth"])
spl = res.spl(at=probes)
spl.export(here / "Files/spl_1m.csv")
drivers = [res.driver(n) for n in NAMES]
excursion = lp.Data(np.column_stack([d.excursion().values for d in drivers]),
                    {"f": res.frequencies, "point": np.array([f"{n} driver" for n in NAMES])}, "u_z")
low = res.frequencies < 2 * f_qw                                        # line tuning: excursion minimum
f_min = res.frequencies[low][np.argmin(np.abs(excursion.values[low, 1]))]

with lp.plot.pdf_pages(here / "Figures/column_response.pdf") as pdf:
    fig, ax = lp.plot.figure("tall", 2, 1, sharex=True)
    spl.plot(ax=ax[0])
    excursion.plot(ax=ax[1])
    for a in ax:
        a.axvline(f_qw, color="0.6", lw=0.7, ls=":")
    ax[0].set_xlabel("")
    ax[1].set_title(f"middle-driver excursion minimum {f_min:.0f} Hz (quarter-wave estimate {f_qw:.0f} Hz)",
                    fontsize=9)
    pdf.savefig(fig)

    # directivity: vertical plane x = X0 (0° front axis, 90° up, 180° behind) and horizontal plane
    vertical = res.spl(at=lp.Arc(DIST, np.arange(-20, 201, 2), center=src, axis="z", toward="y")).normalize(angle=0)
    horizontal = res.spl(at=lp.Arc(DIST, np.arange(-180, 181, 2), center=src, axis="z", toward="x")).normalize(angle=0)
    vertical.export(here / "Files/directivity_vertical.csv")
    fig, ax = lp.plot.figure("wide", 1, 2)
    vertical.plot(kind="map", ax=ax[0], contours=[-6, -3], title="vertical (90° up, 180° behind)")
    horizontal.plot(kind="map", ax=ax[1], contours=[-6, -3], title="horizontal")
    pdf.savefig(fig)

    fig = lp.plot.figure("wide", 1, 2, subplot_kw=dict(projection="polar"))[0]
    lp.plot.polar(vertical.sel(f=POLAR_F), ax=fig.axes[0], zero="E", clockwise=False,
                  title="vertical, dB re front axis")
    lp.plot.polar(horizontal.sel(f=POLAR_F), ax=fig.axes[1], title="horizontal, dB re front axis")
    pdf.savefig(fig)

    # per driver: share of the kinetic energy in m ≥ 1, excursion phase re the middle driver
    share = np.column_stack([100 * (1 - d.kinetic_energy(by_order=True).sel(order=0).values
                                    / d.kinetic_energy().values) for d in drivers])
    fig, ax = lp.plot.figure("tall", 2, 1, sharex=True)
    lp.Data(share, excursion.coords, lp.results.Quantity("share", "Kinetic energy in orders m ≥ 1", "", "%")
            ).plot(ax=ax[0])
    (excursion / excursion.values[:, 1:2]).phase(deg=True).rename(
        lp.results.Quantity("phase", "Excursion phase re middle driver", r"\Delta\varphi", "°")).plot(ax=ax[1])
    for a in ax:
        a.axvline(f_qw, color="0.6", lw=0.7, ls=":")
    ax[0].set_xlabel("")
    pdf.savefig(fig)
print(f"✓ {here / 'Figures/column_response.pdf'}   (middle-driver excursion minimum {f_min:.0f} Hz, "
      f"quarter-wave estimate {f_qw:.0f} Hz)")
