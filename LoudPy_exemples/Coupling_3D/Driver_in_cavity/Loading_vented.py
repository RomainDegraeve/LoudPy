"""
Loading_vented.py — the vented box (one air domain inside, through the port and outside) against
the sealed box, both on a rigid ground (Running_vented.py merged and Running_exterior.py first).
1 N on the coil.

What this produces
------------------
Results/Figures/vented_response.pdf
  1. SPL at 1 m (driver axis, in front of the port, 30° and 90° horizontal), vented against sealed,
     and the dust-cap excursion (its minimum gives the port tuning)
  2. directivity maps of the vented box re the driver axis
  3. polar curves, vented (solid) against sealed (dashed)
Results/Files/vented_spl_1m.csv
"""
from pathlib import Path

import numpy as np

import loudpy as lp
from enclosure import X0, Y0, Z_F, PORT_X, PORT_Y, port_tuning, vented_volume

here = Path("LoudPy_exemples/Coupling_3D/Driver_in_cavity/Results")
DIST = 1.0
POLAR_F = [50, 70, 100, 300, 1000, 2000, 3000]

vented, sealed = lp.load(here / "Files/vented_merged.h5"), lp.load(here / "Files/exterior.h5")
src = np.array([X0, Y0, Z_F])
probes = lp.Points([src + [0, 0, DIST], [PORT_X, PORT_Y, Z_F + DIST],
                    src + DIST * np.array([np.sin(np.pi / 6), 0, np.cos(np.pi / 6)]), src + [DIST, 0, 0]],
                   labels=["driver axis", "in front of the port", "30° horizontal", "90° horizontal"])
spl_v, spl_s = vented.spl(at=probes), sealed.spl(at=probes)
spl_v.export(here / "Files/vented_spl_1m.csv")
x_v = vented.driver("main").excursion().rename(label="vented")
x_s = sealed.driver("main").excursion().rename(label="sealed")
low = vented.frequencies < 200
fb = vented.frequencies[low][np.argmin(np.abs(x_v.values[low]))]
f_helmholtz = port_tuning(vented_volume())

with lp.plot.pdf_pages(here / "Figures/vented_response.pdf") as pdf:
    fig, ax = lp.plot.figure("tall", 2, 1, sharex=True)
    for j, name in enumerate(probes.labels):
        c = lp.plot.style.COLORS[j]
        spl_v.isel(point=j).rename(label=f"vented, {name}").plot(ax=ax[0], color=c)
        spl_s.isel(point=j).rename(label=f"sealed, {name}").plot(ax=ax[0], color=c, ls="--", lw=0.8)
    lp.plot.line([x_v, x_s], ax=ax[1])
    for a in ax:
        a.axvline(fb, color="0.6", lw=0.7, ls=":")
    ax[0].legend(fontsize=6.5, ncol=2)
    ax[0].set_xlabel("")
    ax[1].set_title(f"excursion minimum {fb:.0f} Hz: the port tuning (Helmholtz estimate {f_helmholtz:.0f} Hz)",
                    fontsize=9)
    pdf.savefig(fig)

    arcs = {"horizontal": lp.Arc(DIST, np.arange(-180, 181, 2), center=src, axis="z", toward="x"),
            "vertical (90° up, −10° near the ground)": lp.Arc(DIST, np.arange(-10, 191, 2), center=src, axis="z",
                                                             toward="y")}
    dirs = {k: (vented.spl(at=a).normalize(angle=0), sealed.spl(at=a).normalize(angle=0)) for k, a in arcs.items()}
    fig, ax = lp.plot.figure("wide", 1, 2)
    for a, (name, (dv, _)) in zip(ax, dirs.items()):
        dv.plot(kind="map", ax=a, contours=[-6, -3], title=f"vented box, {name}")
    pdf.savefig(fig)

    fig = lp.plot.figure("wide", 1, 2, subplot_kw=dict(projection="polar"))[0]
    for a, (name, (dv, ds)), zero, cw in zip(fig.axes, dirs.items(), ("N", "E"), (True, False)):
        lp.plot.polar(dv.sel(f=POLAR_F), ax=a, zero=zero, clockwise=cw, legend=a is fig.axes[1],
                      title=f"{name.split(' (')[0]}: solid vented, dashed sealed")
        a.set_prop_cycle(None)                                            # same colours for the sealed box
        lp.plot.polar(ds.sel(f=POLAR_F), ax=a, zero=zero, clockwise=cw, legend=False, ls="--", lw=0.8,
                      title=f"{name.split(' (')[0]}: solid vented, dashed sealed")
    pdf.savefig(fig)
print(f"✓ {here / 'Figures/vented_response.pdf'}   (tuning from the excursion minimum: {fb:.0f} Hz)")
