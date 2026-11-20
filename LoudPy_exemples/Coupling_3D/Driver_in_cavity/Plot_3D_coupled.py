"""
Plot_3D_coupled.py — coupled acoustic + mechanical 3D views every DF Hz (Running_script.py first).

At each frequency (1 N on the coil) the forced response of the driver in the enclosure (front in
vacuo) is recomputed on a regular grid (fast: the cavity mesh and modes are cached), and one row
shows, at one instant (the phase of the largest displacement):
  - left : the enclosure cut through the driver axis, coloured by the pressure (± its maximum),
           with the deformed driver (the half on the side of the cut);
  - right: the deformed driver from the front, the whole revolution: the circumferential pattern
           of the non-axisymmetric orders on the cone and the surround.
Largest displacement: DEFORMATION × the driver diameter.

What this produces (in Results/Figures/coupled_3d/)
----------------------------------------------------
  - coupled_3d_<f0>-<f1>Hz.pdf : one file per BAND Hz, ROWS frequencies per page
"""
from pathlib import Path

import numpy as np

import loudpy as lp
from loudpy.physics.modal import CavityModes
from loudpy.plot import view3d
from enclosure import X0, sealed_scene

lp.plot.style.use()                       # the loudpy figure style for this script

here = Path("LoudPy_exemples/Coupling_3D/Driver_in_cavity/Results")
F_START, F_END, DF = 20.0, 3000.0, 10.0       # frequencies drawn [Hz]
BAND, ROWS = 500.0, 2                         # frequencies per PDF file [Hz], rows per page
DEFORMATION = 0.10
CAMERA = dict(position=(-0.45, 0.05, 0.25), focus=(0.125, 0.15, -0.04), up=(0, 0, 1), zoom=1.25)
BARS = [(lp.plot.style.DIVERGING, (-1, 1), r"Pressure $p/|p|_{max}$"),
        (lp.plot.style.CMAPS["displacement"], (0, 1), r"Driver $|u|/|u|_{max}$")]

freqs = np.arange(F_START, F_END + 0.5 * DF, DF)
res = lp.FrequencyResponse(sealed_scene(), drive=lp.Force(1.0)).solve(freqs, log=lambda *a, **k: None)
cav = CavityModes.load(here / "Files/cavity_modes.h5")
hp = res.driver("main")
E = hp.kinetic_energy(by_order=True)
print(f"{len(freqs)} frequencies, {len(cav.kappa2)} enclosure modes, {len(hp.modes)} driver modes")


def row(axl, axr, f):
    half, phase = hp.surface(f, theta=(-np.pi / 2, np.pi / 2), n_theta=37, deformation=DEFORMATION)
    full, _ = hp.surface(f, local=True, deformation=DEFORMATION, phase=phase)
    p = res.cavity_field(f, cavity=cav)
    p_max = float(np.abs(p.values).max())
    cut = view3d.clip(view3d.tet_grid(p.mesh, p.at_phase(phase).values / p_max, "p"), (-1, 0, 0), (X0, 0, 0))
    (lp.plot.View3D((820, 620)).add(cut, scalars="p", cmap=BARS[0][0], clim=(-1, 1), colorbar=False)
     .add(half, scalars="|u|", clim=(0, 1), smooth_shading=True, colorbar=False).camera(**CAMERA).show(axl))
    (lp.plot.View3D((520, 520)).add(full, scalars="|u|", clim=(0, 1), smooth_shading=True, colorbar=False)
     .camera((0, 0, 1), (0, 0, 0), (0, 1, 0), zoom=1.25).show(axr))
    i = int(np.argmin(np.abs(res.frequencies - f)))
    share = ", ".join(f"$m$ = {m}: {100 * e / E.values[i].sum():.0f} %" for m, e in zip(E.coords["order"], E.values[i]))
    axl.set_title(f"{f:.0f} Hz: pressure in the enclosure (max {lp.spl(p_max):.0f} dB) and driver", fontsize=8)
    axr.set_title(f"front view\nkinetic energy {share}", fontsize=6)


for b0 in np.arange(F_START - F_START % BAND, F_END, BAND):
    sel = [f for f in freqs if b0 <= f < b0 + BAND or (f == F_END and b0 + BAND >= F_END)]
    if not sel:
        continue
    with lp.plot.pdf_pages(here / f"Figures/coupled_3d/coupled_3d_{sel[0]:.0f}-{sel[-1]:.0f}Hz.pdf") as pdf:
        for s0 in range(0, len(sel), ROWS):
            fig, axs = lp.plot.figure((7.2, 3.0 * ROWS), ROWS, 2, squeeze=False, gridspec_kw=dict(width_ratios=[1.45, 1]))
            for a in axs.ravel():
                a.set_axis_off()
            for k, f in enumerate(sel[s0:s0 + ROWS]):
                row(axs[k, 0], axs[k, 1], f)
            view3d.colorbars_for(fig, axs, BARS, shrink=0.45)
            pdf.savefig(fig)
    print(f"  ✓ {sel[0]:.0f}–{sel[-1]:.0f} Hz")
