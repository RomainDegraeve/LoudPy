"""
plot_3d.py — pressure inside and outside the enclosure and the deformed driver, every
computed frequency (run.py first). 1 N on the coil.

One row per frequency, at one instant (the phase of the largest displacement):
  - a vertical cut through the driver axis and a horizontal cut at the driver height (from above):
    the pressure outside (± its maximum beyond NEAR_FIELD from the driver: the near field
    saturates) and inside the enclosure (± its own maximum), with the half of the deformed driver
    on the side of the cut;
  - the deformed driver from the front (largest displacement DEFORMATION × diameter).

What this produces (in Results/Figures/exterior_3d/)
-----------------------------------------------------
  - exterior_3d_<f0>-<f1>Hz.pdf : one file per BAND Hz, ROWS frequencies per page
"""
import sys
from pathlib import Path

import numpy as np

import loudpy as lp
from loudpy.physics.modal import CavityModes
from loudpy.plot import view3d
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))   # the shared box
from enclosure import X0, Y0, Z_F  # noqa: E402

HERE = Path(__file__).resolve().parent               # this example's folder
EXAMPLES = HERE.parent                               # examples/

lp.plot.style.use()                       # the loudpy figure style for this script

here = HERE / "Results"
BAND, ROWS = 500.0, 2
DEFORMATION, NEAR_FIELD = 0.10, 0.08
CUTS = {"vertical": ((-1, 0, 0), (X0, 0, 0), dict(position=(-1.25, 0.45, 0.25), focus=(0.125, 0.22, -0.03),
                                                  up=(0, 1, 0), zoom=1.15), (-np.pi / 2, np.pi / 2)),
        "horizontal": ((0, 1, 0), (0, Y0, 0), dict(position=(0.125, 2.2, -0.03), focus=(0.125, 0.0, -0.03),
                                                   up=(0, 0, 1), zoom=1.15), (np.pi, 2 * np.pi))}
BARS = [(lp.plot.style.DIVERGING, (-1, 1), r"Pressure $p/|p|_{max}$ (outside, inside)"),
        (lp.plot.style.CMAPS["displacement"], (0, 1), r"Driver $|u|/|u|_{max}$")]

res = lp.load(here / "Files/exterior.h5")
hp = res.driver("main")
cav = CavityModes.load(EXAMPLES / "shared/meshes/cavity_modes.h5")
E = hp.kinetic_energy(by_order=True)


def row(axs, i):
    p_out = res.field("p", index=i)
    f = p_out.coordinate
    p_in = res.cavity_field(f, cavity=cav)
    _, phase = hp.surface(f, n_theta=13, deformation=DEFORMATION)
    far = np.linalg.norm(p_out.mesh.coords - [X0, Y0, Z_F], axis=1) > NEAR_FIELD
    out_max, in_max = float(np.abs(p_out.values[far]).max()), float(np.abs(p_in.values).max())
    g_out = view3d.tet_grid(p_out.mesh, np.clip(p_out.at_phase(phase).values / out_max, -1, 1), "p")
    g_in = view3d.tet_grid(p_in.mesh, p_in.at_phase(phase).values / in_max, "p")
    for ax, (normal, origin, camera, theta) in zip(axs[:2], CUTS.values()):
        half, _ = hp.surface(f, theta=theta, n_theta=37, deformation=DEFORMATION, phase=phase)
        v = lp.plot.View3D((640, 560))
        for g in (g_out, g_in):
            v.add(view3d.clip(g, normal, origin), scalars="p", cmap=BARS[0][0], clim=(-1, 1), colorbar=False)
        v.add(half, scalars="|u|", clim=(0, 1), smooth_shading=True, colorbar=False).camera(**camera).show(ax)
    full, _ = hp.surface(f, local=True, deformation=DEFORMATION, phase=phase)
    (lp.plot.View3D((480, 480)).add(full, scalars="|u|", clim=(0, 1), smooth_shading=True, colorbar=False)
     .camera((0, 0, 1), (0, 0, 0), (0, 1, 0), zoom=1.2).show(axs[2]))
    share = ", ".join(f"$m$ = {m}: {100 * e / E.values[i].sum():.0f} %" for m, e in zip(E.coords["order"], E.values[i]))
    axs[0].set_title(f"{f:.0f} Hz: vertical cut\nmax outside {lp.spl(out_max):.0f} dB, inside {lp.spl(in_max):.0f} dB",
                     fontsize=7)
    axs[1].set_title(f"horizontal cut, from above\n(driver height {Y0:.2f} m)", fontsize=7)
    axs[2].set_title(f"driver, front view\n{share}", fontsize=6)


freqs = res.frequencies
for b0 in np.arange(freqs[0] - freqs[0] % BAND, freqs[-1], BAND):
    sel = [i for i, f in enumerate(freqs) if b0 <= f < b0 + BAND or (f == freqs[-1] and b0 + BAND >= f)]
    if not sel:
        continue
    with lp.plot.pdf_pages(here / f"Figures/exterior_3d/exterior_3d_{freqs[sel[0]]:.0f}-{freqs[sel[-1]]:.0f}Hz.pdf") as pdf:
        for s0 in range(0, len(sel), ROWS):
            fig, axs = lp.plot.figure((7.2, 2.9 * ROWS), ROWS, 3, squeeze=False,
                                      gridspec_kw=dict(width_ratios=[1.15, 1.15, 0.8]))
            for a in axs.ravel():
                a.set_axis_off()
            for k, i in enumerate(sel[s0:s0 + ROWS]):
                row(axs[k], i)
            view3d.colorbars_for(fig, axs, BARS, shrink=0.45)
            pdf.savefig(fig)
    print(f"  ✓ {freqs[sel[0]]:.0f}–{freqs[sel[-1]]:.0f} Hz ({len(sel)} frequencies)")
