"""
plot.py — the driver coupled to the enclosure air, front in vacuo (run.py first).
1 N on the coil.

What this produces
------------------
Results/Figures/
  - response.pdf        : 1. kinetic energy of the driver per azimuthal order, with and without the
                             enclosure; share of the non-axisymmetric orders; dust-cap velocity;
                             SPL in the enclosure at a probe (grey lines: rigid-wall enclosure modes)
                          2. rocking signature: axial displacement of the cone at four angles
  - coupled_modes.pdf   : coupled driver–enclosure modes, coloured by their dominant driver order
  - cavity_modes.pdf    : the first enclosure modes on the walls and on a cut through the driver axis
  - deformation_3d.pdf  : the deformed driver (all orders summed) at the frequencies where the
                          non-axisymmetric orders peak
Results/Files/
  - coupled_modes.csv   : the coupled modes (f, ζ, share of each driver order, main enclosure mode)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import loudpy as lp
from loudpy.physics.modal import CavityModes
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))   # the shared box
from enclosure import X0, Z_B  # noqa: E402

HERE = Path(__file__).resolve().parent               # this example's folder
EXAMPLES = HERE.parent                               # LoudPy_exemples/

lp.plot.style.use()                       # the loudpy figure style for this script

here = HERE / "Results"
N_PEAKS = 6                     # frequencies shown in deformation_3d.pdf
RIM = (0.040, 0.0265)           # cone point of the rocking signature (r, z) [m]
N_CAV_SHOWN = 8                 # enclosure modes shown
MIC = (0.03, 0.04, Z_B - 0.08)  # pressure probe in the enclosure [m]

res = lp.load(here / "Files/coupled.h5")
hp = res.driver("main")
vac = hp.in_vacuo()
f_cav = res.cavity_frequencies


def cavity_lines(ax):
    for fc in f_cav[1:]:
        if fc < res.frequencies[-1]:
            ax.axvline(fc, color="0.85", lw=0.5, zorder=0)


# ── response.pdf ───────────────────────────────────────────────────────────────
E = hp.kinetic_energy(by_order=True)                                   # (f, order)
E_tot = hp.kinetic_energy()
share = lp.Data(100 * E.values[:, 1:] / E_tot.values[:, None], {"f": res.frequencies, "order": E.coords["order"][1:]},
                lp.results.Quantity("share", "Share of the kinetic energy", "", "%"))
with lp.plot.pdf_pages(here / "Figures/response.pdf") as pdf:
    fig, ax = lp.plot.figure("tall", 2, 2, sharex=True)
    E.plot(ax=ax[0, 0])
    vac.kinetic_energy().rename(label="no enclosure").plot(ax=ax[0, 0], color="k", ls="--", lw=0.8)
    ax[0, 0].legend()
    share.plot(ax=ax[0, 1])
    lp.Data(share.values.sum(axis=1), {"f": res.frequencies}, share.quantity, "all m ≥ 1").plot(ax=ax[0, 1], color="k")
    hp.excursion().velocity().rename(label="in the enclosure").plot(ax=ax[1, 0])
    vac.excursion().velocity().rename(label="no enclosure").plot(ax=ax[1, 0], color="k", ls="--", lw=0.8)
    res.spl(at=MIC).plot(ax=ax[1, 1])
    ax[1, 1].set_title(f"in the enclosure at {np.round(MIC, 3)} m", fontsize=8)
    for a in ax.ravel():
        cavity_lines(a)
    for a in ax[0]:
        a.set_xlabel("")
    pdf.savefig(fig)

    fig, ax = lp.plot.figure("wide")
    lp.plot.line([hp.displacement(at=RIM, theta=np.deg2rad(th), component="z").rename(label=f"θ = {th}°")
                  for th in (0, 90, 180, 270)], ax=ax,
                 title=f"rocking signature: axial displacement at r = {1e3 * RIM[0]:.0f} mm, four angles "
                       f"(identical without the enclosure)")
    pdf.savefig(fig)

# ── coupled_modes.pdf / .csv ───────────────────────────────────────────────────
cm = lp.load(here / "Files/coupled_modes.h5")
s, Xq, XP = cm.extra("s"), cm.extra("q").T, cm.extra("P").T
fn, zeta = np.abs(s) / (2 * np.pi), -s.real / np.abs(s)
orders = list(E.coords["order"])
part = np.array([[np.sum(np.abs(Xq[hp.modes.m == m, i]) ** 2) for m in orders] for i in range(len(s))])
part = 100 * part / part.sum(axis=1, keepdims=True)
table = pd.DataFrame({"f [Hz]": fn, "zeta": zeta, **{f"% m={m}": part[:, k] for k, m in enumerate(orders)},
                      "main enclosure mode [Hz]": f_cav[np.argmax(np.abs(XP), axis=0)]})
table.index.name = "mode"
table.to_csv(here / "Files/coupled_modes.csv")
fig, ax = lp.plot.figure("wide")
dom = np.argmax(part, axis=1)
for k, m in enumerate(orders):
    ax.plot(fn[dom == k], zeta[dom == k], "o", ms=3.5, label=f"dominant driver order $m$ = {m}")
cavity_lines(ax)
lp.plot.style.frequency_axis(ax)
ax.set_yscale("log")
ax.set_ylabel(lp.plot.style.LABELS["zeta"])
ax.legend()
lp.plot.save(fig, here / "Figures/coupled_modes.pdf")

# ── cavity_modes.pdf: enclosure modes on the walls and on the cut ──────────────
cav = CavityModes.load(EXAMPLES / "shared/meshes/cavity_modes.h5")
camera = ((-0.45, 0.05, 0.25), (0.125, 0.15, -0.04), (0, 0, 1))
with lp.plot.pdf_pages(here / "Figures/cavity_modes.pdf") as pdf:
    for start in range(1, 1 + N_CAV_SHOWN, 4):
        fig, axs = lp.plot.figure("wide", 2, 4)
        for j, n in enumerate(range(start, start + 4)):
            mode = res.cavity_mode(n, cavity=cav)
            for ax, cut in ((axs[0, j], None), (axs[1, j], ((-1, 0, 0), (X0, 0, 0)))):
                mode.plot("real", ax=ax, camera=camera, cut=cut, colorbar=False, size=(600, 600),
                          title=f"mode {n}: {mode.coordinate:.0f} Hz" if cut is None else "cut through the axis")
        pdf.savefig(fig)

# ── deformation_3d.pdf: the driver where the non-axisymmetric orders peak ─────
frac = share.values.sum(axis=1)
cand = [i for i in range(1, len(frac) - 1) if frac[i] >= frac[i - 1] and frac[i] >= frac[i + 1]]
peaks = sorted(sorted(cand, key=lambda i: -frac[i])[:N_PEAKS])
with lp.plot.pdf_pages(here / "Figures/deformation_3d.pdf") as pdf:
    for start in range(0, len(peaks), 3):
        fig, axs = lp.plot.figure((7.2, 8.2), 3, 2, squeeze=False)
        for a in axs.ravel():
            a.set_axis_off()
        for row, i in zip(axs, peaks[start:start + 3]):
            f = res.frequencies[i]
            for ax, sweep, view in ((row[0], 1.5 * np.pi, ("xz", 35, 25)), (row[1], 2 * np.pi, ("xy", 0, 0))):
                surface, _ = hp.surface(f, local=True, theta=(0.0, sweep))
                lp.plot.View3D((620, 500)).add(surface, scalars="|u|", clim=(0, 1), colorbar=False,
                                               smooth_shading=True).view(view[0], view[1], view[2], 1.3).show(ax)
            parts = ", ".join(f"$m$ = {m}: {E.values[i, k] / E_tot.values[i] * 100:.0f} %" for k, m in enumerate(orders))
            row[0].set_title(f"{f:.0f} Hz — kinetic energy {parts}", fontsize=8)
            row[1].set_title("front view", fontsize=8)
        pdf.savefig(fig)
print(f"✓ figures → {here / 'Figures'}")
