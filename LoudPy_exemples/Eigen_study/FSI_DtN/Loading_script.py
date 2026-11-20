"""
Loading_script.py — the coupled structure + air modes with the exact DtN radiation condition
(Running_script.py first).

The in-vacuo modes of ../Meca (when available) tell the two families of coupled modes apart:
  - air-loaded structural modes: close to an in-vacuo mode (within MATCH_F_TOL in frequency and
    MATCH_ZETA_TOL in damping ratio);
  - radiation ("air") modes: the others. They depend on the DtN radius; individually they have no
    physical meaning, only their sum does (modal superposition of the pressure).

What this produces
------------------
Results/Figures/
  - eigenvalues.pdf        : ζ against f, and the eigenvalues in the complex plane
  - modes_structural.pdf   : relative pressure level in the air + deformed structure, structural modes
  - modes_first.pdf        : the same for the first N_FIRST coupled modes
Results/Files/
  - modes.csv              : the coupled mode table (f, ζ, family)
"""
from pathlib import Path

import numpy as np

import loudpy as lp

lp.plot.style.use()                       # the loudpy figure style for this script

here = Path("LoudPy_exemples/Eigen_study/FSI_DtN/Results")
vacuo_file = Path("LoudPy_exemples/Eigen_study/Meca/Results/Files/modes.h5")
MATCH_F_TOL, MATCH_ZETA_TOL = 0.05, 0.15   # structural ↔ in-vacuo: relative tolerances on f and ζ
N_FIRST, PER_PAGE = 12, 6
WINDOW = dict(xlim=(0.0, 0.21), ylim=(-0.24, 0.24))       # air shown around the driver [m]

modes = lp.load(here / "Files/modes.h5").set("fsi")
freq, zeta, s = modes.frequencies, modes.zeta, modes.extra("s")
print(f"{modes}: {modes.mesh('solid').n_nodes} structural and {modes.mesh('air').n_nodes} air nodes")

# ── structural modes: each in-vacuo mode matched to its closest coupled mode ───
pairs = []
if vacuo_file.exists():
    vac = lp.load(vacuo_file)
    df = np.abs(freq[None, :] - vac.frequencies[:, None]) / vac.frequencies[:, None]
    dz = np.abs(zeta[None, :] - vac.zeta[:, None]) / vac.zeta[:, None]
    cost = np.where((df < MATCH_F_TOL) & (dz < MATCH_ZETA_TOL), df + dz, np.inf)
    used_v, used_c = set(), set()
    for iv, ic in zip(*np.unravel_index(np.argsort(cost, axis=None), cost.shape)):
        if not np.isfinite(cost[iv, ic]):
            break
        if iv not in used_v and ic not in used_c:
            pairs.append((int(iv), int(ic)))
            used_v.add(iv)
            used_c.add(ic)
    print(f"{len(pairs)} structural modes matched to the {len(vac)} in-vacuo modes")
else:
    vac = None
    print(f"no in-vacuo modes ({vacuo_file}): run ../Meca/Running_script.py to classify the modes")
vac_of = {ic: iv for iv, ic in sorted(pairs)}
structural = np.array(sorted(vac_of, key=lambda k: freq[k]), int)
table = modes.table()
table["family"] = ["structural" if k in vac_of else "radiation" for k in range(len(modes))]
table.to_csv(here / "Files/modes.csv")

# ── eigenvalues.pdf ────────────────────────────────────────────────────────────
air = np.setdiff1d(np.arange(len(freq)), structural)
fig, (a1, a2) = lp.plot.figure("wide", 1, 2)
a1.plot(freq[air], zeta[air], "o", ms=2.5, color="0.65", label=f"radiation modes ({len(air)})")
a2.plot(-s.real[air] / (2 * np.pi), s.imag[air] / (2 * np.pi), "o", ms=2.5, color="0.65")
if len(structural):
    a1.plot(freq[structural], zeta[structural], "o", ms=4, label=f"air-loaded structural ({len(structural)})")
    a2.plot(-s.real[structural] / (2 * np.pi), s.imag[structural] / (2 * np.pi), "o", ms=4)
if vac is not None:
    a1.plot(vac.frequencies, vac.zeta, "x", ms=4, label=f"in vacuo ({len(vac)})")
    a2.plot(vac.zeta * vac.frequencies, vac.frequencies * np.sqrt(1 - vac.zeta ** 2), "x", ms=4)
lp.plot.style.frequency_axis(a1)
a1.set_ylabel(lp.plot.style.LABELS["zeta"])
a1.legend()
a2.set_xscale("symlog", linthresh=1.0)
a2.set_xlabel(r"$-\mathrm{Re}\,s/2\pi$ [Hz]")
a2.set_ylabel(r"$\mathrm{Im}\,s/2\pi$ [Hz]")
lp.plot.save(fig, here / "Figures/eigenvalues.pdf")


# ── mode pages ─────────────────────────────────────────────────────────────────
def pages(name, indices):
    with lp.plot.pdf_pages(here / f"Figures/{name}") as pdf:
        for start in range(0, len(indices), PER_PAGE):
            idx = [int(k) for k in indices[start:start + PER_PAGE]]
            labels = [f"#{k} (in vacuo #{vac_of[k]})" if k in vac_of else f"#{k} (radiation)" for k in idx]
            pdf.savefig(lp.plot.coupled_modes(modes, idx, labels=labels, **WINDOW))
    print(f"  ✓ {len(indices)} modes → {here / 'Figures' / name}")


if len(structural):
    pages("modes_structural.pdf", structural)
pages("modes_first.pdf", np.argsort(freq)[:N_FIRST])
