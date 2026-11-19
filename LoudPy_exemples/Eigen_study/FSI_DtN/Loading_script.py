"""
Loading_script.py — the coupled structure + air modes of the DtN study (Running_script.py first).

The in-vacuo modes of ../Meca (when available) tell the two families of coupled modes apart:
  - air-loaded structural modes: close to an in-vacuo mode (within MATCH_F_TOL in frequency
    and MATCH_ZETA_TOL in damping ratio);
  - radiation ("air") modes: the others. They depend on the DtN radius; individually they
    have no physical meaning, only their sum does (modal superposition of the pressure).

What this produces (in Results/Figures/)
-----------------------------------------
  - eigenvalues.pdf        : ζ vs f and complex plane (coupled, structural, in-vacuo modes)
  - modes_structural.pdf   : SPL in the air + deformed structure of every structural mode
  - modes_first.pdf        : the same for the first N_FIRST coupled modes (by frequency)
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

import loudpy as lp

lp.plot.style.use()                       # loudpy figure style (fonts, colours, SPL in dB RMS)

# ── Paths ──────────────────────────────────────────────────────────────────────
in_path = "LoudPy_exemples/Eigen_study/FSI_DtN/Results/Files/modes.h5"
vacuo_path = Path("LoudPy_exemples/Eigen_study/Meca/Results/Files/modes.h5")      # ../Meca
out_dir = Path("LoudPy_exemples/Eigen_study/FSI_DtN/Results/Figures")
out_dir.mkdir(parents=True, exist_ok=True)

# ── Settings ───────────────────────────────────────────────────────────────────
MATCH_F_TOL = 0.05             # structural ↔ in-vacuo: relative tolerance on f
MATCH_ZETA_TOL = 0.15          #                        relative tolerance on ζ
N_FIRST = 12                   # first coupled modes shown in modes_first.pdf
PER_PAGE = 6                   # modes per page of the mode grids
P_XLIM, P_YLIM = (0.0, 0.21), (-0.24, 0.24)   # air window of the mode plots [m]
SPL_RANGE = 60.0               # colour range of the SPL maps [dB]

# ── Load ───────────────────────────────────────────────────────────────────────
modes = lp.load(in_path).set("fsi")
freq, zeta = modes.frequencies, modes.zeta
s_k = modes.extra("s") if "s" in modes.blocks[0].extra else \
    2 * np.pi * freq * (-zeta + 1j * np.sqrt(np.maximum(1 - zeta ** 2, 0.0)))
print(f"{modes}: {modes.mesh('solid').n_nodes} structural and {modes.mesh('air').n_nodes} air nodes")

# ── Structural modes: match each in-vacuo mode to its closest coupled mode ─────
structural, pairs = np.array([], int), []
if vacuo_path.exists():
    vac = lp.load(vacuo_path)
    vf, vz = vac.frequencies, vac.zeta
    df = np.abs(freq[None, :] - vf[:, None]) / vf[:, None]
    dz = np.abs(zeta[None, :] - vz[:, None]) / vz[:, None]
    cost = np.where((df < MATCH_F_TOL) & (dz < MATCH_ZETA_TOL), df + dz, np.inf)
    used_v, used_c = set(), set()
    for iv, ic in zip(*np.unravel_index(np.argsort(cost, axis=None), cost.shape)):
        if not np.isfinite(cost[iv, ic]):
            break
        if iv not in used_v and ic not in used_c:
            pairs.append((int(iv), int(ic)))
            used_v.add(iv)
            used_c.add(ic)
    pairs.sort()
    structural = np.array([ic for _, ic in pairs], int)
    print(f"{len(structural)} structural modes matched to the {len(vf)} in-vacuo modes")
else:
    vf = vz = None
    print(f"No in-vacuo modes ({vacuo_path}): run ../Meca/Running_script.py to classify the modes")
air = np.setdiff1d(np.arange(len(freq)), structural)

# ── eigenvalues.pdf ────────────────────────────────────────────────────────────
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5))
a1.plot(freq[air], zeta[air], "o", ms=3, color="0.6", label=f"radiation (air) modes ({len(air)})")
a2.plot(-s_k.real[air] / (2 * np.pi), s_k.imag[air] / (2 * np.pi), "o", ms=3, color="0.6")
if len(structural):
    a1.plot(freq[structural], zeta[structural], "o", ms=6, color="C0",
            label=f"air-loaded structural modes ({len(structural)})")
    a2.plot(-s_k.real[structural] / (2 * np.pi), s_k.imag[structural] / (2 * np.pi), "o", ms=6, color="C0")
if vf is not None:
    a1.plot(vf, vz, "x", ms=6, color="C3", label=f"in vacuo ({len(vf)})")
    a2.plot(vz * vf, vf * np.sqrt(1 - vz ** 2), "x", ms=6, color="C3")
a1.set_xlabel("f [Hz]")
a1.set_ylabel(r"$\zeta$")
a1.set_title("Damping ratio vs frequency", fontsize=10)
a1.legend(fontsize=8)
a2.set_xscale("symlog", linthresh=1.0)
a2.set_xlabel(r"$-\mathrm{Re}(\tilde s_m)/2\pi$ [Hz]")
a2.set_ylabel(r"$\mathrm{Im}(\tilde s_m)/2\pi$ [Hz]")
a2.set_title("Complex plane (upper half)", fontsize=10)
for a in (a1, a2):
    a.grid(True)
fig.suptitle("Coupled fluid-structure modes, exact DtN radiation condition", fontsize=11)
fig.tight_layout()
lp.plot.style.tidy(fig).savefig(out_dir / "eigenvalues.pdf", bbox_inches="tight")
plt.close(fig)

# ── mode grids ─────────────────────────────────────────────────────────────────
vac_of = {ic: iv for iv, ic in pairs}


def label(k):
    """Mode number and family (the grid adds f and zeta itself)."""
    return f"#{k} (in vacuo #{vac_of[k]})" if k in vac_of else f"#{k} (air)"


def mode_pages(name, idx_all, title):
    if not len(idx_all):
        return
    with PdfPages(out_dir / name) as pdf:
        for start in range(0, len(idx_all), PER_PAGE):
            idx = [int(k) for k in idx_all[start:start + PER_PAGE]]
            fig = lp.plot.coupled_modes_grid(modes, idx, labels=[label(k) for k in idx], p_xlim=P_XLIM,
                                             p_ylim=P_YLIM, spl_range=SPL_RANGE, deform_scale=0.05)
            fig.suptitle(f"{title}: {start + 1}-{start + len(idx)} of {len(idx_all)}", fontsize=9, y=1.01)
            pdf.savefig(lp.plot.style.tidy(fig), bbox_inches="tight", dpi=150)
            plt.close(fig)
    print(f"  ✓ {len(idx_all)} modes → {out_dir / name}")


mode_pages("modes_structural.pdf", structural[np.argsort(freq[structural])], "Air-loaded structural modes")
mode_pages("modes_first.pdf", np.argsort(freq)[:N_FIRST], "First coupled modes")
print(f"✓ eigenvalues.pdf → {out_dir}")
