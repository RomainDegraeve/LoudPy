"""
Loading_script.py — the results of the modal analysis (Running_script.py first).

What this produces (in Results/Figures/)
-----------------------------------------
  - modes_grid.pdf        : all computed mode shapes
  - modes_selection.pdf   : a hand-picked subset of mode shapes
  - mode_k_full.pdf       : one mode on the whole structure
  - mode_k_surround.pdf   : that mode on the surround only
  - mode_k_interface.pdf  : deformed profile of the fluid-structure interface
"""
from pathlib import Path

import matplotlib.pyplot as plt

import loudpy as lp

lp.plot.style.use()                       # loudpy figure style (fonts, colours, SPL in dB RMS)

# ── Paths ──────────────────────────────────────────────────────────────────────
in_path = "LoudPy_exemples/Eigen_study/Meca/Results/Files/modes.h5"
out_dir = Path("LoudPy_exemples/Eigen_study/Meca/Results/Figures")
out_dir.mkdir(parents=True, exist_ok=True)

modes = lp.load(in_path)
print(modes)
print("Regions   :", modes.mesh("solid").names(2))
print("Boundaries:", list(modes.boundaries))

# ── Mode-shape grids ───────────────────────────────────────────────────────────
# |u| on the deformed mesh; deform_scale sets the visible deformation.
fig = lp.plot.modes_grid(modes, n=77, ncols=4, deform_scale=0.02)
lp.plot.style.tidy(fig).savefig(out_dir / "modes_grid.pdf", bbox_inches="tight")

fig = lp.plot.modes_grid(modes, indices=[0, 1, 2, 4, 7, 16, 17, 20, 32, 34, 39, 41, 42, 46, 47],
                         ncols=5, deform_scale=0.02)
lp.plot.style.tidy(fig).savefig(out_dir / "modes_selection.pdf", bbox_inches="tight")

# ── One mode: whole structure, surround only, interface profile ───────────────
k = 2                                           # 0-based: mode 3
mode = modes.mode(k)
title = f"Mode {k + 1} - {mode.coordinate:.1f} Hz"

ax = lp.plot.field(mode, title=title)
lp.plot.style.tidy(ax.figure).savefig(out_dir / f"mode_{k + 1}_full.pdf", bbox_inches="tight")

ax = lp.plot.field(mode.restrict("surround"), title=f"Surround - {title}")
lp.plot.style.tidy(ax.figure).savefig(out_dir / f"mode_{k + 1}_surround.pdf", bbox_inches="tight")

# rest position (dots) against the scaled deformed position (red) along the cone / air boundary
ax = lp.plot.deformed_boundary(mode, modes.boundary("interface_acou_meca_front"), scale=0.1,
                               title=f"Interface - {title}")
lp.plot.style.tidy(ax.figure).savefig(out_dir / f"mode_{k + 1}_interface.pdf", bbox_inches="tight")

plt.show()
