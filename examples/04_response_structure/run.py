"""
run.py — structural harmonic sweep (structure only, no air).

The structural matrices (K, M, C) are assembled once and (−ω²M + jωC + K) u = F is solved at
every frequency (the factorisation is reused between close frequencies). Valid when the air
load on the cone is negligible, or handled separately.

What this produces
------------------
    Results/Files/sweep.h5   — one entry per frequency (loudpy result format 3)

Then run plot.py.
"""
from pathlib import Path
import numpy as np

import loudpy as lp

HERE = Path(__file__).resolve().parent               # this example's folder
EXAMPLES = HERE.parent                               # examples/

lp.set_verbosity("info")                 # progress messages (loudpy is silent by default)

# ── Paths ──────────────────────────────────────────────────────────────────────
geometry = str(EXAMPLES / "Geometries/Loudspeaker/HPNEW-Sketch.geo")
mesh_path = str(HERE / "Results/Files/mesh.msh")   # reused on reruns
out_path = str(HERE / "Results/Files/sweep.h5")

# ── Sweep ──────────────────────────────────────────────────────────────────────
freqs = np.logspace(np.log10(10), np.log10(20000), 3000)   # [Hz]
force = 1.0                                                 # on the voice coil [N]

# ── Model ──────────────────────────────────────────────────────────────────────
size = 0.0005
model = lp.Model(geometry)
model.add(
    lp.Solid("cone", "Paper", mesh_size=size),
    lp.Solid("coil", "Copper", mesh_size=size),
    lp.Solid("surround", "Rubber", mesh_size=size),
    lp.Solid("spider", "PhenolicCloth", mesh_size=size),
    lp.Solid("former", "Kapton", mesh_size=size),
    lp.Solid("glue", "SolidGlue", mesh_size=size),
    lp.Solid("dustcap", "Polypropylene", mesh_size=size),
    lp.Clamped("rims", on=lp.Group("surround_rim", "spider_rim")), lp.Clamped("axis", dofs="r"),
    lp.Coil("coil_drive"),
    lp.FluidStructure("wet_front"),     # kept in the results for the plots
    lp.FluidStructure("wet_rear"),
)
model.mesh(mesh_path)

# ── Solve ──────────────────────────────────────────────────────────────────────
study = lp.FrequencyResponse(model, drive=lp.Force(force))
res = study.solve(freqs)
print(res)
study.save(out_path)
print(f"Saved → {out_path}")
