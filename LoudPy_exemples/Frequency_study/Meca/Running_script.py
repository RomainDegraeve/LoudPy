"""
Running_script.py — structural harmonic sweep (structure only, no air).

The structural matrices (K, M, C) are assembled once and (−ω²M + jωC + K) u = F is solved at
every frequency (the factorisation is reused between close frequencies). Valid when the air
load on the cone is negligible, or handled separately.

What this produces
------------------
    Results/Files/sweep.h5   — one entry per frequency (loudpy result format 2)

Then run Loading_script.py.
"""
import numpy as np

import loudpy as lp

lp.set_verbosity("info")                 # progress messages (loudpy is silent by default)

# ── Paths ──────────────────────────────────────────────────────────────────────
geometry = "LoudPy_exemples/Geometries/Loudspeaker/HPNEW-Sketch.geo"
mesh_path = "LoudPy_exemples/Frequency_study/Meca/Results/Files/mesh.msh"   # reused on reruns
out_path = "LoudPy_exemples/Frequency_study/Meca/Results/Files/sweep.h5"

# ── Sweep ──────────────────────────────────────────────────────────────────────
freqs = np.logspace(np.log10(10), np.log10(20000), 3000)   # [Hz]
force = 1.0                                                 # on the voice coil [N]

# ── Model ──────────────────────────────────────────────────────────────────────
size = 0.0005
model = lp.Model(geometry)
model.add(
    lp.Solid("membranne", "Paper", mesh_size=size),
    lp.Solid("coil", "Copper", mesh_size=size),
    lp.Solid("surround", "Rubber", mesh_size=size),
    lp.Solid("spider", "PhenolicCloth", mesh_size=size),
    lp.Solid("former", "Kapton", mesh_size=size),
    lp.Solid("glue", "SolidGlue", mesh_size=size),
    lp.Solid("dustcap", "Polypropylene", mesh_size=size),
    lp.Clamped("interface_constrained"),
    lp.Coil("interface_forced"),
    lp.FluidStructure("interface_acou_meca_front"),     # kept in the results for the plots
    lp.FluidStructure("interface_acou_meca_rear"),
)
model.mesh(mesh_path)

# ── Solve ──────────────────────────────────────────────────────────────────────
study = lp.FrequencyResponse(model, drive=lp.Force(force))
res = study.solve(freqs)
print(res)
study.save(out_path)
print(f"Saved → {out_path}")
