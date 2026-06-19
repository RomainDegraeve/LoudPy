"""
Running_script.py — Structural harmonic sweep (mechanics only, no acoustics).

The mechanical FE matrices (K, M, C) are assembled once.  The complex linear
system  (-ω²M + jωC + K) u = F  is then solved independently at each
frequency in f_array.  This is valid when the acoustic back-pressure on the
cone is negligible (in-air, low-frequency regime) or when the acoustic
coupling is handled separately.

What this produces
------------------
    Results/Files/snap_f_20-20000Hz.h5   — one snapshot per frequency

Typical use
-----------
Run once.  Then use Loading_script.py to visualise the results.
"""
import os
import numpy as np

# Avoids an OpenMP conflict that arises when scipy (OpenBLAS/MKL) and
# numba both load their own OpenMP runtime simultaneously.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from loudpy.Studies import Problem, FreqStudy
from loudpy import (
    DomainSpecMecaRayleigh,
    InterfaceSpecClamped,
    InterfaceSpecForced,
    InterfaceSpecAcouMeca,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
geo_path = "LoudPy_exemples/Geometries/Loudspeaker/HPNEW-Sketch.geo"
msh_path = "LoudPy_exemples/Geometries/Loudspeaker/HPNEW-Sketch.msh"
mat_path = "src/loudpy/Materials_Bank/materials.json"
out_path = "LoudPy_exemples/Frequency_study/Meca/Results/Files/"

# ── Simulation parameters ──────────────────────────────────────────────────────
f_array = np.logspace(np.log10(20), np.log10(20000), 400)  # 400 log-spaced freqs [Hz]
force   = 0.1   # force amplitude applied at the excitation interface [N]

# ── Problem definition ─────────────────────────────────────────────────────────
problem = Problem(geo_path=geo_path, msh_path=msh_path,
                  mat_path=mat_path, subdomains_key="sub")

problem.add_sub_domain(
    DomainSpecMecaRayleigh("membranne",  material="Paper",         size=0.0005),
    DomainSpecMecaRayleigh("coil",       material="Copper",        size=0.0005),
    DomainSpecMecaRayleigh("surround",   material="Rubber",        size=0.0005),
    DomainSpecMecaRayleigh("spider",     material="PhenolicCloth", size=0.0005),
    DomainSpecMecaRayleigh("former",     material="Kapton",        size=0.0005),
    DomainSpecMecaRayleigh("glue",       material="SolidGlue",     size=0.0005),
    DomainSpecMecaRayleigh("dustcap",    material="Polypropylene", size=0.0005),
)

problem.add_interface(
    InterfaceSpecClamped("interface_constrained"),
    InterfaceSpecForced("interface_forced"),
    InterfaceSpecAcouMeca("interface_acou_meca_front"),
    InterfaceSpecAcouMeca("interface_acou_meca_rear"),
)

# ── Mesh ───────────────────────────────────────────────────────────────────────
problem.mesh(show_mesh_gui=False)

# ── Frequency sweep ────────────────────────────────────────────────────────────
# assemble_meca() builds K, M, C once.
# solve_meca() factorises and solves at each frequency (the most expensive step).
study = FreqStudy(problem)
study.assemble_meca()

for f in f_array:
    study.solve_meca(freq=f, force=force)

# ── Save ───────────────────────────────────────────────────────────────────────
out_file = out_path + f"snap_f_{int(f_array.min())}-{int(f_array.max())}Hz.h5"
study.save(out_file, case="meca_sweep")
print(f"Saved → {out_file}")
