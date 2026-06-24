"""
Running_script.py — Structural modal analysis of a loudspeaker.

Computes the damped natural frequencies and mode shapes of the full
mechanical structure using an ARPACK shift-invert eigensolver.

What this produces
------------------
    Results/Files/eigen_study.h5   — frequencies, damping ratios, mode shapes

Typical use
-----------
Run once.  Then use Loading_script.py to visualise the results.
"""
from loudpy.Studies import Problem, EigenStudy
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
out_path = "LoudPy_exemples/Eigen_study/Results/Files/eigen_study.h5"

# ── Problem definition ─────────────────────────────────────────────────────────
# Each DomainSpecMecaRayleigh assigns a material and a target element size to
# one named physical group defined in the .geo file.
# Rayleigh damping coefficients (α, β) are read from materials.json and scale
# the mass and stiffness matrices: C = α·M + β·K.
problem = Problem(geo_path=geo_path, msh_path=msh_path, mat_path=mat_path)

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
)



# ── Meshing ────────────────────────────────────────────────────────────────────
# Set show_mesh_gui=True to preview the mesh in GMSH before solving.
problem.mesh(show_mesh_gui=False)

# ── Modal solve ────────────────────────────────────────────────────────────────
# ARPACK shift-invert finds n_modes eigenvalues near f_target [Hz].
# Increasing n_modes captures more resonances at the cost of memory and time.
study = EigenStudy(problem)
study.solve_meca_eigen_ARPAC(n_modes=45, f_target=1)

# ── Save ───────────────────────────────────────────────────────────────────────
study.save(out_path)
print(f"Saved → {out_path}")
