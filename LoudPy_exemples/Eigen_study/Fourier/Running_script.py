"""
Running_script.py — non-axisymmetric (azimuthal order m) modes of the loudspeaker.

The structure is axisymmetric, so its 3D modes separate into azimuthal orders m:

    u_r = U_r(r,z) cos mθ,   u_z = U_z(r,z) cos mθ,   u_θ = U_θ(r,z) sin mθ

  m = 0 : axisymmetric modes (pistonic, breakup) — the only ones the voice coil drives
  m = 1 : rocking (tilting of the moving parts), excited by any asymmetric load
          (box, neighbouring drivers, off-axis mass, ...)
  m ≥ 2 : ovalisation and higher circumferential modes of the cone / surround

Each order is a 2D problem on the same meridian mesh (3 DOF per node for m ≥ 1, exact
axis conditions set automatically). EigenStudy.solve_meca_eigen_fourier(m) uses the same
multishift solver, Rayleigh damping and normalisation as the axisymmetric study, so the
modes of all orders can be combined in a modal superposition (e.g. coupled to a 3D box).

What this produces
------------------
    Results/Files/eigen_fourier.h5   — modes/meca (m = 0) and modes/meca_m1 … meca_m4

Then run Loading_script.py.
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
msh_path = "LoudPy_exemples/Eigen_study/Fourier/Results/Files/mesh.msh"   # reused on reruns
mat_path = "src/loudpy/Materials_Bank/materials.json"
out_path = "LoudPy_exemples/Eigen_study/Fourier/Results/Files/eigen_fourier.h5"

# ── Settings ───────────────────────────────────────────────────────────────────
ORDERS = [0, 1, 2, 3, 4]       # azimuthal orders m
F_MIN, F_MAX = 20.0, 16000.0   # band of the mode search [Hz]
N_PER_SHIFT = 30               # eigenvalues requested per shift

# ── Problem (same loudspeaker and materials as ../Meca) ────────────────────────
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
# Clamped interfaces: "_rz_" also blocks the azimuthal displacement for m ≥ 1.
# The axis condition ("_r_" on r = 0) is replaced automatically by the exact one of each m.
problem.add_interface(
    InterfaceSpecClamped("interface_constrained"),
    InterfaceSpecForced("interface_forced"),
    InterfaceSpecAcouMeca("interface_acou_meca_front"),
)
problem.mesh(show_mesh_gui=False, reuse_mesh=True)

# ── Modal solve, one azimuthal order at a time ─────────────────────────────────
study = EigenStudy(problem)
for m in ORDERS:
    study.solve_meca_eigen_fourier(m, f_min=F_MIN, f_max=F_MAX, n_modes_per_shift=N_PER_SHIFT)
study.save(out_path)
print(f"Saved → {out_path}")
