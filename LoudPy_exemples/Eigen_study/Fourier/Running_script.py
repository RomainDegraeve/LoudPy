"""
Running_script.py — non-axisymmetric (azimuthal order m) modes of the loudspeaker.

The structure is axisymmetric, so its 3D modes separate into azimuthal orders m:

    u_r = U_r(r,z) cos mθ,   u_z = U_z(r,z) cos mθ,   u_θ = U_θ(r,z) sin mθ

  m = 0 : axisymmetric modes (pistonic, breakup) — the only ones the voice coil drives
  m = 1 : rocking (tilting of the moving parts), excited by any asymmetric load
          (box, neighbouring drivers, off-axis mass, ...)
  m ≥ 2 : ovalisation and higher circumferential modes of the cone / surround

Each order is a 2D problem on the same meridian mesh (3 DOF per node for m ≥ 1, exact axis
conditions set automatically), solved with the same multishift solver, Rayleigh damping and
normalisation as the axisymmetric modes, so the modes of all orders can be combined in a
modal superposition (e.g. coupled to a 3D box: ../../Coupling_3D).

Clamped boundaries: "_rz_" also blocks the azimuthal displacement for m ≥ 1. The axis
condition ("_r_" on r = 0) is replaced automatically by the exact one of each m.

What this produces
------------------
    Results/Files/eigen_fourier.h5   — one mode set per order m (loudpy result format 3)

Then run Loading_script.py and Plot_3D_modes.py.
"""
import loudpy as lp

lp.set_verbosity("info")                 # progress messages (loudpy is silent by default)

# ── Paths ──────────────────────────────────────────────────────────────────────
geometry = "LoudPy_exemples/Geometries/Loudspeaker/HPNEW-Sketch.geo"
mesh_path = "LoudPy_exemples/Eigen_study/Fourier/Results/Files/mesh.msh"   # reused on reruns
out_path = "LoudPy_exemples/Eigen_study/Fourier/Results/Files/eigen_fourier.h5"

# ── Settings ───────────────────────────────────────────────────────────────────
ORDERS = [0, 1, 2, 3, 4]       # azimuthal orders m
F_MIN, F_MAX = 20.0, 16000.0   # band of the mode search [Hz]
N_PER_SHIFT = 30               # eigenvalues requested per shift

# ── Model (same loudspeaker and materials as ../Meca) ──────────────────────────
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
    lp.Clamped("interface_constrained_rz"), lp.Clamped("interface_constrained_r", dofs="r"),
    lp.Coil("interface_forced"),
    lp.FluidStructure("interface_acou_meca_front"),
)
model.mesh(mesh_path)

# ── Modes, one azimuthal order at a time ───────────────────────────────────────
study = lp.Modes(model)
modes = study.solve(f_min=F_MIN, f_max=F_MAX, orders=ORDERS, n_per_shift=N_PER_SHIFT)
print(modes, {m: len(modes.order(m)) for m in modes.orders})
study.save(out_path)
print(f"Saved → {out_path}")
