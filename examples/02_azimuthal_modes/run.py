"""
run.py — non-axisymmetric (azimuthal order m) modes of the loudspeaker.

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

Then run plot.py and plot_3d.py.

The model is built at the top; meshing and solving run only when this file is run (python3 run.py), so the
interface opens it without computing (python -m loudpy.ui on this folder or file) and can run it in the background.
"""
from pathlib import Path
import loudpy as lp

HERE = Path(__file__).resolve().parent               # this example's folder
EXAMPLES = HERE.parent                               # examples/

lp.set_verbosity("info")                 # progress messages (loudpy is silent by default)

# ── Paths ──────────────────────────────────────────────────────────────────────
geometry = str(EXAMPLES / "Geometries/Loudspeaker/HPNEW-Sketch.geo")
mesh_path = str(HERE / "Results/Files/mesh.msh")   # reused on reruns
out_path = str(HERE / "Results/Files/eigen_fourier.h5")

# ── Settings ───────────────────────────────────────────────────────────────────
ORDERS = [0, 1, 2, 3, 4]       # azimuthal orders m
F_MIN, F_MAX = 20.0, 16000.0   # band of the mode search [Hz]
N_PER_SHIFT = 30               # eigenvalues requested per shift

# ── Model (same loudspeaker and materials as ../Meca) ──────────────────────────
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
    lp.FluidStructure("wet_front"),
)
if __name__ == "__main__":
    model.mesh(mesh_path)

    # ── Modes, one azimuthal order at a time ───────────────────────────────────────
    study = lp.Modes(model)
    modes = study.solve(f_min=F_MIN, f_max=F_MAX, orders=ORDERS, n_per_shift=N_PER_SHIFT)
    print(modes, {m: len(modes.order(m)) for m in modes.orders})
    study.save(out_path)
    print(f"Saved → {out_path}")
