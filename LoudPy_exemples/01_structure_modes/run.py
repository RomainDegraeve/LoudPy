"""
run.py — structural modal analysis of a loudspeaker.

Damped natural frequencies and mode shapes of the whole moving structure in vacuo
(Rayleigh damping per material), with shift-invert solves spread over the band. These
in-vacuo modes are also used to identify the air-loaded structural modes of the coupled
DtN study (../FSI_DtN).

What this produces
------------------
    Results/Files/modes.h5   — frequencies, damping ratios, mode shapes (loudpy result format 3)

Then run plot.py.
"""
from pathlib import Path
import loudpy as lp

HERE = Path(__file__).resolve().parent               # this example's folder
EXAMPLES = HERE.parent                               # LoudPy_exemples/

lp.set_verbosity("info")                 # progress messages (loudpy is silent by default)

# ── Paths ──────────────────────────────────────────────────────────────────────
geometry = str(EXAMPLES / "Geometries/Loudspeaker/HPNEW-Sketch.geo")
mesh_path = str(HERE / "Results/Files/mesh.msh")      # reused on reruns
out_path = str(HERE / "Results/Files/modes.h5")

# ── Model ──────────────────────────────────────────────────────────────────────
size = 0.0005                                   # structural element size [m]
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
    lp.FluidStructure("interface_acou_meca_front"),     # kept in the results for the plots
)
model.mesh(mesh_path)                           # reused when the geometry and settings are unchanged

# ── Modes ──────────────────────────────────────────────────────────────────────
study = lp.Modes(model)
modes = study.solve(f_min=20.0, f_max=40000.0, n_per_shift=30)
print(modes)
study.save(out_path)
print(f"Saved → {out_path}")
