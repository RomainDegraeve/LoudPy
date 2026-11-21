"""
run.py — a GEOMETRY sweep: the first modes of a clamped steel plate for every radius a and thickness h.

The dimensions are DefineConstant[…] entries of Geometries/Plate/clamped_plate_param.geo; the model's parameters
of the same name replace them, so each run rebuilds and remeshes the geometry (as COMSOL's Parametric Sweep).

  model    : lp.Model(geo, parameters=lp.Parameters(a=…, h=…)), mesh size h/2 (lp.P: follows the thickness)
  sweep    : lp.Sweep(lp.Modes(model), {"a": […], "h": […]}) — all combinations (4 × 3 = 12 runs)

What this produces (Results/Files/)
------------------------------------
  plate_sweep.h5             every run in one file (lp.load gives the SweepResults back)
  plate_<a>_<h>.msh          the mesh of each run (reused on reruns)
Then run plot.py.

The model is built at the top; meshing and solving run only when this file is run (python3 run.py), so the
interface opens it without computing (python -m loudpy.ui on this folder or file) and can run it in the background.
"""
from pathlib import Path
import loudpy as lp

HERE = Path(__file__).resolve().parent               # this example's folder
EXAMPLES = HERE.parent                               # examples/

lp.set_verbosity("info")

geometry = str(EXAMPLES / "Geometries/Plate/clamped_plate_param.geo")
files = str(HERE / "Results/Files")
RADII = [0.06, 0.08, 0.10, 0.12]           # a [m]
THICKNESSES = [1e-3, 2e-3, 3e-3]           # h [m]

steel = lp.Material("Steel", rho=7850.0, E=2.1e11, nu=0.3)
model = lp.Model(geometry, parameters=lp.Parameters(a=0.1, h=2e-3))
model.add(
    lp.Solid("plate", steel, damping=lp.Undamped(), mesh_size=0.5 * lp.P("h")),   # two cells across
    lp.Clamped("rim"),
    lp.Clamped("axis", dofs="r"),
)
if __name__ == "__main__":
    model.mesh(f"{files}/plate.msh")
    print(model)

    sweep = lp.Sweep(lp.Modes(model), {"a": RADII, "h": THICKNESSES}).solve(f_min=10.0, f_max=12000.0)
    print(sweep)
    print(sweep.table())
    sweep.save(f"{files}/plate_sweep.h5")
    print(f"Saved → {files}/plate_sweep.h5")
