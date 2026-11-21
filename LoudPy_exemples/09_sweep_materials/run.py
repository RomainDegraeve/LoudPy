"""
run.py — a MATERIAL sweep: the HPNEW driver in vacuo for every stiffness of its surround and spider.

The Young's moduli of the two suspension parts are model parameters (lp.P) and replace the library values of
Rubber and PhenolicCloth; the geometry and the mesh do not change, so every run reuses the same mesh.

  model    : the driver of 04_response_structure, E of the surround and of the spider = lp.P(...)
  sweep    : lp.Sweep(lp.FrequencyResponse(model, drive=lp.Force(1.0)), {…}) — all combinations (3 × 3 = 9 runs),
             20 Hz – 2 kHz, 1 N on the voice coil

What this produces (Results/Files/)
------------------------------------
  speaker_sweep.h5           every run in one file (lp.load gives the SweepResults back)
Then run plot.py.
"""
from pathlib import Path
import numpy as np

import loudpy as lp

HERE = Path(__file__).resolve().parent               # this example's folder
EXAMPLES = HERE.parent                               # LoudPy_exemples/

lp.set_verbosity("info")

geometry = str(EXAMPLES / "Geometries/Loudspeaker/HPNEW-Sketch.geo")
files = str(HERE / "Results/Files")
E_RUBBER, E_PHENOLIC = lp.materials["Rubber"].E, lp.materials["PhenolicCloth"].E    # library values [Pa]
FACTORS = [0.5, 1.0, 2.0]                                                            # × the library value
FREQS = np.geomspace(20.0, 2000.0, 300)

size = 0.0005
model = lp.Model(geometry, parameters=lp.Parameters(E_surround=E_RUBBER, E_spider=E_PHENOLIC))
model.add(
    lp.Solid("membranne", "Paper", mesh_size=size),
    lp.Solid("coil", "Copper", mesh_size=size),
    lp.Solid("surround", "Rubber", E=lp.P("E_surround"), mesh_size=size),
    lp.Solid("spider", "PhenolicCloth", E=lp.P("E_spider"), mesh_size=size),
    lp.Solid("former", "Kapton", mesh_size=size),
    lp.Solid("glue", "SolidGlue", mesh_size=size),
    lp.Solid("dustcap", "Polypropylene", mesh_size=size),
    lp.Clamped("interface_constrained_rz"), lp.Clamped("interface_constrained_r", dofs="r"),
    lp.Coil("interface_forced"),
)
model.mesh(f"{files}/speaker.msh")

study = lp.FrequencyResponse(model, drive=lp.Force(1.0))
sweep = lp.Sweep(study, {"E_surround": [k * E_RUBBER for k in FACTORS],
                         "E_spider": [k * E_PHENOLIC for k in FACTORS]}).solve(FREQS)
print(sweep)
sweep.save(f"{files}/speaker_sweep.h5")
print(f"Saved → {files}/speaker_sweep.h5")
