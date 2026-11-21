"""
run.py — coupled structure + air (FSI) frequency sweep, force on the coil.

Why the mesh changes during the sweep
--------------------------------------
The air mesh must resolve about 6 elements per wavelength λ = c/f, and λ shrinks by a factor
of 1000 from 20 Hz to 20 kHz: one mesh fine enough at 20 kHz would be far too large at 20 Hz.
lp.PerWavelength remeshes every REMESH_EVERY frequencies: the air element size, and the PML
thickness, element size and tuning follow the wavelength at the first frequency of each band.

What this produces
------------------
    Results/Files/sweep.h5         — all frequencies, one block per mesh (loudpy result format 3)
    Results/Files/mesh_<f>Hz.msh   — the mesh of each band (reused on reruns)

Then run plot.py.

The model is built at the top; meshing and solving run only when this file is run (python3 run.py), so the
interface opens it without computing (python -m loudpy.ui on this folder or file) and can run it in the background.
"""
from pathlib import Path
import numpy as np

import loudpy as lp

HERE = Path(__file__).resolve().parent               # this example's folder
EXAMPLES = HERE.parent                               # examples/

lp.set_verbosity("info")                 # progress messages (loudpy is silent by default)

# ── Paths ──────────────────────────────────────────────────────────────────────
geometry = str(EXAMPLES / "Geometries/Loudspeaker/HPNEW-Sketch.geo")
files = str(HERE / "Results/Files")
out_path = f"{files}/sweep.h5"

# ── Sweep ──────────────────────────────────────────────────────────────────────
force = 0.1                                     # on the voice coil [N]
# integer frequencies: no duplicate FFT bins when compared with time-domain results
freqs = np.unique(np.logspace(np.log10(20), np.log10(20000), 320).astype(int)).astype(float)
REMESH_EVERY = 30                               # frequencies sharing one mesh

# ── Model ──────────────────────────────────────────────────────────────────────
size = 0.0005
model = lp.Model(geometry)
model.add(
    lp.Solid("cone", "Paper", mesh_size=size),
    lp.Solid("coil", "Copper", mesh_size=0.0015),
    lp.Solid("surround", "Rubber", mesh_size=size),
    lp.Solid("spider", "PhenolicCloth", mesh_size=size),
    lp.Solid("former", "Kapton", mesh_size=size),
    lp.Solid("glue", "SolidGlue", mesh_size=size),
    lp.Solid("dustcap", "Polypropylene", mesh_size=size),
    lp.Air("air", mesh_size=0.005),                         # at most 5 mm (see remesh)
    lp.PML("PML", order=2, strength=1.5, thickness=0.1, tuned_to=1000.0),   # quadratic profile; the remeshed
                                                                         # bands set their own thickness / tuning
    lp.Clamped("rims", on=lp.Group("surround_rim", "spider_rim")), lp.Clamped("axis", dofs="r"),
    lp.Coil("coil_drive"),
    lp.FluidStructure("wet_front"),
    lp.FluidStructure("wet_rear"),
)

# air: λ/6 (at most 5 mm); PML: λ/4 thick, 4 elements across, tuned to the band start
remesh = lp.PerWavelength(every=REMESH_EVERY, air_elements=6, air_max_size=0.005,
                          pml_thickness=0.25, pml_elements=4, c=344.0, folder=files)

if __name__ == "__main__":
    # ── Solve ──────────────────────────────────────────────────────────────────────
    study = lp.FrequencyResponse(model, drive=lp.Force(force))
    res = study.solve(freqs, remesh=remesh)
    print(res)
    study.save(out_path)
    print(f"Saved → {out_path}")
