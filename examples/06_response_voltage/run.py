"""
run.py — coupled structure + air (FSI) frequency sweep driven by a voltage on the
voice coil (electrical circuit Ze = Re + (jω)^n Le, force factor Bl).

The air mesh follows the wavelength (lp.PerWavelength): a new mesh when the sweep crosses
one of REMESH_AT; the air element size, the PML thickness, element size and tuning follow
the wavelength at the first frequency of each band.

What this produces
------------------
    Results/Files/sweep.h5         — all frequencies, one block per mesh (loudpy result format 3),
                                     with the coil current I_coil per frequency
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

# ── Sweep and drive ────────────────────────────────────────────────────────────
# integer-valued frequencies avoid duplicate FFT bins against time-domain results
freqs = np.concatenate([
    np.array([20, 30, 40, 50, 60, 70, 80]),
    np.linspace(82, 150, 80),
    np.linspace(151, 300, 60),
    np.linspace(301, 1500, 60),
    np.linspace(1501, 20000, 140),
])
P, Re = 1.0, 8.0                                # 1 W into the coil resistance
drive = lp.Voltage(np.sqrt(P * Re), Re=Re, Le=4e-3, n=0.8, Bl=7.0)
REMESH_AT = [20, 40, 80, 160, 320, 640, 1280, 1810, 2560, 3620, 5120, 7240, 10240, 15000]   # [Hz]

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
    lp.Air("air", mesh_size=0.005),
    lp.PML("PML", thickness=0.1, tuned_to=1000.0),   # strength / profile of the Air material; the remeshed bands
                                                     # set their own thickness and tuning
    lp.Clamped("rims", on=lp.Group("surround_rim", "spider_rim")), lp.Clamped("axis", dofs="r"),
    lp.Coil("coil_drive"),
    lp.FluidStructure("wet_front"),
    lp.FluidStructure("wet_rear"),
)

# air: λ/6 (at most 5 mm); PML: one wavelength thick, 8 elements across, tuned to the band start
remesh = lp.PerWavelength(bands=REMESH_AT, air_elements=6, air_max_size=0.005,
                          pml_thickness=1.0, pml_elements=8, c=344.0, folder=files)

if __name__ == "__main__":
    # ── Solve ──────────────────────────────────────────────────────────────────────
    study = lp.FrequencyResponse(model, drive=drive)
    res = study.solve(freqs, remesh=remesh)
    print(res)
    study.save(out_path)
    print(f"Saved → {out_path}")
