"""
Running_script.py — coupled structure + air (FSI) frequency sweep driven by a voltage on the
voice coil (electrical circuit Ze = Re + (jω)^n Le, force factor Bl).

The air mesh follows the wavelength (lp.PerWavelength): a new mesh when the sweep crosses
one of REMESH_AT; the air element size, the PML thickness, element size and tuning follow
the wavelength at the first frequency of each band.

What this produces
------------------
    Results/Files/sweep.h5         — all frequencies, one block per mesh (loudpy result format 2),
                                     with the coil current I_coil per frequency
    Results/Files/mesh_<f>Hz.msh   — the mesh of each band (reused on reruns)

Then run Loading_script.py.
"""
import numpy as np

import loudpy as lp

lp.set_verbosity("info")                 # progress messages (loudpy is silent by default)

# ── Paths ──────────────────────────────────────────────────────────────────────
geometry = "LoudPy_exemples/Geometries/Loudspeaker/HPNEW-Sketch.geo"
files = "LoudPy_exemples/Frequency_study/FSI_elec/Results/Files"
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
    lp.Solid("membranne", "Paper", mesh_size=size),
    lp.Solid("coil", "Copper", mesh_size=0.0015),
    lp.Solid("surround", "Rubber", mesh_size=size),
    lp.Solid("spider", "PhenolicCloth", mesh_size=size),
    lp.Solid("former", "Kapton", mesh_size=size),
    lp.Solid("glue", "SolidGlue", mesh_size=size),
    lp.Solid("dustcap", "Polypropylene", mesh_size=size),
    lp.Air("subacou", mesh_size=0.005),
    lp.PML("PML"),                              # strength and profile order of the Air material
    lp.Clamped("interface_constrained"),
    lp.Coil("interface_forced"),
    lp.FluidStructure("interface_acou_meca_front"),
    lp.FluidStructure("interface_acou_meca_rear"),
)

# air: λ/6 (at most 5 mm); PML: one wavelength thick, 8 elements across, tuned to the band start
remesh = lp.PerWavelength(bands=REMESH_AT, air_elements=6, air_max_size=0.005,
                          pml_thickness=1.0, pml_elements=8, c=344.0, folder=files)

# ── Solve ──────────────────────────────────────────────────────────────────────
study = lp.FrequencyResponse(model, drive=drive)
res = study.solve(freqs, remesh=remesh)
print(res)
study.save(out_path)
print(f"Saved → {out_path}")
