"""
bench_solvers.py — timing (and reference results) of the loudpy solvers on the HPNEW driver.

    python3 LoudPy_exemples/Benchmarks/bench_solvers.py <case> <tag>

case : fsi_elec   lp.FrequencyResponse, voltage drive, structure + air + PML on one mesh (1.9 kHz band),
                  N_FREQ frequencies from 1900 Hz
       fsi        same, force drive
       meca       lp.FrequencyResponse of the structure, N_FREQ_MECA frequencies
       eigen      lp.Modes (multishift), 20 Hz – EIG_FMAX
       fsi_eigen  lp.Modes(physics="coupled") (exact DtN), 10 Hz – 3.2 kHz, 12 shifts
       time       lp.Transient, N_STEPS steps of a two-tone signal
tag  : label of the run (e.g. "baseline", "fast"); results go to Results/<case>_<tag>.npz so that
       two runs can be compared (compare_solvers.py).
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

import loudpy as lp

lp.set_verbosity("info")                 # progress messages (loudpy is silent by default)

CASE = sys.argv[1]
TAG = sys.argv[2] if len(sys.argv) > 2 else "run"
out = Path("LoudPy_exemples/Benchmarks/Results")
out.mkdir(parents=True, exist_ok=True)
GEO = "LoudPy_exemples/Geometries/Loudspeaker/HPNEW-Sketch.geo"
DTN_GEO = "LoudPy_exemples/Geometries/Loudspeaker/HPNEW-Sketch_front_rear_dtn.geo"
N_FREQ, N_FREQ_MECA, EIG_FMAX = 40, 200, 20000.0
N_STEPS = int(os.environ.get("N_STEPS", 150))
PARTS = [("membranne", "Paper"), ("coil", "Copper"), ("surround", "Rubber"), ("spider", "PhenolicCloth"),
         ("former", "Kapton"), ("glue", "SolidGlue"), ("dustcap", "Polypropylene")]


def structure(geo=GEO, coil=0.0005):
    model = lp.Model(geo)
    model.add([lp.Solid(n, m, mesh_size=coil if n == "coil" else 0.0005) for n, m in PARTS],
              lp.Clamped("interface_constrained_rz"), lp.Clamped("interface_constrained_r", dofs="r"), lp.Coil("interface_forced"),
              lp.FluidStructure("interface_acou_meca_front"))
    return model


def coupled(f0=1900.0, c=344.0):
    lam = c / f0
    model = structure(coil=0.0015)
    model.add(lp.FluidStructure("interface_acou_meca_rear"), lp.Air("subacou", mesh_size=min(lam / 6, 0.005)),
              lp.PML("PML", thickness=lam, mesh_size=lam / 8, tuned_to=f0))
    model.mesh(out / "mesh_fsi.msh")
    return model


t_all = time.time()
if CASE in ("fsi_elec", "fsi"):
    drive = lp.Voltage(np.sqrt(8.0), Re=8, Le=4e-3, n=0.8, Bl=7) if CASE == "fsi_elec" else lp.Force(1.0)
    study = lp.FrequencyResponse(coupled(), drive=drive)
    t0 = time.time()
    res = study.solve(np.linspace(1900, 2560, N_FREQ))
    t_sol = time.time() - t0
    data = dict(freqs=res.frequencies, U=res.blocks[0].fields["u"], P=res.blocks[0].fields["p"])
    print(f"{CASE}: {N_FREQ} frequencies in {t_sol:.2f} s ({t_sol / N_FREQ:.3f} s/freq), {study.solver_stats}")
elif CASE == "meca":
    model = structure()
    model.mesh(out / "mesh_meca.msh")
    study = lp.FrequencyResponse(model)
    t0 = time.time()
    res = study.solve(np.logspace(1, np.log10(20000), N_FREQ_MECA))
    t_sol = time.time() - t0
    data = dict(freqs=res.frequencies, U=res.blocks[0].fields["u"])
    print(f"meca: {N_FREQ_MECA} frequencies in {t_sol:.2f} s ({t_sol / N_FREQ_MECA * 1e3:.1f} ms/freq)")
elif CASE == "eigen":
    model = structure()
    model.mesh(out / "mesh_meca.msh")
    t0 = time.time()
    modes = lp.Modes(model).solve(f_max=EIG_FMAX, n_per_shift=30)
    t_sol = time.time() - t0
    data = dict(freqs=modes.frequencies, zeta=modes.zeta)
    print(f"eigen: {len(modes)} modes ≤ {EIG_FMAX:.0f} Hz in {t_sol:.2f} s")
elif CASE == "fsi_eigen":
    model = structure(DTN_GEO, coil=0.0015)
    model.add(lp.FluidStructure("interface_acou_meca_rear"), lp.Air("subacou", mesh_size=0.01),
              lp.Radiation("dtn_boundary"))
    model.mesh(out / "mesh_dtn.msh")
    t0 = time.time()
    modes = lp.Modes(model, physics="coupled").solve(
        f_max=3200.0, f_min=10.0, n_per_shift=30, mac_tol=0.95, cover_zeta_max=0.999, max_shifts=12,
        dtn_center={"dtn_boundary_front": (0.0, 0.032), "dtn_boundary_rear": (0.0, 0.031)}, dtn_half=True,
        dtn_order=18)
    t_sol = time.time() - t0
    data = dict(freqs=modes.frequencies, zeta=modes.zeta)
    print(f"fsi_eigen: {len(modes)} coupled modes in {t_sol:.2f} s")
elif CASE == "time":
    model = structure()
    model.mesh(out / "mesh_meca.msh")
    fs = 48000
    t = np.arange(N_STEPS + 1) / fs
    sig = np.sin(2 * np.pi * 200 * t) + 0.5 * np.sin(2 * np.pi * 950 * t + 1.0)
    t0 = time.time()
    res = lp.Transient(model, drive=lp.Force(sig, fs=fs)).solve()
    t_sol = time.time() - t0
    data = dict(U=res.blocks[0].fields["u"])
    print(f"time: {N_STEPS} steps in {t_sol:.2f} s ({t_sol / N_STEPS * 1e3:.1f} ms/step)")
else:
    raise SystemExit(__doc__)
data["t_solve"] = t_sol
np.savez_compressed(out / f"{CASE}_{TAG}.npz", **data)
print(f"total {time.time() - t_all:.1f} s → {out / f'{CASE}_{TAG}.npz'}")
