"""
bench_solvers.py — timing (and reference results) of the LoudPy solvers on the HPNEW driver.

    python3 LoudPy_exemples/Benchmarks/bench_solvers.py <case> <tag>

case : fsi_elec  FreqStudy.solve_fsi, voltage drive, FSI + PML block of the 1900 Hz mesh,
                 N_FREQ frequencies from 1900 Hz
       fsi       same, force drive
       meca      FreqStudy.solve_meca, N_FREQ_MECA frequencies
       eigen     EigenStudy.solve_meca_eigen_multishift, 20 Hz – EIG_FMAX
       fsi_eigen EigenStudy.solve_fsi_eigen (coupled, exact DtN), 10 Hz – 3.2 kHz, 12 shifts
       time      TimeStudy.solve_time_domain_rayleigh_nl, N_STEPS steps of the multitone signal
tag  : label of the run (e.g. "baseline", "fast"); results go to Results/<case>_<tag>.npz so
       two runs can be compared (compare_solvers.py).
"""
import os
import sys
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np

from loudpy.Studies import Problem, FreqStudy, EigenStudy, TimeStudy
from loudpy import (DomainSpecMecaRayleigh, DomainSpecAcou, DomainSpecPML, InterfaceSpecClamped,
                    InterfaceSpecForced, InterfaceSpecAcouMeca, InterfaceSpecDtN)

CASE = sys.argv[1]
TAG = sys.argv[2] if len(sys.argv) > 2 else "run"
out = Path("LoudPy_exemples/Benchmarks/Results")
out.mkdir(parents=True, exist_ok=True)
geo = "LoudPy_exemples/Geometries/Loudspeaker/HPNEW-Sketch.geo"
mat = "src/loudpy/Materials_Bank/materials.json"
N_FREQ, N_FREQ_MECA, EIG_FMAX = 40, 200, 20000.0
N_STEPS = int(os.environ.get("N_STEPS", 150))
MECA = [("membranne", "Paper"), ("coil", "Copper"), ("surround", "Rubber"), ("spider", "PhenolicCloth"),
        ("former", "Kapton"), ("glue", "SolidGlue"), ("dustcap", "Polypropylene")]


def meca_problem(msh):
    pb = Problem(geo_path=geo, msh_path=msh, mat_path=mat)
    pb.add_sub_domain(*[DomainSpecMecaRayleigh(n, material=m, size=0.0005) for n, m in MECA])
    pb.add_interface(InterfaceSpecClamped("interface_constrained"), InterfaceSpecForced("interface_forced"),
                     InterfaceSpecAcouMeca("interface_acou_meca_front"))
    pb.mesh(show_mesh_gui=False, reuse_mesh=True)
    return pb


def fsi_problem(f0=1900.0, c=344.0):
    msh = "LoudPy_exemples/Frequency_study/FSI_elec/Results/Files/mesh_1900Hz.msh"
    pb = Problem(geo_path=geo, msh_path=msh, mat_path=mat, subdomains_key="sub")
    pb.add_sub_domain(*[DomainSpecMecaRayleigh(n, material=m, size=0.0005) for n, m in MECA],
                      DomainSpecAcou("subacou", "Air"))
    pb.add_interface(InterfaceSpecClamped("interface_constrained"), InterfaceSpecForced("interface_forced"),
                     InterfaceSpecAcouMeca("interface_acou_meca_front"),
                     InterfaceSpecAcouMeca("interface_acou_meca_rear"))
    pml = DomainSpecPML("PML", "Air")
    pb.add_sub_domain(pml)
    lam = c / f0
    pml.size, pml.f_pml, pml.t = lam / 8, f0, lam
    pb.set_mesh_sizes({"coil": 0.0015, "subacou": min(lam / 6, 0.005)})
    pb.mesh(show_mesh_gui=False, reuse_mesh=True)
    return pb


t_all = time.time()
res = {}
if CASE in ("fsi_elec", "fsi"):
    pb = fsi_problem()
    st = FreqStudy(pb)
    t0 = time.time()
    st.assemble_domains()
    t_asm = time.time() - t0
    freqs = np.linspace(1900, 2560, N_FREQ)
    t0 = time.time()
    U, P = [], []
    for f in freqs:
        if CASE == "fsi_elec":
            u, p, I = st.solve_fsi(freq=f, Re=8, Le=4e-3, n=0.8, u=np.sqrt(8.0), Bl=7, record=False)
        else:
            u, p, I = st.solve_fsi(freq=f, force=True, record=False)
        U.append(u), P.append(p)
    t_sol = time.time() - t0
    res = dict(freqs=freqs, U=np.array(U), P=np.array(P))
    if hasattr(st, "solver_stats"):
        print("solver stats:", st.solver_stats())
    print(f"{CASE}: n_meca={st.n_dof_meca} n_acou={st.n_dof_acou} | assembly {t_asm:.2f} s | "
          f"{N_FREQ} solves {t_sol:.2f} s ({t_sol / N_FREQ:.3f} s/freq)")
elif CASE == "meca":
    pb = meca_problem("LoudPy_exemples/Frequency_study/Meca/Results/Files/mesh.msh")
    st = FreqStudy(pb)
    st.assemble_meca()
    freqs = np.logspace(1, np.log10(20000), N_FREQ_MECA)
    t0 = time.time()
    U = [st.solve_meca(freq=f, force=1.0, record=False) for f in freqs]
    t_sol = time.time() - t0
    res = dict(freqs=freqs, U=np.array(U))
    if hasattr(st, "solver_stats"):
        print("solver stats:", st.solver_stats())
    print(f"meca: n={st.n_dof_meca} | {N_FREQ_MECA} solves {t_sol:.2f} s ({t_sol / N_FREQ_MECA * 1e3:.1f} ms/freq)")
elif CASE == "eigen":
    pb = meca_problem("LoudPy_exemples/Eigen_study/Meca/Results/Files/mesh.msh")
    st = EigenStudy(pb)
    t0 = time.time()
    fr, ze, modes = st.solve_meca_eigen_multishift(f_min=20.0, f_max=EIG_FMAX, n_modes_per_shift=30)
    t_sol = time.time() - t0
    res = dict(freqs=np.asarray(fr), zeta=np.asarray(ze))
    print(f"eigen: {len(fr)} modes ≤ {EIG_FMAX:.0f} Hz in {t_sol:.2f} s")
elif CASE == "fsi_eigen":
    pb = Problem(geo_path="LoudPy_exemples/Geometries/Loudspeaker/HPNEW-Sketch_front_rear_dtn.geo",
                 msh_path="LoudPy_exemples/Eigen_study/FSI_DtN/Results/Files/mesh.msh",
                 mat_path=mat, subdomains_key="sub")
    pb.add_sub_domain(*[DomainSpecMecaRayleigh(n, material=m, size=0.0005) for n, m in MECA],
                      DomainSpecAcou("subacou", material="Air"))
    pb.add_interface(InterfaceSpecClamped("interface_constrained"), InterfaceSpecForced("interface_forced"),
                     InterfaceSpecAcouMeca("interface_acou_meca_front"),
                     InterfaceSpecAcouMeca("interface_acou_meca_rear"), InterfaceSpecDtN("dtn_boundary"))
    pb.set_mesh_sizes({"coil": 0.0015, "subacou": 0.01})
    pb.mesh(show_mesh_gui=False, reuse_mesh=True)
    st = EigenStudy(pb)
    t0 = time.time()
    out_ = st.solve_fsi_eigen(f_min=10.0, f_max=3200.0, n_modes_per_shift=30, mac_tol=0.95,
                              cover_zeta_max=0.999, max_shifts=12,
                              dtn_center={"dtn_boundary_front": (0.0, 0.032), "dtn_boundary_rear": (0.0, 0.031)},
                              dtn_half=True, dtn_order=18)
    t_sol = time.time() - t0
    r = st._results[-1]
    res = dict(freqs=np.asarray(r["freqs"]), zeta=np.asarray(r["zeta"]))
    print(f"fsi_eigen: {len(res['freqs'])} coupled modes in {t_sol:.2f} s")
elif CASE == "time":
    pb = meca_problem("LoudPy_exemples/Time_study/Multitone/Results/mesh.msh")
    st = TimeStudy(pb)
    st.assemble_init()
    fs = 48000
    t = np.arange(N_STEPS + 1) / fs
    sig = np.sin(2 * np.pi * 200 * t) + 0.5 * np.sin(2 * np.pi * 950 * t + 1.0)
    t0 = time.time()
    st.solve_time_domain_rayleigh_nl(force_amplitude=1.0, force_direction="_z_", force_signal=sig,
                                     dt=1.0 / fs, n_steps=N_STEPS)
    t_sol = time.time() - t0
    r = st._results[-1]
    U = np.asarray(r["fields"]["u_meca"]) if "fields" in r else np.asarray(r.get("U"))
    res = dict(U=U)
    print(f"time: {N_STEPS} steps in {t_sol:.2f} s ({t_sol / N_STEPS * 1e3:.1f} ms/step)")
res["t_solve"] = t_sol
np.savez_compressed(out / f"{CASE}_{TAG}.npz", **res)
print(f"total {time.time() - t_all:.1f} s → {out / f'{CASE}_{TAG}.npz'}")
