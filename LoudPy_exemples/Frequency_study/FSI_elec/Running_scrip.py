"""
Running_script.py — Coupled fluid-structure (FSI) frequency sweep.

Why the mesh changes during the sweep
--------------------------------------
In acoustics the mesh must resolve at least 6 elements per wavelength
(λ = c/f).  Since λ shrinks by a factor of 1000 from 20 Hz to 20 kHz, a
single mesh fine enough at 20 kHz would be prohibitively large at 20 Hz —
and a mesh coarse enough at 20 Hz would be inaccurate at 20 kHz.

The solution is to remesh every REMESH_EVERY frequency steps, adapting the
acoustic element size and PML parameters to the current wavelength.  After
each remesh the matrices are re-assembled and the results accumulated so far
are flushed to a new HDF5 block file.

What this produces
------------------
    Results/Files/snap_0000_0029_f20-...Hz.h5
    Results/Files/snap_0030_0059_f...-...Hz.h5
    ...   (one file per mesh block)

Typical use
-----------
Run once.  Then use Loading_script.py to visualise all blocks together.
"""
import os
import numpy as np

# Avoids an OpenMP conflict that arises when scipy (OpenBLAS/MKL) and
# numba both load their own OpenMP runtime simultaneously.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from loudpy.Studies import Problem, FreqStudy
from loudpy import (
    DomainSpecMecaRayleigh,
    DomainSpecAcou,
    DomainSpecPML,
    InterfaceSpecClamped,
    InterfaceSpecForced,
    InterfaceSpecAcouMeca,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
geo_path = "LoudPy_exemples/Geometries/Loudspeaker/HPNEW-Sketch.geo"
msh_path = "LoudPy_exemples/Geometries/Loudspeaker/HPNEW-Sketch.msh"
mat_path = "src/loudpy/Materials_Bank/materials.json"
out_path = "LoudPy_exemples/Frequency_study/FSI_elec/Results/Files/"

# ── Simulation parameters ──────────────────────────────────────────────────────
c     = 344.0   # speed of sound in air [m/s]
force = 0.1     # applied force amplitude [N]

# Integer-valued frequencies avoid duplicate FFT bins when comparing with
# time-domain results (where bin spacing is 1/T_block).
f_array = np.concatenate([
    np.array([20, 30, 40, 50, 60, 70, 80]),
    np.linspace(82, 150, 80),
    np.linspace(151, 300, 60),
    np.linspace(301, 1500, 60),
    np.linspace(1501, 20000, 140),
])
Bl = 7         # Force factor [T/m]
n = 0.8        # [-]
Re = 8         # Resitor [Ohm]
Le = 4e-3      # Inductor [H]
P     = 1.0    # Power [Watt]
u = np.sqrt(P * Re)   # Voltage [V]



# ── Fréquences (Hz) auxquelles on veut (re)mailler ──────────────────────────
# Un remesh se déclenche dès qu'on franchit un de ces seuils.
REMESH_AT = [20, 40, 80, 160, 320, 640, 1280, 1810, 2560, 3620, 5120, 7240, 10240, 15000]

def block_index(f, thresholds):
    """Indice du bloc de mesh pour la fréquence f (nb de seuils franchis)."""
    return int(np.searchsorted(thresholds, f, side="right"))

# ── Problem definition ─────────────────────────────────────────────────────────
problem = Problem(geo_path=geo_path, msh_path=msh_path,
                  mat_path=mat_path, subdomains_key="sub")

# Structural sub-domains
problem.add_sub_domain(
    DomainSpecMecaRayleigh("membranne",  material="Paper",         size=0.0005),
    DomainSpecMecaRayleigh("coil",       material="Copper",        size=0.0005),
    DomainSpecMecaRayleigh("surround",   material="Rubber",        size=0.0005),
    DomainSpecMecaRayleigh("spider",     material="PhenolicCloth", size=0.0005),
    DomainSpecMecaRayleigh("former",     material="Kapton",        size=0.0005),
    DomainSpecMecaRayleigh("glue",       material="SolidGlue",     size=0.0005),
    DomainSpecMecaRayleigh("dustcap",    material="Polypropylene", size=0.0005),
    # Acoustic fluid domain (interior air cavity)
    DomainSpecAcou("subacou", "Air"),
)

problem.add_interface(
    InterfaceSpecClamped("interface_constrained"),
    InterfaceSpecForced("interface_forced"),
    InterfaceSpecAcouMeca("interface_acou_meca_front"),
    InterfaceSpecAcouMeca("interface_acou_meca_rear"),
)

# The PML (Perfectly Matched Layer) absorbs outgoing acoustic waves and
# eliminates spurious reflections from the outer mesh boundary.
# Its thickness (t) and attenuation frequency (f_pml) are updated at each
# remesh to match the current wavelength.
pml = DomainSpecPML("PML", "Air")
problem.add_sub_domain(pml)

# ── FSI frequency sweep ────────────────────────────────────────────────────────
study        = None
prev_block   = None   # bloc de la fréquence précédente
block_start  = 0      # indice k du début du bloc courant

for k, f in enumerate(f_array):
    cur_block = block_index(f, REMESH_AT)

    # ── Remesh quand on entre dans un nouveau bloc ────────────────────────
    if cur_block != prev_block:
        # flush du bloc précédent avant de remailler
        if study is not None:
            f_start = f_array[block_start]
            f_end   = f_array[k - 1]
            lam     = c / f_end
            fpath   = (out_path
                       + f"snap_{block_start:04d}_{k-1:04d}"
                       + f"_f{f_start:.0f}-{f_end:.0f}Hz.h5")
            study.save(fpath, case="fsi_sweep", index=k-1, lam=lam)
            print(f"  saved block [{block_start}→{k-1}] → {fpath}")

        lam       = c / f
        pml.size  = lam / 8
        pml.f_pml = f
        pml.t     = lam
        problem.set_mesh_sizes({"coil": 0.0015, "subacou": min(lam / 6, 0.005)})
        problem.mesh(show_mesh_gui=False)

        study = FreqStudy(problem)
        study.assemble_domains()
        print(f"  → remeshed at k={k}, f={f:.1f} Hz "
              f"(block {cur_block}, λ = {lam*100:.1f} cm)")

        block_start = k
        prev_block  = cur_block

    # ── Solve ─────────────────────────────────────────────────────────────
    study.solve_fsi(freq=f, Re=Re, Le=Le, n=n, u=u, Bl=Bl)
    print(f"[{k+1}/{len(f_array)}]  f = {f:8.2f} Hz  (block {cur_block})")

# ── Flush du dernier bloc ───────────────────────────────────────────────────
if study is not None:
    f_start = f_array[block_start]
    f_end   = f_array[-1]
    lam     = c / f_end
    k       = len(f_array) - 1
    fpath   = (out_path
               + f"snap_{block_start:04d}_{k:04d}"
               + f"_f{f_start:.0f}-{f_end:.0f}Hz.h5")
    study.save(fpath, case="fsi_sweep", index=k, lam=lam)
    print(f"  saved final block [{block_start}→{k}] → {fpath}")