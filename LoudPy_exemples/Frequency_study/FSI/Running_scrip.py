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
out_path = "LoudPy_exemples/Frequency_study/FSI/Results/Files/"

# ── Simulation parameters ──────────────────────────────────────────────────────
c     = 344.0   # speed of sound in air [m/s]
force = 0.1     # applied force amplitude [N]

# Integer-valued frequencies avoid duplicate FFT bins when comparing with
# time-domain results (where bin spacing is 1/T_block).
f_array = np.unique(np.logspace(np.log10(20), np.log10(20000), 320).astype(int))

# Number of consecutive frequencies that share the same mesh.
# Smaller values keep the mesh better-resolved but increase setup overhead.
REMESH_EVERY = 30

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
study = None

for k, f in enumerate(f_array):

    # ── Remesh + reassemble every REMESH_EVERY steps ──────────────────────────
    if k % REMESH_EVERY == 0:
        lam       = c / f             # wavelength at the current frequency [m]
        pml.size  = lam / 8           # PML element size (λ/8)
        pml.f_pml = f                 # centre frequency for PML attenuation
        pml.t     = lam               # PML thickness (one wavelength)

        # Acoustic element size: λ/6 gives 6 elements per wavelength.
        # The cap at 0.1 m prevents over-refinement at very low frequencies.
        problem.set_mesh_sizes({"coil": 0.0015, "subacou": min(lam / 6, 0.1)})
        problem.mesh(show_mesh_gui=False)

        study = FreqStudy(problem)
        study.assemble_domains()
        print(f"  → remeshed at k={k},  f={f:.1f} Hz  (λ = {lam*100:.1f} cm)")

    # ── Solve coupled system at this frequency ────────────────────────────────
    study.solve_fsi(freq=f, force=force)

    # ── Flush block to disk when the mesh block is complete ───────────────────
    is_last_in_block = (k + 1) % REMESH_EVERY == 0 or k == len(f_array) - 1
    if is_last_in_block:
        block_start = (k // REMESH_EVERY) * REMESH_EVERY
        f_start     = f_array[block_start]
        lam         = c / f
        fpath       = (out_path
                       + f"snap_{block_start:04d}_{k:04d}"
                       + f"_f{f_start:.0f}-{f:.0f}Hz.h5")
        study.save(fpath, case="fsi_sweep", index=k, lam=lam)
        print(f"[{k+1}/{len(f_array)}]  saved block [{block_start}→{k}] → {fpath}")
    else:
        print(f"[{k+1}/{len(f_array)}]  f = {f:8.2f} Hz")
