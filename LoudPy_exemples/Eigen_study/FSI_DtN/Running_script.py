"""
Running_script.py — Coupled fluid-structure modes of the loudspeaker radiating
in free space, with the EXACT radiation condition (DtN).

Same loudspeaker, materials, interfaces and 1 mm rigid baffle as the forced FSI
example, with FRONT and REAR air (HPNEW-Sketch_front_rear_dtn.geo). Each
half-space is closed by a hemisphere centred on its own baffle face — front
(0, 0.032), rear (0, 0.031), radii R_front / R_rear in the .geo — carrying the
exact Dirichlet-to-Neumann condition for outgoing waves (loudpy.assembly.dtn):

    ∂p/∂r = Σ_n D_n(s) p_n P_n(cos θ),   D_n(s) = (s/c) k_n'(sR/c) / k_n(sR/c)

truncated at order N (even orders only: rigid baffle). D_n is exactly rational
in s, so each order adds n auxiliary unknowns per arc and the problem stays a
quadratic eigenvalue problem — no absorbing layer.

loudpy reads the orientation from the names: "front" → +1, "rear" → −1 for the
acoustic sub-domains (subacou_front / subacou_rear), the fluid-structure
interfaces (interface_acou_meca_front / _rear) and the DtN arcs
(dtn_boundary_front / _rear); "_r_", "_z_", "_rz_" give the directions of the
clamped and forced interfaces.

The modes are found by a gap-free covering of the s-plane (ARPACK
shift-invert), up to ζ = COVER_ZETA_MAX: the heavily damped radiation modes are
needed by the modal sum of the pressure. The covering stops early once the
modal sum matches the exact forced pressure at the GOAL receivers.

Two families of modes come out:
  - air-loaded structural modes (each one close to an in-vacuo mode);
  - radiation ("air") modes: they depend on the DtN radius and only their sum
    has a physical meaning.

What this produces
------------------
    Results/Files/fsi_eigen_study_dtn.h5   — frequencies, ζ, s, u and p mode shapes

The pressure is then rebuilt from these modes by the separate modal-pressure
scripts (fsi_modal_pressure).
"""
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np

from loudpy.Studies import Problem, EigenStudy
from loudpy import (
    DomainSpecMecaRayleigh,
    DomainSpecAcou,
    InterfaceSpecClamped,
    InterfaceSpecForced,
    InterfaceSpecAcouMeca,
    InterfaceSpecDtN,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
geo_path = "LoudPy_exemples/Geometries/Loudspeaker/HPNEW-Sketch_front_rear_dtn.geo"
msh_path = "LoudPy_exemples/Eigen_study/FSI_DtN/Results/Files/mesh.msh"   # own mesh, reused on reruns
mat_path = "src/loudpy/Materials_Bank/materials.json"
out_path = "LoudPy_exemples/Eigen_study/FSI_DtN/Results/Files/fsi_eigen_study_dtn.h5"

# ── Simulation parameters ──────────────────────────────────────────────────────
F_MAX     = 9000.0         # top of the band where the pressure is used [Hz]
AIR_SIZE  = 0.01           # air element size [m] (as all the runs so far)

# DtN boundaries: one hemisphere per half-space (radii set in the .geo, 0.2 m)
DTN_CENTER = {"dtn_boundary_front": (0.0, 0.032),   # on the front baffle face
              "dtn_boundary_rear":  (0.0, 0.031)}   # on the rear baffle face
DTN_HALF   = True          # rigid baffle → even orders only
DTN_ORDER  = 44            # rule of thumb ⌈k(F_MAX)·R⌉ + 10 = 44 at 9 kHz, R = 0.2 m

# Mode search
F_MODES_MIN       = 10.0
F_MODES_MAX       = 1.6 * F_MAX   # search above the band: the sum needs those modes too
COVER_ZETA_MAX    = 0.999         # keep ≈ 0.999: heavily damped radiation modes are needed
n_modes_per_shift = 30
MAX_SHIFTS        = 120           # safety cap: results are saved even if the goal is not reached
CHECKPOINT_EVERY  = 3             # the modes found so far are saved to out_path every 3 shifts

# Goal-oriented stop: front receivers INSIDE the front sphere (≳ 4 cm from the
# DtN arc, where the modal sum is accurate), on an arc centred on the baffle.
R_GOAL = 0.10
GOAL = dict(
    receivers=[(R_GOAL * np.sin(t), 0.032 + R_GOAL * np.cos(t))
               for t in np.radians([0, 10, 30, 45, 60])],
    freqs=np.geomspace(50, F_MAX, 30),       # the whole band where the pressure is used
    tol_db=0.5,
)

# ── Problem definition ─────────────────────────────────────────────────────────
problem = Problem(geo_path=geo_path, msh_path=msh_path,
                  mat_path=mat_path, subdomains_key="sub")

problem.add_sub_domain(
    DomainSpecMecaRayleigh("membranne",  material="Paper",         size=0.0005),
    DomainSpecMecaRayleigh("coil",       material="Copper",        size=0.0005),
    DomainSpecMecaRayleigh("surround",   material="Rubber",        size=0.0005),
    DomainSpecMecaRayleigh("spider",     material="PhenolicCloth", size=0.0005),
    DomainSpecMecaRayleigh("former",     material="Kapton",        size=0.0005),
    DomainSpecMecaRayleigh("glue",       material="SolidGlue",     size=0.0005),
    DomainSpecMecaRayleigh("dustcap",    material="Polypropylene", size=0.0005),
    DomainSpecAcou("subacou", material="Air"),        # subacou_front and subacou_rear
)

problem.add_interface(
    InterfaceSpecClamped("interface_constrained"),
    InterfaceSpecForced("interface_forced"),          # used by the goal check
    InterfaceSpecAcouMeca("interface_acou_meca_front"),
    InterfaceSpecAcouMeca("interface_acou_meca_rear"),
    InterfaceSpecDtN("dtn_boundary"),                 # dtn_boundary_front and _rear
)

# ── Meshing ────────────────────────────────────────────────────────────────────
problem.set_mesh_sizes({"coil": 0.0015, "subacou": AIR_SIZE})
problem.mesh(show_mesh_gui=True, reuse_mesh=True)   # loads msh_path if built from the same geo/specs

# ── Coupled modal solve ────────────────────────────────────────────────────────
study = EigenStudy(problem)
study.solve_fsi_eigen(
    f_min=F_MODES_MIN,
    f_max=F_MODES_MAX,
    n_modes_per_shift=n_modes_per_shift,
    mac_tol=0.95,            
    cover_zeta_max=COVER_ZETA_MAX,
    max_shifts=MAX_SHIFTS,
    dtn_center=DTN_CENTER,
    dtn_half=DTN_HALF,
    dtn_order=DTN_ORDER,
    cover_goal=GOAL,
    checkpoint_path=out_path,               # usable results even if the run is stopped
    checkpoint_every=CHECKPOINT_EVERY,
)
study.save(out_path, case="fsi_eigen_dtn")
print(f"Saved → {out_path}")
