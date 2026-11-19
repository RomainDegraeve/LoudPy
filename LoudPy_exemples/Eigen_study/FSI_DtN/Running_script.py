"""
Running_script.py — coupled structure + air modes of the loudspeaker radiating in free space,
with the EXACT radiation condition (DtN).

Same loudspeaker, materials, interfaces and 1 mm rigid baffle as the forced FSI example, with
FRONT and REAR air (HPNEW-Sketch_front_rear_dtn.geo). Each half-space is closed by a hemisphere
centred on its own baffle face — front (0, 0.032), rear (0, 0.031), radii in the .geo — carrying
the exact Dirichlet-to-Neumann condition for outgoing waves (loudpy.physics.radiation):

    ∂p/∂r = Σ_n D_n(s) p_n P_n(cos θ),   D_n(s) = (s/c) k_n'(sR/c) / k_n(sR/c)

truncated at order N (even orders only: rigid baffle). D_n is exactly rational in s, so each
order adds n auxiliary unknowns per arc and the problem stays a quadratic eigenvalue problem —
no absorbing layer.

The side of each air region, coupling boundary and DtN arc ("front" → +1, "rear" → −1) comes
from the group names; "_r_", "_z_", "_rz_" give the blocked / forced directions.

The modes are found by a gap-free covering of the s-plane (shift-invert), up to
ζ = COVER_ZETA_MAX: the heavily damped radiation modes are needed by the modal sum of the
pressure. The covering stops early once the modal sum matches the exact forced pressure at the
GOAL receivers. Two families of modes come out:
  - air-loaded structural modes (each one close to an in-vacuo mode, ../Meca);
  - radiation ("air") modes: they depend on the DtN radius and only their sum has a meaning.

What this produces
------------------
    Results/Files/modes.h5               — frequencies, ζ, s, u and p mode shapes (format 2)
    Results/Files/fsi_dtn_checkpoint.h5  — the modes found so far, every CHECKPOINT_EVERY shifts
                                           (usable with lp.load if the run is stopped)
Then run Loading_script.py.
"""
import numpy as np

import loudpy as lp

# ── Paths ──────────────────────────────────────────────────────────────────────
geometry = "LoudPy_exemples/Geometries/Loudspeaker/HPNEW-Sketch_front_rear_dtn.geo"
files = "LoudPy_exemples/Eigen_study/FSI_DtN/Results/Files"
mesh_path = f"{files}/mesh.msh"                 # reused on reruns
out_path = f"{files}/modes.h5"
checkpoint_path = f"{files}/fsi_dtn_checkpoint.h5"

# ── Settings ───────────────────────────────────────────────────────────────────
F_MAX = 9000.0                 # top of the band where the pressure is used [Hz]
AIR_SIZE = 0.01                # air element size [m]
DTN_CENTER = {"dtn_boundary_front": (0.0, 0.032),   # on the front baffle face
              "dtn_boundary_rear": (0.0, 0.031)}    # on the rear baffle face
DTN_HALF = True                # rigid baffle → even orders only
DTN_ORDER = 44                 # rule of thumb ⌈k(F_MAX)·R⌉ + 10 = 44 at 9 kHz, R = 0.2 m
F_MODES_MIN = 10.0
F_MODES_MAX = 1.6 * F_MAX      # search above the band: the sum needs those modes too
COVER_ZETA_MAX = 0.999         # keep ≈ 0.999: heavily damped radiation modes are needed
N_PER_SHIFT = 30
MAX_SHIFTS = 120               # safety cap: results are saved even if the goal is not reached
CHECKPOINT_EVERY = 3           # the modes found so far are saved every 3 shifts
R_GOAL = 0.10
GOAL = dict(
    receivers=[(R_GOAL * np.sin(t), 0.032 + R_GOAL * np.cos(t)) for t in np.radians([0, 10, 30, 45, 60])],
    freqs=np.geomspace(50, F_MAX, 30),      # the whole band where the pressure is used
    tol_db=0.5,
)

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
    lp.Air("subacou", mesh_size=AIR_SIZE),              # subacou_front and subacou_rear
    lp.Clamped("interface_constrained"),
    lp.Coil("interface_forced"),                        # used by the goal check
    lp.FluidStructure("interface_acou_meca_front"),
    lp.FluidStructure("interface_acou_meca_rear"),
    lp.Radiation("dtn_boundary"),                       # dtn_boundary_front and _rear
)
model.mesh(mesh_path)

# ── Coupled modes ──────────────────────────────────────────────────────────────
study = lp.Modes(model, physics="coupled")
modes = study.solve(
    f_min=F_MODES_MIN, f_max=F_MODES_MAX, n_per_shift=N_PER_SHIFT,
    mac_tol=0.95, cover_zeta_max=COVER_ZETA_MAX, max_shifts=MAX_SHIFTS,
    dtn_center=DTN_CENTER, dtn_half=DTN_HALF, dtn_order=DTN_ORDER, cover_goal=GOAL,
    checkpoint_path=checkpoint_path, checkpoint_every=CHECKPOINT_EVERY,
)
print(modes)
study.save(out_path)
print(f"Saved → {out_path}")
