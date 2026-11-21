"""
enclosure.py — the demo box of examples 10–13: an 8.4 L box with a slanted back and a brace, standing on the ground
y = 0, the HPNEW driver off-centre on the front; sealed (no wall thickness) or vented (15 mm walls, a port).

Two lp.cabinets.Box and the ready-made scenes lp.Scene.sealed_box / vented_box; the rest is the driver (its modes
from 02_azimuthal_modes, its wet profiles selected on the 2D geometry) and the names the plot scripts use.

Frame: driver axis along z (front +z), y up (the ground y = 0), x across. The driver sets the front: the inside ends
on its rear baffle plane Z_B, the front face of the box is its front plane Z_F.
"""
from pathlib import Path

import loudpy as lp

HERE = Path(__file__).resolve().parent               # shared/
EXAMPLES = HERE.parent                               # LoudPy_exemples/
MESHES = str(HERE / "meshes")                        # the box meshes and cavity modes, shared by examples 10–13

geo_path = str(EXAMPLES / "Geometries/Loudspeaker/HPNEW-Sketch.geo")
eig_path = str(EXAMPLES / "02_azimuthal_modes/Results/Files/eigen_fourier.h5")

# ── the driver: its modes (m = 0 … 3) and its wet profiles, selected on the 2D geometry ─────────────────────────
Z_B, Z_F = 0.031, 0.032                       # rear baffle plane, front face of the box [m] (the profiles' planes)
SOLIDS = lp.Group("membranne", "coil", "surround", "spider", "former", "glue", "dustcap")
REAR = lp.Profile(lp.BoundaryOf("subacou_rear"), start=(0.05435, Z_B), end=(0.00633, 0.0052))   # baffle → cone foot
FRONT = lp.Profile(lp.Interface(SOLIDS, "subacou_front"), start=(0.05735, Z_F))


def driver():
    """The HPNEW driver (its azimuthal-order modes, ../02_azimuthal_modes) and its wet profiles."""
    return lp.Driver.from_modes(eig_path, [0, 1, 2, 3], f_max=10000.0, geometry=geo_path, rear=REAR, front=FRONT)


# ── the boxes ─────────────────────────────────────────────────────────────────────────────────────────────────
W, H = 0.20, 0.30                             # inner width (x), height (y) [m]
D_BOTTOM, D_TOP = 0.18, 0.11                  # inner depth at the bottom and at the top (slanted back) [m]
X0, Y0 = 0.125, 0.190                         # driver axis (off-centre) [m]
WALL_T = 0.015                                # vented box: side / top / back walls [m]
BRACE = lp.cabinets.Brace(y=0.08, thickness=0.015, depth=0.08)
PORT = lp.cabinets.Port(at=(0.060, 0.060), radius=0.020, length=0.062, wall=0.003)
PORT_X, PORT_Y = PORT.at
PLANES = dict(front=Z_B, front_face=Z_F)      # given here so that volumes need no driver (checked against it)

SEALED = lp.cabinets.Box(W, H, D_BOTTOM, depth_top=D_TOP, braces=[BRACE], **PLANES)
VENTED = lp.cabinets.Box(W, H, D_BOTTOM, depth_top=D_TOP, wall=WALL_T, braces=[BRACE], ports=[PORT], **PLANES)
back_z = SEALED.back_z                        # z of the inner back wall at height y


def vented_volume():
    """Air inside the vented box (brace, port tube and bore removed) [m³]."""
    return lp.cabinets.volume(VENTED)


def port_tuning(volume=None):
    """Helmholtz estimate of the port tuning [Hz]."""
    return PORT.tuning(vented_volume() if volume is None else volume)


# ── the scenes (meshed in shared/meshes, reused while unchanged) ──────────────────────────────────────────────
def sealed_scene(hp=None):
    """The driver in the sealed box, front in vacuo (cavity modes to 6 kHz, loss factor 0.02)."""
    return lp.Scene.sealed_box(SEALED, hp or driver(), at=(X0, Y0), f_max=3000.0).mesh(MESHES)


def exterior_scene(hp=None):
    """The sealed box on a rigid ground: cavity modes (rear) + exterior FE / DtN (front)."""
    return lp.Scene.sealed_box(SEALED, hp or driver(), at=(X0, Y0), f_max=3000.0, radiating=True).mesh(MESHES)


def vented_scene(hp=None):
    """The vented box on a rigid ground: ONE air domain (inside + port + outside)."""
    return lp.Scene.vented_box(VENTED, hp or driver(), at=(X0, Y0), f_max=3000.0,
                               air_file="vented_merged_{f_hi:.0f}Hz.msh").mesh(MESHES)
