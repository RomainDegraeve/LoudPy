"""
column.py — parameters and geometry of a column loudspeaker: three identical HPNEW drivers
stacked on the front of a narrow cabinet, loaded at the rear by a folded transmission line
(labyrinth), standing on a rigid ground. Imported by the Build / Running / Loading / Plot scripts.

Side view (y up, z towards the listener; full inner width W_IN along x):

        back                     front
      +-------------------------------+
      |  ch3  |  ch2  | ch1  [top]    |   ← closed end of the line, above the drivers
      |   |   |   ^   |  |   [middle] |      the three cones radiate into channel 1
      |   |   |   |   |  |   [bottom] |
      |   v   |   |   |  v            |
      |   |   |   |   |  |            |
     ←mouth   +-------+  +---- gap g1 |   panel 1 hangs from the top,
      |  ch3      ch2  <-- ch1        |   panel 2 stands on the ground (gap g2 at the top)
      +-------------------------------+  ground y = 0

Path: down channel 1 (front, drivers), up channel 2, down channel 3 (back), out of the mouth
through the bottom of the back wall. The channel depth shrinks from the drivers to the mouth.
Walls WALL_T thick (sides, top, back); front wall = the drivers' baffle (Z_B → Z_F); the floor of
the line is the ground. Driver axes: +z through (X0, Y0S[j]).
"""
import numpy as np

import loudpy as lp

geo_path = "LoudPy_exemples/Geometries/Loudspeaker/HPNEW-Sketch.geo"
eig_path = "LoudPy_exemples/Eigen_study/Fourier/Results/Files/eigen_fourier.h5"

# ── driver (HPNEW meridian geometry, curve ids) ────────────────────────────────
Z_B, Z_F = 0.031, 0.032                       # rear baffle plane, front face of the cabinet [m]
REAR_START, FRONT_START, CONE_FOOT = (0.05435, Z_B), (0.05735, Z_F), (0.00633, 0.0052)
SOLIDS = lp.Group("membranne", "coil", "surround", "spider", "former", "glue", "dustcap")
# wet profiles selected on the 2D geometry: the rear air's boundary from the baffle to the cone foot, and
# the interface between the structure and the front air
REAR = lp.Profile(lp.BoundaryOf("subacou_rear"), start=REAR_START, end=CONE_FOOT)
FRONT = lp.Profile(lp.Interface(SOLIDS, "subacou_front"), start=FRONT_START)
ORDERS = [0, 1, 2, 3]                         # azimuthal orders of the driver modes
DRV_F_MAX = 6000.0                            # driver modes kept [Hz]
NAMES = ["top", "middle", "bottom"]

# ── column and line ────────────────────────────────────────────────────────────
W_IN = 0.140                                  # inner width (x) = line width [m]
H_IN = 0.660                                  # inner height [m]
DEPTHS = [0.080, 0.060, 0.045]                # channel depths d1 (drivers) … d3 (mouth) [m]
GAP_BOTTOM = 0.070                            # opening under panel 1 (ch1 → ch2) [m]
GAP_TOP = 0.060                               # opening above panel 2 (ch2 → ch3) [m]
MOUTH_H = 0.050                               # mouth: slot through the back wall, y ∈ [0, MOUTH_H] [m]
PANEL_T = 0.012                               # internal panel thickness [m]
WALL_T = 0.015                                # outer wall thickness [m]
DRIVER_TOP = 0.080                            # top driver axis below the inner top [m]
DRIVER_PITCH = 0.130                          # vertical spacing of the driver axes [m]

D_IN = sum(DEPTHS) + 2 * PANEL_T              # inner depth (z) [m]
Z_BACK = Z_B - D_IN                           # inner back wall
X0 = W_IN / 2                                 # driver axes (x)
Y0S = [H_IN - DRIVER_TOP - k * DRIVER_PITCH for k in range(3)]     # driver axes (y), top first
CENTERS = [(X0, y) for y in Y0S]


def channels():
    """(z_back, z_front) of channels 1 (front) … 3 (back)."""
    out, z = [], Z_B
    for d in DEPTHS:
        out.append((z - d, z))
        z -= d + PANEL_T
    return out


def panels():
    """Internal panels as (z_back, z_front, y_bottom, y_top)."""
    (z1, _), (z2, _), _ = channels()
    return [(z1 - PANEL_T, z1, GAP_BOTTOM, H_IN),           # hangs from the top
            (z2 - PANEL_T, z2, 0.0, H_IN - GAP_TOP)]        # stands on the ground


def line_length():
    """Centre-line length from the closed top of channel 1 to the mouth [m] (geometric)."""
    d1, d2, d3 = DEPTHS
    return ((H_IN - GAP_BOTTOM / 2) + (d1 / 2 + PANEL_T + d2 / 2)
            + (H_IN - GAP_TOP / 2 - GAP_BOTTOM / 2) + (d2 / 2 + PANEL_T + d3 / 2)
            + (H_IN - GAP_TOP / 2 - MOUTH_H / 2) + d3 / 2)


def quarter_wave(c=None, end_correction=True):
    """Quarter-wave estimate of the tuning [Hz] (mouth end correction ≈ 0.6 √(A/π))."""
    c = lp.materials["Air"].c if c is None else c
    L = line_length()
    if end_correction:
        L += 0.6 * np.sqrt(W_IN * MOUTH_H / np.pi)
    return c / (4 * L)


def add_cabinet_solids(occ):
    """Inner air box, the internal panels, the outer cabinet and the mouth slot (OCC volume tags)."""
    inner = occ.addBox(0.0, 0.0, Z_BACK, W_IN, H_IN, D_IN)
    pans = [occ.addBox(0.0, y0, z0, W_IN, y1 - y0, z1 - z0) for z0, z1, y0, y1 in panels()]
    outer = occ.addBox(-WALL_T, 0.0, Z_BACK - WALL_T, W_IN + 2 * WALL_T, H_IN + WALL_T,
                       Z_F - Z_BACK + WALL_T)
    slot = occ.addBox(0.0, 0.0, Z_BACK - WALL_T, W_IN, MOUTH_H, WALL_T)
    return inner, pans, outer, slot


def cabinet(occ):
    """lp.Scene cabinet builder: inside air, panels removed, outer cabinet, mouth slot."""
    import loudpy as lp
    inner, pans, outer, slot = add_cabinet_solids(occ)
    return lp.Cabinet(inner=inner, cuts=pans, outer=outer, openings=[slot])


def driver():
    """The HPNEW driver (its azimuthal-order modes, ../../Eigen_study/Fourier) and its wet profiles."""
    import loudpy as lp
    return lp.Driver.from_modes(eig_path, ORDERS, f_max=DRV_F_MAX, geometry=geo_path,
                                rear=REAR,
                                front=FRONT)
