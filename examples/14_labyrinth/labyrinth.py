"""
labyrinth.py — parameters and geometry of a folded transmission-line ("labyrinth") cabinet for
the HPNEW driver, standing on a rigid ground (imported by the Running / Loading / Plot scripts).

Side view (y up, z towards the listener; full inner width W_IN along x):

        back                                   front
      +---------------------------------------+
      |  layer 1 (h1)  <--------------  [driver]   ← the cone radiates into the closed end
      |    gap g1    +========================|    panel 1 (on the front wall)
      |  layer 2 (h2)  -------------->        |
      |========================+    gap g2    |    panel 2 (on the back wall)
      |  layer 3 (h3)  <--------------        |
      |    gap g3    +========================|    panel 3 (on the front wall)
      |  layer 4 (h4)  -------------------->  mouth (front, on the ground)
      +---------------------------------------+  ground y = 0

The line section W_IN·h_i shrinks from the driver end to the mouth (taper ≈ h1/h4 ≈ 2.8:1).
Path length ≈ 1.2 m (+ end correction) → quarter-wave tuning ≈ 70 Hz (no stuffing).
Walls WALL_T thick (sides, top, back); front wall = the driver's baffle (Z_B → Z_F); the floor of
the line is the ground. Frame of the driver model: axis z (front +z), y vertical, x across.
"""
from pathlib import Path
import numpy as np

import loudpy as lp

HERE = Path(__file__).resolve().parent               # this example's folder
EXAMPLES = HERE.parent                               # examples/

geo_path = str(EXAMPLES / "Geometries/Loudspeaker/HPNEW-Sketch.geo")
eig_path = str(EXAMPLES / "02_azimuthal_modes/Results/Files/eigen_fourier.h5")

# ── driver (HPNEW meridian geometry, curve ids) ────────────────────────────────
Z_B, Z_F = 0.031, 0.032                       # rear baffle plane, front face of the cabinet [m]
REAR_START, FRONT_START, CONE_FOOT = (0.05435, Z_B), (0.05735, Z_F), (0.00633, 0.0052)
SOLIDS = lp.Group("cone", "coil", "surround", "spider", "former", "glue", "dustcap")
# wet profiles selected on the 2D geometry: the rear air's boundary from the baffle to the cone foot, and
# the interface between the structure and the front air
REAR = lp.Profile(lp.BoundaryOf("air_rear"), start=REAR_START, end=CONE_FOOT)
FRONT = lp.Profile(lp.Interface(SOLIDS, "air_front"), start=FRONT_START)
ORDERS = [0, 1, 2, 3]                         # azimuthal orders of the driver modes
DRV_F_MAX = 10000.0                           # driver modes kept [Hz]

# ── labyrinth ──────────────────────────────────────────────────────────────────
W_IN = 0.16                                   # inner width (x) = line width [m]
D_IN = 0.28                                   # inner depth (z) [m]
LAYERS = [0.140, 0.100, 0.072, 0.050]         # line heights h1 (driver) … h4 (mouth) [m]
GAPS = [0.090, 0.075, 0.060]                  # fold openings g1 (back), g2 (front), g3 (back) [m]
PANEL_T = 0.012                               # internal panel thickness [m]
WALL_T = 0.015                                # outer wall thickness [m]

H_IN = sum(LAYERS) + PANEL_T * (len(LAYERS) - 1)          # inner height [m]
Z_BACK = Z_B - D_IN                                        # inner back wall
X0 = W_IN / 2                                              # driver axis
Y0 = H_IN - LAYERS[0] / 2


def layer_bounds():
    """(y_bottom, y_top) of each layer, from layer 1 (top, driver) to layer 4 (bottom, mouth)."""
    out, y = [], H_IN
    for h in LAYERS:
        out.append((y - h, y))
        y -= h + PANEL_T
    return out


def panels():
    """Internal panels as (y_bottom, z_from, z_to): alternately on the front and back walls."""
    out = []
    for i, ((yb, _), g) in enumerate(zip(layer_bounds()[:-1], GAPS)):
        if i % 2 == 0:                                     # on the front wall, opening at the back
            out.append((yb - PANEL_T, Z_BACK + g, Z_B))
        else:                                              # on the back wall, opening at the front
            out.append((yb - PANEL_T, Z_BACK, Z_B - g))
    return out


def mouth():
    """Mouth: (y_bottom, y_top) of the opening in the front wall (full inner width)."""
    return layer_bounds()[-1]


def line_length():
    """Centre-line length of the folded line, from the driver to the mouth [m] (geometric)."""
    lb = layer_bounds()
    L = (Z_B - 0.0) - (Z_BACK + GAPS[0] / 2)               # layer 1: driver (front) → gap 1
    for i in range(1, len(LAYERS)):
        y_prev, y_cur = np.mean(lb[i - 1]), np.mean(lb[i])
        L += y_prev - y_cur                                # vertical turn
        if i < len(LAYERS) - 1:                            # runs between two gaps
            L += D_IN - GAPS[i - 1] / 2 - GAPS[i] / 2
        else:                                              # last run: gap → mouth
            L += D_IN - GAPS[i - 1] / 2
    return L


def add_cabinet_solids(occ):
    """Inner air box, the internal panels and the outer cabinet (OCC volume tags)."""
    inner = occ.addBox(0.0, 0.0, Z_BACK, W_IN, H_IN, D_IN)
    pans = [occ.addBox(0.0, yb, z0, W_IN, PANEL_T, z1 - z0) for yb, z0, z1 in panels()]
    outer = occ.addBox(-WALL_T, 0.0, Z_BACK - WALL_T, W_IN + 2 * WALL_T, H_IN + WALL_T, Z_F - Z_BACK + WALL_T)
    yb, yt = mouth()
    slot = occ.addBox(0.0, yb, Z_B, W_IN, yt - yb, Z_F - Z_B)        # mouth through the front wall
    return inner, pans, outer, slot


def cabinet(occ):
    """lp.Scene cabinet builder: inside air, panels removed, outer cabinet, mouth slot."""
    import loudpy as lp
    inner, pans, outer, slot = add_cabinet_solids(occ)
    return lp.Cabinet(inner=inner, cuts=pans, outer=outer, openings=[slot])


def driver():
    """The HPNEW driver (its azimuthal-order modes, ../02_azimuthal_modes) and its wet profiles."""
    import loudpy as lp
    return lp.Driver.from_modes(eig_path, ORDERS, f_max=DRV_F_MAX, geometry=geo_path,
                                rear=REAR,
                                front=FRONT)


def export_cabinet(folder, r_hole):
    """Cabinet (walls + panels, driver hole, mouth) and the air of the line as STEP files, for CAD."""
    from loudpy.geometry import gmsh_session
    with gmsh_session() as g:
        g.model.add("cabinet")
        occ = g.model.occ
        inner, pans, outer, slot = add_cabinet_solids(occ)
        hole = occ.addCylinder(X0, Y0, Z_B - 1e-3, 0, 0, Z_F - Z_B + 2e-3, r_hole)
        line_air, _ = occ.cut([(3, inner)], [(3, p) for p in pans])
        occ.cut([(3, outer)], line_air + [(3, slot), (3, hole)], removeTool=False)
        occ.remove(line_air, recursive=True)
        occ.synchronize()
        g.write(str(folder / "labyrinth_cabinet.step"))


def quarter_wave(c=None, end_correction=True):
    """Quarter-wave estimate of the tuning [Hz] (mouth end correction ≈ 0.6 √(A/π))."""
    c = lp.materials["Air"].c if c is None else c
    L = line_length()
    if end_correction:
        L += 0.6 * np.sqrt(W_IN * LAYERS[-1] / np.pi)
    return c / (4 * L)
