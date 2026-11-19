"""
enclosure.py — shared definition of the demo enclosure, its driver and the ground
(imported by Build_cavity.py, Build_exterior.py and the running / plotting scripts).

Frame: driver axis along z (front +z), y vertical (the box stands on the ground y = 0),
x across. Rear baffle plane z = Z_B (inside), front face of the box z = Z_F.
"""
import numpy as np

geo_path = "LoudPy_exemples/Geometries/Loudspeaker/HPNEW-Sketch.geo"
eig_path = "LoudPy_exemples/Eigen_study/Fourier/Results/Files/eigen_fourier.h5"

# ── driver (HPNEW meridian geometry, curve ids) ────────────────────────────────
Z_B, Z_F = 0.031, 0.032                       # rear baffle plane, front face of the box [m]
REAR_CURVES, REAR_START = [14, 7, 8, 33, 32], (0.05435, Z_B)        # wet rear profile
FRONT_CURVES, FRONT_START = [5, 4, 3, 2, 1, 113, 22, 21, 20], (0.05735, Z_F)  # wet front profile
COIL_CURVE = 37

# ── enclosure ──────────────────────────────────────────────────────────────────
W, H = 0.20, 0.30                             # width (x), height (y) [m]
D_BOTTOM, D_TOP = 0.18, 0.11                  # inner depth at y = 0 and y = H [m]
X0, Y0 = 0.125, 0.190                         # driver axis (off-centre in x and y) [m]
BRACE_Y, BRACE_T, BRACE_DEPTH = 0.08, 0.015, 0.08


def back_z(y):
    """z of the (slanted) back wall at height y."""
    return Z_B - D_BOTTOM + (D_BOTTOM - D_TOP) * y / H


def add_box(occ, z_front):
    """Trapezoidal box (slanted back) from the back wall to the plane z = z_front: volume tag."""
    pts = [(0.0, z_front), (0.0, back_z(0.0)), (H, back_z(H)), (H, z_front)]
    p = [occ.addPoint(0.0, y, z) for y, z in pts]
    loop = occ.addCurveLoop([occ.addLine(p[i], p[(i + 1) % 4]) for i in range(4)])
    return [t for d, t in occ.extrude([(2, occ.addPlaneSurface([loop]))], W, 0, 0) if d == 3][0]


def box_corners():
    return np.array([(x, y, z) for x in (0.0, W) for y in (0.0, H) for z in (Z_F, back_z(y))])


# ── vented version (Vented_*.py) ───────────────────────────────────────────────
WALL_T = 0.015                                # side / top / back wall thickness [m] (front: Z_F − Z_B)
PORT_X, PORT_Y = 0.060, 0.060                 # port axis on the front face [m]
PORT_A, PORT_L = 0.020, 0.062                 # port radius and length (from the front face inwards) [m]
PORT_T = 0.003                                # port tube wall thickness inside the box [m]


def add_outer_box(occ):
    """Outer volume of the vented box (walls WALL_T thick, bottom on the ground y = 0)."""
    t = WALL_T
    pts = [(0.0, Z_F), (0.0, back_z(0.0) - t), (H + t, back_z(H + t) - t), (H + t, Z_F)]
    p = [occ.addPoint(-t, y, z) for y, z in pts]
    loop = occ.addCurveLoop([occ.addLine(p[i], p[(i + 1) % 4]) for i in range(4)])
    return [v for d, v in occ.extrude([(2, occ.addPlaneSurface([loop]))], W + 2 * t, 0, 0) if d == 3][0]


def outer_corners():
    t = WALL_T
    return np.array([(x, y, z) for x in (-t, W + t) for y in (0.0, H + t)
                     for z in (Z_F, back_z(y) - t)])


def port_tuning(volume, c=343.0):
    """Helmholtz estimate f_b = c/2π √(S / (V L')), L' = L + 0.85a (flanged) + 0.6a (unflanged)."""
    S = np.pi * PORT_A ** 2
    return c / (2 * np.pi) * np.sqrt(S / (volume * (PORT_L + 1.45 * PORT_A)))
