"""
enclosure.py — shared definition of the demo enclosure, its driver and the ground
(imported by the running / plotting scripts).

Builders for lp.Scene: sealed_cabinet (inside air with the brace, outer box = the inside up to the
front face), vented_cabinet (walls WALL_T thick, port through the front face) and driver().

Frame: driver axis along z (front +z), y vertical (the box stands on the ground y = 0),
x across. Rear baffle plane z = Z_B (inside), front face of the box z = Z_F.
"""
import numpy as np

import loudpy as lp

geo_path = "LoudPy_exemples/Geometries/Loudspeaker/HPNEW-Sketch.geo"
eig_path = "LoudPy_exemples/Eigen_study/Fourier/Results/Files/eigen_fourier.h5"

# ── driver (HPNEW meridian geometry, curve ids) ────────────────────────────────
Z_B, Z_F = 0.031, 0.032                       # rear baffle plane, front face of the box [m]
REAR_START, FRONT_START, CONE_FOOT = (0.05435, Z_B), (0.05735, Z_F), (0.00633, 0.0052)
SOLIDS = lp.Group("membranne", "coil", "surround", "spider", "former", "glue", "dustcap")
# wet profiles selected on the 2D geometry: the rear air's boundary from the baffle to the cone foot, and
# the interface between the structure and the front air
REAR = lp.Profile(lp.BoundaryOf("subacou_rear"), start=REAR_START, end=CONE_FOOT)
FRONT = lp.Profile(lp.Interface(SOLIDS, "subacou_front"), start=FRONT_START)
ORDERS = [0, 1, 2, 3]                         # azimuthal orders of the driver modes
DRV_F_MAX = 10000.0                           # driver modes kept [Hz]

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


def port_tuning(volume, c=None):
    """Helmholtz estimate f_b = c/2π √(S / (V L')), L' = L + 0.85a (flanged) + 0.6a (unflanged)."""
    c = lp.materials["Air"].c if c is None else c
    S = np.pi * PORT_A ** 2
    return c / (2 * np.pi) * np.sqrt(S / (volume * (PORT_L + 1.45 * PORT_A)))


# ── lp.Scene builders ──────────────────────────────────────────────────────────
def add_brace(occ):
    return occ.addBox(0.0, BRACE_Y, back_z(BRACE_Y + BRACE_T) - 0.01, W, BRACE_T, BRACE_DEPTH + 0.01)


def sealed_cabinet(occ):
    """Sealed box: the inside (slanted back, brace) and the box up to its front face (no wall thickness)."""
    import loudpy as lp
    return lp.Cabinet(inner=add_box(occ, Z_B), cuts=[add_brace(occ)], outer=add_box(occ, Z_F))


def vented_cabinet(occ, port_size=0.005):
    """Vented box: walls WALL_T thick, brace, port tube (wall PORT_T inside the box) through the front."""
    import loudpy as lp
    z_in = Z_F - PORT_L
    tube_out = occ.addCylinder(PORT_X, PORT_Y, z_in, 0, 0, Z_B - z_in, PORT_A + PORT_T)
    tube_in = occ.addCylinder(PORT_X, PORT_Y, z_in, 0, 0, Z_B - z_in, PORT_A)
    wall, _ = occ.cut([(3, tube_out)], [(3, tube_in)])
    port = occ.addCylinder(PORT_X, PORT_Y, z_in, 0, 0, PORT_L, PORT_A)
    return lp.Cabinet(inner=add_box(occ, Z_B), cuts=[add_brace(occ)] + [t for d, t in wall if d == 3],
                      outer=add_outer_box(occ), openings=[port], opening_size=port_size)


def driver():
    """The HPNEW driver (its azimuthal-order modes, ../../Eigen_study/Fourier) and its wet profiles."""
    import loudpy as lp
    return lp.Driver.from_modes(eig_path, ORDERS, f_max=DRV_F_MAX, geometry=geo_path,
                                rear=REAR,
                                front=FRONT)


FILES = "LoudPy_exemples/Coupling_3D/Driver_in_cavity/Results/Files"
CAV_F_MAX, ETA_AIR = 6000.0, 0.02             # cavity modes kept [Hz], their loss factor


def sealed_scene(hp=None):
    """The driver in the sealed box, front in vacuo (cavity modes): configured and meshed."""
    import loudpy as lp
    hp = hp or driver()
    scene = lp.Scene(sealed_cabinet, [hp.place("main", at=(X0, Y0))])
    scene.cavity(mesh_size=(0.003, 0.012), f_max=CAV_F_MAX, eta=ETA_AIR, file="cavity.msh")
    scene.mesh(FILES)
    return scene


BANDS = [(20.0, 1000.0), (1000.0, 2000.0), (2000.0, 3000.0)]   # one exterior mesh each [Hz]


def exterior_scene(hp=None):
    """The sealed box on a rigid ground: cavity modes (rear) + exterior FE/DtN (front), meshed."""
    import loudpy as lp
    hp = hp or driver()
    scene = lp.Scene(sealed_cabinet, [hp.place("main", at=(X0, Y0))])
    scene.cavity(mesh_size=(0.003, 0.012), f_max=CAV_F_MAX, eta=ETA_AIR, file="cavity.msh")
    scene.exterior(ground="y", bands=BANDS, points_per_wavelength=4.5, max_size=0.06, driver_size=0.003,
                   margin=0.05)
    scene.mesh(FILES)
    return scene


def vented_scene(hp=None):
    """The vented box on a rigid ground: ONE air domain (inside + port + outside), meshed."""
    import loudpy as lp
    hp = hp or driver()
    scene = lp.Scene(vented_cabinet, [hp.place("main", at=(X0, Y0))])
    scene.merged(ground="y", bands=BANDS, points_per_wavelength=4.0, max_size=0.06, driver_size=0.003,
                 margin=0.05, file="vented_merged_{f_hi:.0f}Hz.msh")
    scene.mesh(FILES)
    return scene


def vented_volume():
    """Air volume inside the vented box (brace, port tube and port air removed) [m³]."""
    from loudpy.geometry import gmsh_session
    with gmsh_session() as g:
        g.model.add("volume")
        occ = g.model.occ
        cab = vented_cabinet(occ)
        vol, _ = occ.cut([(3, cab.inner)], [(3, t) for t in cab.cuts + cab.openings])
        occ.synchronize()
        return sum(occ.getMass(d, t) for d, t in vol if d == 3)
