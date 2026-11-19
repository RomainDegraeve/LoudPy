"""
Build_cavity.py — demo non-axisymmetric enclosure behind the HPNEW driver (3D gmsh mesh).

Enclosure (enclosure.py; driver frame: axis z, front +z, baffle plane z = Z_B):
  - width W along x, height H along y, depth D_BOTTOM at the bottom and D_TOP at the top
    (slanted back wall);
  - a horizontal brace (shelf) fixed to the back wall, BRACE_DEPTH deep, at height BRACE_Y;
  - the driver off-centre, its axis at (X0, Y0).
The driver is cut out with its wet rear profile (surround clamp, surround roll, cone), revolved
around its axis; the spider, former and coil are left acoustically transparent and the pocket
under the dust cap is closed by a rigid lid at the cone foot.

Writes Results/Files/cavity.msh (P2 tetrahedra, physical groups "cavity", "interface_driver",
"walls"). Then run Running_script.py.
"""
import numpy as np

from loudpy.Studies.Cavity_coupling import _gmsh, sample_profile, add_driver_cutout, mesh_cavity

from enclosure import (geo_path, Z_B, REAR_CURVES, REAR_START, W, H, X0, Y0, BRACE_Y, BRACE_T,
                       BRACE_DEPTH, back_z, add_box)

msh_path = "LoudPy_exemples/Coupling_3D/Driver_in_cavity/Results/Files/cavity.msh"
CONE_FOOT_LID = True                           # close the cone foot to the axis (rigid lid)
H_NEAR, H_FAR = 0.003, 0.012                   # mesh size at the driver / far [m]

segments = sample_profile(geo_path, REAR_CURVES, REAR_START)
if CONE_FOOT_LID:
    foot = segments[-1][-1]
    segments.append(np.array([foot, [0.0, foot[1]]]))

gmsh = _gmsh()
gmsh.clear()
gmsh.model.add("cavity")
occ = gmsh.model.occ
box = add_box(occ, Z_B)
brace = occ.addBox(0.0, BRACE_Y, back_z(BRACE_Y + BRACE_T) - 0.01, W, BRACE_T, BRACE_DEPTH + 0.01)
E, C, r_c, z_b = add_driver_cutout(segments, center=(X0, Y0))
fused, _ = occ.fuse([(3, box)], [(3, C)])
cav, _ = occ.cut(fused, [(3, E), (3, brace)])
occ.synchronize()
vols = [t for d, t in cav if d == 3]
assert len(vols) == 1, vols
driver, walls = mesh_cavity(msh_path, vols[0], (X0, Y0), r_c, z_b, H_NEAR, H_FAR)
n_nodes = len(gmsh.model.mesh.getNodes()[0])
print(f"✓ cavity mesh → {msh_path}: {n_nodes} nodes, wet surfaces {driver}, "
      f"volume {occ.getMass(3, vols[0]) * 1e3:.2f} L")
gmsh.clear()
