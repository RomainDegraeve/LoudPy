"""
Build_exterior.py — exterior air of the enclosure standing on a rigid ground (3D gmsh meshes).

Domain: hemisphere centred on the ground (y = 0) below the box centre, radius = smallest that
contains the box and the driver + MARGIN, minus the box (front face z = Z_F) and the driver
(its wet front profile revolved about its axis). The ground is reflective (natural condition)
and the hemisphere carries the exact half-space DtN (Exterior_dtn).

One mesh per frequency band, sized for its top frequency (P2, ~PTS_PER_WAVELENGTH elements per
wavelength, refined to H_NEAR at the driver):
  Results/Files/exterior_<f_max>Hz.msh   and   exterior.json (centre, radius, bands)
"""
import json
from pathlib import Path

import numpy as np

from loudpy.Studies.Cavity_coupling import _gmsh, sample_profile, C_AIR
from loudpy.Studies.Exterior_dtn import HalfSpaceDtN, add_driver_front, mesh_exterior
from enclosure import geo_path, Z_F, FRONT_CURVES, FRONT_START, W, X0, Y0, add_box, box_corners

out = Path("LoudPy_exemples/Coupling_3D/Driver_in_cavity/Results/Files")
BANDS = [(20.0, 1000.0), (1000.0, 2000.0), (2000.0, 3000.0)]   # [Hz], one mesh each
PTS_PER_WAVELENGTH = 4.5       # P2 elements per wavelength at the top of the band
H_NEAR, H_MAX = 0.003, 0.06    # mesh size at the driver, cap [m]
MARGIN = 0.05                  # hemisphere radius beyond the structure [m]
Z_LOW = 0.0                    # bottom of the driver plug inside the box [m]

segments = sample_profile(geo_path, FRONT_CURVES, FRONT_START)
pts = np.vstack(segments)
c = box_corners()
center = np.array([W / 2, 0.0, 0.5 * (c[:, 2].min() + max(c[:, 2].max(), pts[:, 1].max()))])
drv_pts = np.array([(X0 + r * np.cos(a), Y0 + r * np.sin(a), z) for r, z in pts
                    for a in np.linspace(0, 2 * np.pi, 16)])
R = float(np.linalg.norm(np.vstack([c, drv_pts]) - center, axis=1).max() + MARGIN)
dtn = HalfSpaceDtN(center, R, up=1)
print(f"Hemisphere: centre {np.round(center, 3)} m, radius {R:.3f} m")

meshes = []
for f_lo, f_hi in BANDS:
    h_far = min(H_MAX, C_AIR / f_hi / PTS_PER_WAVELENGTH)
    gmsh = _gmsh()
    gmsh.clear()
    gmsh.model.add("exterior")
    occ = gmsh.model.occ
    sph = occ.addSphere(*center, R)
    half = occ.addBox(center[0] - 2 * R, 0.0, center[2] - 2 * R, 4 * R, 2 * R, 4 * R)
    hemi, _ = occ.intersect([(3, sph)], [(3, half)])
    rev, cyl, r_f, z_f = add_driver_front(segments, (X0, Y0), Z_LOW)
    box, _ = occ.cut([(3, add_box(occ, Z_F))], [(3, cyl)])
    solid, _ = occ.fuse(box, [(3, rev)])
    air, _ = occ.cut(hemi, solid)
    occ.synchronize()
    vols = [t for d, t in air if d == 3]
    assert len(vols) == 1, vols
    msh = out / f"exterior_{f_hi:.0f}Hz.msh"
    groups = mesh_exterior(msh, vols[0], dtn, (X0, Y0), r_f, z_f, H_NEAR, h_far)
    n = len(gmsh.model.mesh.getNodes()[0])
    print(f"  ✓ {f_lo:.0f}–{f_hi:.0f} Hz: h = {h_far * 1e3:.0f} mm, {n} nodes, surfaces {groups} → {msh}")
    meshes.append(dict(f_lo=f_lo, f_hi=f_hi, msh=str(msh), h=h_far, nodes=n))
    gmsh.clear()

(out / "exterior.json").write_text(json.dumps(dict(center=center.tolist(), radius=R, up=1,
                                                   bands=meshes), indent=2))
print(f"✓ {out / 'exterior.json'}")
