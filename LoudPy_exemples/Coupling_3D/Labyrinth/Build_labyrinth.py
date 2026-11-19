"""
Build_labyrinth.py — geometry and meshes of the labyrinth (folded transmission line) cabinet on
a rigid ground, for the merged-domain coupled simulation (labyrinth.py holds the dimensions).

  1. cabinet export (for gmsh / CAD):  Results/Files/labyrinth_cabinet.step, .geo_unrolled
     (walls, internal panels, driver hole, mouth) and labyrinth_line.step (the air of the line)
  2. ONE air domain per frequency band: inside of the line + mouth + exterior half-space
     (hemisphere on the ground, DtN), the driver cut out with its revolved wet profiles on both
     faces:  Results/Files/labyrinth_<f_max>Hz.msh  (+ labyrinth.json: hemisphere, bands)
Physical surfaces: "interface_driver" (front + rear wet surfaces), "dtn", "ground", "walls".
"""
import json
from pathlib import Path

import numpy as np

from loudpy.Studies.Cavity_coupling import _gmsh, sample_profile, add_driver_cutout, C_AIR
from loudpy.Studies.Exterior_dtn import HalfSpaceDtN, add_driver_front, mesh_volume
from labyrinth import (geo_path, Z_B, Z_F, REAR_CURVES, REAR_START, FRONT_CURVES, FRONT_START, W_IN, H_IN,
                       X0, Y0, add_cabinet_solids, outer_corners, line_length, quarter_wave)

out = Path("LoudPy_exemples/Coupling_3D/Labyrinth/Results/Files")
out.mkdir(parents=True, exist_ok=True)
BANDS = [(20.0, 1000.0), (1000.0, 2000.0), (2000.0, 3000.0)]
PTS_PER_WAVELENGTH, H_MAX = 4.0, 0.06          # P2 elements per wavelength, cap [m]
H_DRIVER = 0.003                               # mesh size at the driver [m]
MARGIN, Z_LOW = 0.05, 0.0

rear = sample_profile(geo_path, REAR_CURVES, REAR_START)
rear.append(np.array([rear[-1][-1], [0.0, rear[-1][-1][1]]]))           # rigid lid at the cone foot
front = sample_profile(geo_path, FRONT_CURVES, FRONT_START)
fpts = np.vstack(front)
r_f = front[0][0][0]
c = outer_corners()
center = np.array([W_IN / 2, 0.0, 0.5 * (c[:, 2].min() + max(c[:, 2].max(), fpts[:, 1].max()))])
drv_pts = np.array([(X0 + r * np.cos(a), Y0 + r * np.sin(a), z) for r, z in fpts
                    for a in np.linspace(0, 2 * np.pi, 16)])
R = float(np.linalg.norm(np.vstack([c, drv_pts]) - center, axis=1).max() + MARGIN)
dtn = HalfSpaceDtN(center, R, up=1)
print(f"Labyrinth: line {line_length():.3f} m, quarter-wave estimate {quarter_wave():.1f} Hz | "
      f"hemisphere centre {np.round(center, 3)} m, radius {R:.3f} m")


def classify(t):
    g = _gmsh()
    xmin, ymin, zmin, xmax, ymax, zmax = g.model.getBoundingBox(2, t)
    flat_z = abs(zmax - zmin) < 1e-6
    hw = max(xmax - X0, X0 - xmin, ymax - Y0, Y0 - ymin)
    if g.model.getType(2, t) == "Sphere":
        return "dtn"
    if ymax < 1e-6:
        return "ground"
    if hw <= r_f * 1.0001 + 1e-6 and not (flat_z and min(abs(zmax - Z_F), abs(zmax - Z_B)) < 1e-6):
        return "interface_driver"
    return "walls"


def export_cabinet():
    """Cabinet (walls + panels, driver hole, mouth) and the air of the line, for gmsh / CAD."""
    g = _gmsh()
    g.clear()
    g.model.add("cabinet")
    occ = g.model.occ
    inner, pans, outer, slot = add_cabinet_solids(occ)
    hole = occ.addCylinder(X0, Y0, Z_B - 1e-3, 0, 0, Z_F - Z_B + 2e-3, r_f)
    line_air, _ = occ.cut([(3, inner)], [(3, p) for p in pans])
    cab, _ = occ.cut([(3, outer)], line_air + [(3, slot), (3, hole)], removeTool=False)
    occ.synchronize()
    g.write(str(out / "labyrinth_cabinet.step"))
    g.write(str(out / "labyrinth_cabinet.geo_unrolled"))
    vol = sum(occ.getMass(d, t) for d, t in cab)
    air = sum(occ.getMass(d, t) for d, t in line_air)
    g.clear()
    g.model.add("line")
    occ = g.model.occ
    inner, pans, outer, slot = add_cabinet_solids(occ)
    line_air, _ = occ.cut([(3, inner)], [(3, p) for p in pans])
    occ.remove([(3, outer), (3, slot)], recursive=True)
    occ.synchronize()
    g.write(str(out / "labyrinth_line.step"))
    g.clear()
    print(f"  ✓ cabinet {vol * 1e3:.2f} L of wood, line air {air * 1e3:.2f} L → "
          f"labyrinth_cabinet.step / .geo_unrolled, labyrinth_line.step")


def build_air(h_far, path):
    g = _gmsh()
    g.clear()
    g.model.add("labyrinth_air")
    occ = g.model.occ
    inner, pans, outer, slot = add_cabinet_solids(occ)
    E, C, _, _ = add_driver_cutout(rear, center=(X0, Y0))
    fused, _ = occ.fuse([(3, inner)], [(3, C)])
    line_air, _ = occ.cut(fused, [(3, E)] + [(3, p) for p in pans])
    sph = occ.addSphere(*center, R)
    half = occ.addBox(center[0] - 2 * R, 0.0, center[2] - 2 * R, 4 * R, 2 * R, 4 * R)
    hemi, _ = occ.intersect([(3, sph)], [(3, half)])
    rev, cyl, _, _ = add_driver_front(front, (X0, Y0), Z_LOW)
    box, _ = occ.cut([(3, outer)], [(3, cyl)])
    solid, _ = occ.fuse(box, [(3, rev)])
    ext, _ = occ.cut(hemi, solid)
    air, _ = occ.fuse(line_air + ext, [(3, slot)])
    occ.synchronize()
    vols = [t for d, t in air if d == 3]
    assert len(vols) == 1, f"{len(vols)} air volumes (the line must connect to the outside)"
    volume = occ.getMass(3, vols[0])
    groups = mesh_volume(path, vols[0], classify, {"interface_driver": (H_DRIVER, 0.004, 0.08)}, h_far)
    n = len(g.model.mesh.getNodes()[0])
    g.clear()
    print(f"  ✓ air mesh h = {h_far * 1e3:.0f} mm: {n} nodes, {volume * 1e3:.1f} L, {groups} → {path}")
    return n


export_cabinet()
cfg = dict(center=center.tolist(), radius=R, up=1, line_length=line_length(),
           f_quarter_wave=quarter_wave(), bands=[])
for f_lo, f_hi in BANDS:
    h = min(H_MAX, C_AIR / f_hi / PTS_PER_WAVELENGTH)
    path = out / f"labyrinth_{f_hi:.0f}Hz.msh"
    n = build_air(h, path)
    cfg["bands"].append(dict(f_lo=f_lo, f_hi=f_hi, msh=str(path), h=h, nodes=n))
(out / "labyrinth.json").write_text(json.dumps(cfg, indent=2))
print(f"✓ {out / 'labyrinth.json'}")
