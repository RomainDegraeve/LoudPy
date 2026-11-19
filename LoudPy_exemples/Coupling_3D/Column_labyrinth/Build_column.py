"""
Build_column.py — geometry and meshes of the three-driver column with a folded transmission line,
on a rigid ground (column.py holds the dimensions).

ONE air domain per frequency band: the inside of the line + the mouth + the exterior half-space
(hemisphere on the ground, exact DtN), the three drivers cut out with their revolved wet
profiles on both faces. Each driver gets its own wet surface, so the coupling (MultiDriver) maps
every surface point to the right driver:

  Physical surfaces: "interface_driver_0" (top), "_1" (middle), "_2" (bottom) — front + rear wet
  surfaces of each driver; "dtn", "ground", "walls".

What this produces (Results/Files/)
------------------------------------
  - column_<f_max>Hz.msh : P2 air mesh of each band
  - column.json          : hemisphere, bands, line length and tuning estimate
Then run Running_column.py.
"""
import json
from pathlib import Path

import numpy as np

from loudpy.Studies.Cavity_coupling import _gmsh, sample_profile, add_driver_cutout, carve_drivers, C_AIR
from loudpy.Studies.Exterior_dtn import HalfSpaceDtN, add_driver_front, mesh_volume
from column import (geo_path, Z_B, Z_F, REAR_CURVES, REAR_START, FRONT_CURVES, FRONT_START, W_IN, H_IN,
                    X0, Y0S, CENTERS, NAMES, add_cabinet_solids, outer_corners, line_length, quarter_wave)

out = Path("LoudPy_exemples/Coupling_3D/Column_labyrinth/Results/Files")
out.mkdir(parents=True, exist_ok=True)
BANDS = [(20.0, 1000.0), (1000.0, 2000.0)]     # [Hz], one mesh each
PTS_PER_WAVELENGTH, H_MAX = 4.0, 0.07          # P2 elements per wavelength, cap [m]
H_DRIVER = 0.004                               # mesh size at the drivers [m]
MARGIN, Z_LOW = 0.05, 0.0

rear = sample_profile(geo_path, REAR_CURVES, REAR_START)
rear.append(np.array([rear[-1][-1], [0.0, rear[-1][-1][1]]]))           # rigid lid at the cone foot
front = sample_profile(geo_path, FRONT_CURVES, FRONT_START)
fpts = np.vstack(front)
r_f = front[0][0][0]
c = outer_corners()
center = np.array([W_IN / 2, 0.0, 0.5 * (c[:, 2].min() + max(c[:, 2].max(), fpts[:, 1].max()))])
drv_pts = np.array([(X0 + r * np.cos(a), y0 + r * np.sin(a), z) for y0 in Y0S for r, z in fpts
                    for a in np.linspace(0, 2 * np.pi, 16)])
R = float(np.linalg.norm(np.vstack([c, drv_pts]) - center, axis=1).max() + MARGIN)
dtn = HalfSpaceDtN(center, R, up=1)
print(f"Column: 3 drivers at y = {np.round(Y0S, 3)} m | line {line_length():.3f} m, quarter-wave "
      f"estimate {quarter_wave():.1f} Hz | hemisphere centre {np.round(center, 3)} m, radius {R:.3f} m")


def classify(t):
    """Surface → physical name. A driver's wet surface lies in the cylinder r ≤ R_f about its axis,
    without the flat faces of the baffle (z = Z_B) and of the cabinet front (z = Z_F)."""
    g = _gmsh()
    xmin, ymin, zmin, xmax, ymax, zmax = g.model.getBoundingBox(2, t)
    if g.model.getType(2, t) == "Sphere":
        return "dtn"
    if ymax < 1e-6:
        return "ground"
    flat = abs(zmax - zmin) < 1e-6 and min(abs(zmax - Z_F), abs(zmax - Z_B)) < 1e-6
    for j, y0 in enumerate(Y0S):
        hw = max(xmax - X0, X0 - xmin, ymax - y0, y0 - ymin)        # bounding-box half-width
        if hw <= r_f * 1.0001 + 1e-6 and not flat:
            return f"interface_driver_{j}"
    return "walls"


def build_air(h_far, path):
    g = _gmsh()
    g.clear()
    g.model.add("column_air")
    occ = g.model.occ
    inner, pans, outer, slot = add_cabinet_solids(occ)
    cuts = [add_driver_cutout(rear, center=cxy) for cxy in CENTERS]
    line_air = [(3, t) for t in carve_drivers(inner, cuts, extra_cuts=pans)]
    sph = occ.addSphere(*center, R)
    half = occ.addBox(center[0] - 2 * R, 0.0, center[2] - 2 * R, 4 * R, 2 * R, 4 * R)
    hemi, _ = occ.intersect([(3, sph)], [(3, half)])
    plugs = [add_driver_front(front, cxy, Z_LOW) for cxy in CENTERS]
    box, _ = occ.cut([(3, outer)], [(3, p[1]) for p in plugs])
    solid, _ = occ.fuse(box, [(3, p[0]) for p in plugs])
    ext, _ = occ.cut(hemi, solid)
    air, _ = occ.fuse(line_air + ext, [(3, slot)])
    occ.synchronize()
    vols = [t for d, t in air if d == 3]
    assert len(vols) == 1, f"{len(vols)} air volumes (the line must connect to the outside)"
    volume = occ.getMass(3, vols[0])
    refine = {f"interface_driver_{j}": (H_DRIVER, 0.004, 0.08) for j in range(len(CENTERS))}
    groups = mesh_volume(path, vols[0], classify, refine, h_far)
    n = len(g.model.mesh.getNodes()[0])
    g.clear()
    missing = [f"interface_driver_{j}" for j in range(len(CENTERS)) if f"interface_driver_{j}" not in groups]
    assert not missing, f"no wet surface for {missing}"
    print(f"  ✓ air mesh h = {h_far * 1e3:.0f} mm: {n} nodes, {volume * 1e3:.1f} L, {groups} → {path}")
    return n


cfg = dict(center=center.tolist(), radius=R, up=1, line_length=line_length(),
           f_quarter_wave=quarter_wave(), names=NAMES, centers=[list(cxy) for cxy in CENTERS], bands=[])
for f_lo, f_hi in BANDS:
    h = min(H_MAX, C_AIR / f_hi / PTS_PER_WAVELENGTH)
    path = out / f"column_{f_hi:.0f}Hz.msh"
    n = build_air(h, path)
    cfg["bands"].append(dict(f_lo=f_lo, f_hi=f_hi, msh=str(path), h=h, nodes=n))
(out / "column.json").write_text(json.dumps(cfg, indent=2))
print(f"✓ {out / 'column.json'}")
