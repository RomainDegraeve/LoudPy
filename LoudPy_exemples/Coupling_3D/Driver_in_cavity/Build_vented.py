"""
Build_vented.py — meshes of the vented (bass-reflex) version of the enclosure, standing on a
rigid ground (enclosure.py: walls WALL_T thick, port of radius PORT_A and length PORT_L through
the front face, its tube protruding into the box).

Two models:
  1. merged (reference): ONE air domain — inside of the box + port + exterior half-space, the
     DtN on the hemisphere. The driver is wetted on both faces by the same field.
       Results/Files/vented_merged_<f_max>Hz.msh      (one mesh per frequency band)
  2. hybrid (fast): the inside of the box (modal, port tube excluded, inner mouth "port_inner")
     and the exterior (port outer mouth "port_outer"), joined by a 1D port model.
       Results/Files/vented_cavity.msh, vented_ext_<f_max>Hz.msh
  + Results/Files/vented.json (hemisphere, bands, volumes, port).
"""
import json
from pathlib import Path

import numpy as np

from loudpy.Studies.Cavity_coupling import _gmsh, sample_profile, add_driver_cutout, C_AIR
from loudpy.Studies.Exterior_dtn import HalfSpaceDtN, add_driver_front, mesh_volume
from enclosure import (geo_path, Z_B, Z_F, REAR_CURVES, REAR_START, FRONT_CURVES, FRONT_START, W, X0, Y0,
                       BRACE_Y, BRACE_T, BRACE_DEPTH, back_z, add_box, add_outer_box, outer_corners,
                       PORT_X, PORT_Y, PORT_A, PORT_L, PORT_T, port_tuning)

out = Path("LoudPy_exemples/Coupling_3D/Driver_in_cavity/Results/Files")
BANDS = [(20.0, 1000.0), (1000.0, 2000.0), (2000.0, 3000.0)]
PTS_PER_WAVELENGTH, H_MAX = 4.0, 0.06          # 4 P2 elements / wavelength (fits in 8 GB)
H_DRIVER, H_PORT = 0.003, 0.005               # mesh size at the driver / at the port [m]
H_CAVITY = 0.012                              # hybrid cavity (modes up to 6 kHz) [m]
MARGIN, Z_LOW = 0.05, 0.0

rear = sample_profile(geo_path, REAR_CURVES, REAR_START)
rear.append(np.array([rear[-1][-1], [0.0, rear[-1][-1][1]]]))           # rigid lid at the cone foot
front = sample_profile(geo_path, FRONT_CURVES, FRONT_START)
fpts = np.vstack(front)
c = outer_corners()
center = np.array([W / 2, 0.0, 0.5 * (c[:, 2].min() + max(c[:, 2].max(), fpts[:, 1].max()))])
drv_pts = np.array([(X0 + r * np.cos(a), Y0 + r * np.sin(a), z) for r, z in fpts
                    for a in np.linspace(0, 2 * np.pi, 16)])
R = float(np.linalg.norm(np.vstack([c, drv_pts]) - center, axis=1).max() + MARGIN)
dtn = HalfSpaceDtN(center, R, up=1)
r_f = front[0][0][0]
Z_IN = Z_F - PORT_L                                                       # inner mouth plane


def classify(t):
    g = _gmsh()
    xmin, ymin, zmin, xmax, ymax, zmax = g.model.getBoundingBox(2, t)
    flat_z = abs(zmax - zmin) < 1e-6
    hw = lambda x0, y0: max(xmax - x0, x0 - xmin, ymax - y0, y0 - ymin)
    if g.model.getType(2, t) == "Sphere":
        return "dtn"
    if ymax < 1e-6:
        return "ground"
    if flat_z and hw(PORT_X, PORT_Y) <= PORT_A * 1.001 + 1e-6:
        return "port_outer" if abs(zmax - Z_F) < 1e-6 else "port_inner"
    if hw(PORT_X, PORT_Y) <= PORT_A + PORT_T + 1e-5:
        return "port_walls"
    if hw(X0, Y0) <= r_f * 1.0001 + 1e-6 and not (flat_z and min(abs(zmax - Z_F), abs(zmax - Z_B)) < 1e-6):
        return "interface_driver"
    return "walls"


REFINE = {"interface_driver": (H_DRIVER, 0.004, 0.08), "port_walls": (H_PORT, 0.004, 0.06),
          "port_inner": (H_PORT, 0.004, 0.06), "port_outer": (H_PORT, 0.004, 0.06)}


def inner(occ, cut_port):
    """Inside of the box (rear driver cut-out, brace, port tube wall; port air removed if cut_port)."""
    E, C, _, _ = add_driver_cutout(rear, center=(X0, Y0))
    brace = occ.addBox(0.0, BRACE_Y, back_z(BRACE_Y + BRACE_T) - 0.01, W, BRACE_T, BRACE_DEPTH + 0.01)
    tube_out = occ.addCylinder(PORT_X, PORT_Y, Z_IN, 0, 0, Z_B - Z_IN, PORT_A + PORT_T)
    tube_in = occ.addCylinder(PORT_X, PORT_Y, Z_IN, 0, 0, Z_B - Z_IN, PORT_A)
    wall, _ = occ.cut([(3, tube_out)], [(3, tube_in)])
    tools = [(3, E), (3, brace)] + wall
    if cut_port:
        tools.append((3, occ.addCylinder(PORT_X, PORT_Y, Z_IN, 0, 0, Z_B - Z_IN, PORT_A)))
    fused, _ = occ.fuse([(3, add_box(occ, Z_B))], [(3, C)])
    vol, _ = occ.cut(fused, tools)
    return vol


def exterior(occ, mouth_disc):
    sph = occ.addSphere(*center, R)
    half = occ.addBox(center[0] - 2 * R, 0.0, center[2] - 2 * R, 4 * R, 2 * R, 4 * R)
    hemi, _ = occ.intersect([(3, sph)], [(3, half)])
    rev, cyl, _, _ = add_driver_front(front, (X0, Y0), Z_LOW)
    box, _ = occ.cut([(3, add_outer_box(occ))], [(3, cyl)])
    solid, _ = occ.fuse(box, [(3, rev)])
    air, _ = occ.cut(hemi, solid)
    if mouth_disc:                                   # imprint the outer mouth on the front face
        disc = occ.addDisk(PORT_X, PORT_Y, Z_F, PORT_A, PORT_A)
        air, _ = occ.fragment(air, [(2, disc)])
        air = [d for d in air if d[0] == 3]
    return air


def build(kind, h_far, path):
    g = _gmsh()
    g.clear()
    g.model.add(kind)
    occ = g.model.occ
    if kind == "merged":
        port = occ.addCylinder(PORT_X, PORT_Y, Z_IN, 0, 0, PORT_L, PORT_A)
        vol, _ = occ.fuse(inner(occ, False) + exterior(occ, False), [(3, port)])
    elif kind == "cavity":
        vol = inner(occ, True)
    else:
        vol = exterior(occ, True)
    occ.synchronize()
    vols = [t for d, t in vol if d == 3]
    assert len(vols) == 1, f"{kind}: {len(vols)} volumes"
    volume = occ.getMass(3, vols[0])
    groups = mesh_volume(path, vols[0], classify, REFINE, h_far)
    n = len(g.model.mesh.getNodes()[0])
    print(f"  ✓ {kind}: h = {h_far * 1e3:.0f} mm, {n} nodes, {volume * 1e3:.2f} L, {groups} → {path}")
    g.clear()
    return volume, n


print(f"Hemisphere: centre {np.round(center, 3)} m, radius {R:.3f} m")
cfg = dict(center=center.tolist(), radius=R, up=1, port=dict(x=PORT_X, y=PORT_Y, a=PORT_A, L=PORT_L),
           merged=[], ext=[])
v_cav, _ = build("cavity", H_CAVITY, out / "vented_cavity.msh")
cfg["cavity"] = dict(msh=str(out / "vented_cavity.msh"), volume=v_cav)
for f_lo, f_hi in BANDS:
    h = min(H_MAX, C_AIR / f_hi / PTS_PER_WAVELENGTH)
    for kind in ("merged", "ext"):
        path = out / f"vented_{kind}_{f_hi:.0f}Hz.msh"
        _, n = build(kind, h, path)
        cfg[kind].append(dict(f_lo=f_lo, f_hi=f_hi, msh=str(path), h=h, nodes=n))
cfg["f_b_estimate"] = float(port_tuning(v_cav))
(out / "vented.json").write_text(json.dumps(cfg, indent=2))
print(f"Helmholtz estimate of the tuning: f_b ≈ {cfg['f_b_estimate']:.1f} Hz (box {v_cav * 1e3:.2f} L)")
print(f"✓ {out / 'vented.json'}")
