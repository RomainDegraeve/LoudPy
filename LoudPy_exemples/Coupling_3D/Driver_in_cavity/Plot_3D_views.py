"""
Plot_3D_views.py — full 3D scenes of the coupled problem from several angles, at a few
frequencies (Running_exterior.py first). 1 N on the voice coil.

Each scene: the box is cut open by the vertical plane through the driver axis (its half x < X0
removed):
  - pressure outside: on the whole ground, on the remaining outer walls of the box and on the
    vertical cut plane in front of the box;
  - pressure inside the enclosure: on its inner walls and on the cut plane;
  - the deformed driver, whole revolution, coloured by |u| (max deformation DEFORMATION × diameter).
Pressure shown as SPL [dB re 20 µPa, RMS, per N], one colour scale inside and outside: RANGE dB
below the larger of the inside maximum and the outside maximum beyond NEAR_FIELD from the driver
(the near field saturates). The floor of the enclosure (also under the removed half) shows the
inside SPL; dashed lines outline the removed half of the box. Driver deformation at the instant of its largest displacement.

What this produces (in Results/Figures/)
-----------------------------------------
  - views_3d.pdf : one page per frequency of FREQS, the scene from the six CAMERAS
"""
import os
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pyvista as pv

from loudpy.Studies.Cavity_coupling import (CavityModes, DriverModes, driver_surface,
                                            C_AIR, RHO_AIR)
from enclosure import eig_path, X0, Y0, Z_F, H, back_z

files = Path("LoudPy_exemples/Coupling_3D/Driver_in_cavity/Results/Files")
out_dir = Path("LoudPy_exemples/Coupling_3D/Driver_in_cavity/Results/Figures")
FREQS = [100, 470, 1000, 1680, 2400]          # [Hz] (nearest computed frequency)
DEFORMATION = 0.10
NEAR_FIELD = 0.08                             # outside colour scale taken beyond this distance [m]
RANGE = 50.0                                  # SPL colour range below the max (inside or outside) [dB]
IMG = (1000, 760)
WALL_OFFSET = 0.002                           # inner walls drawn this far inside the box [m]
FOCUS = (0.10, 0.13, -0.06)
CAMERAS = {                                   # position, view up
    "front left (cut side)": ((-0.80, 0.70, 0.95), (0, 1, 0)),
    "front right (outer walls)": ((1.05, 0.65, 0.95), (0, 1, 0)),
    "side, facing the cut": ((-1.30, 0.35, -0.06), (0, 1, 0)),
    "from above": ((-0.35, 1.50, 0.45), (0, 1, 0)),
    "rear right (back and side walls)": ((1.00, 0.60, -1.05), (0, 1, 0)),
    "close-up on the driver (cut side)": ((-0.30, 0.36, 0.30), (0, 1, 0), (X0, Y0, 0.0), 30),
}

d = np.load(files / "exterior.npz", allow_pickle=True)
cp = np.load(files / "coupled.npz", allow_pickle=True)
freqs, q = d["freqs"], d["q"]
cav = CavityModes.load(files / "cavity_modes.h5")
drv = DriverModes.from_eigen_h5(eig_path, list(d["orders"]), f_max=float(cp["freq"].max()) + 1.0)
assert len(drv) == q.shape[1]
h5 = h5py.File(files / "exterior_field.h5", "r")

# P2 fields drawn on linear sub-elements through the mid-edge nodes (smooth colours):
# TET10 → 8 tetrahedra, TRI6 → 4 triangles (gmsh node order)
_SUB_TET = np.array([[0, 4, 6, 7], [4, 1, 5, 9], [6, 5, 2, 8], [7, 9, 8, 3],
                     [4, 8, 5, 6], [4, 8, 6, 7], [4, 8, 7, 9], [4, 8, 9, 5]])
_SUB_TRI = np.array([[0, 3, 5], [3, 1, 4], [5, 4, 2], [3, 4, 5]])


def p2_grid(nodes, tets):
    return pv.UnstructuredGrid({pv.CellType.TETRA: tets[:, _SUB_TET].reshape(-1, 4)}, nodes)


def surf(nodes, tris):
    t3 = tris[:, _SUB_TRI].reshape(-1, 3)
    return pv.PolyData(nodes, np.hstack([np.full((len(t3), 1), 3), t3]).ravel())


# enclosure: grid (cut) and inner walls (its boundary)
cgrid = p2_grid(cav.nodes, cav.tets)
cwalls = cgrid.extract_surface()
cwalls["node"] = np.asarray(cwalls["vtkOriginalPointIds"]).copy()     # node of the P2 field
cwalls = cwalls.compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True)
cwalls.points = cwalls.points - WALL_OFFSET * cwalls["Normals"]   # inner walls 2 mm inside (zero-
#                                                                    thickness walls: no overlap)
c_corner = np.unique(cav.tets[:, :4])


def dashed(a, b, dash=0.008):
    """Dashed segment a → b (every other piece of length `dash`)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(2, int(np.linalg.norm(b - a) / dash))
    pts = np.linspace(a, b, n + 1)
    segs = [(pts[i], pts[i + 1]) for i in range(0, n, 2)]
    return pv.merge([pv.Line(s0, s1) for s0, s1 in segs])


# outline of the half of the box removed by the cut (x from 0 to X0)
_tz = [(0.0, Z_F), (0.0, back_z(0.0)), (H, back_z(H)), (H, Z_F)]
_edges = [((0.0, y0, z0), (0.0, y1, z1)) for (y0, z0), (y1, z1) in zip(_tz, _tz[1:] + _tz[:1])]
_edges += [((0.0, y, z), (X0, y, z)) for y, z in _tz]
REMOVED_OUTLINE = pv.merge([dashed(a, b) for a, b in _edges])
cut_o = (X0, 0.0, 0.0)
_cc = cwalls.cell_centers().points
floor = cwalls.extract_cells(np.where(_cc[:, 1] < WALL_OFFSET + 0.004)[0]).extract_surface()


def scene(i):
    f, b, r = freqs[i], int(d["band_of"][i]), int(d["row_of"][i])
    w, k = 2 * np.pi * f, 2 * np.pi * f / C_AIR
    g = h5[f"band{b}"]
    nodes, tets = g["nodes"][()], g["tets"][()]
    p_ext = g["p"][r].astype(complex)
    Dc = 1.0 / (cav.kappa2 - k * k + 1j * float(cp["eta_air"]) * np.sqrt(cav.kappa2) * k)
    p_cav = cav.psi @ (-RHO_AIR * w ** 2 * Dc * (cp["S"].T @ q[i]))
    drv_s, ph = driver_surface(drv, q[i], (X0, Y0, 0.0), (0.0, 2 * np.pi), 97, DEFORMATION)
    spl = lambda p: 20 * np.log10(np.maximum(np.abs(p) / np.sqrt(2), 1e-12) / 2e-5)   # RMS, dB re 20 µPa
    pe, pc = spl(p_ext), spl(p_cav)
    corner = np.unique(tets[:, :4])
    far = corner[np.linalg.norm(nodes[corner] - np.array([X0, Y0, Z_F]), axis=1) > NEAR_FIELD]
    e_max, c_max = pe[far].max(), pc[c_corner].max()
    egrid = p2_grid(nodes, tets)
    egrid["p"] = pe
    ground = surf(nodes, g["faces/ground"][()])
    walls = surf(nodes, g["faces/walls"][()])
    for s in (ground, walls):
        s["p"] = pe
    cgrid["p"] = pc
    cwalls["p"] = pc[cwalls["node"]]
    floor["p"] = pc[floor["node"]]
    parts = dict(
        ext=[ground, walls.clip(normal="-x", origin=cut_o),
             egrid.slice(normal="x", origin=cut_o).clip(normal="-z", origin=(0.0, 0.0, Z_F))],
        cav=[cwalls.clip(normal="-x", origin=cut_o), cgrid.slice(normal="x", origin=cut_o), floor],
        drv=drv_s)
    return parts, e_max, c_max, f


def render(parts, e_max, c_max, cam):
    p = pv.Plotter(off_screen=True, window_size=IMG)
    p.set_background("white")
    sb = dict(vertical=True, height=0.45, width=0.05, title_font_size=16, label_font_size=14,
              fmt="%.0f", n_labels=6)
    top = max(e_max, c_max)                                  # one SPL scale inside and outside
    meshes = parts["ext"] + parts["cav"]
    for j, m in enumerate(meshes):
        p.add_mesh(m, scalars="p", cmap="turbo", clim=(top - RANGE, top), lighting=False,
                   interpolate_before_map=True, show_scalar_bar=(j == 0),
                   scalar_bar_args=dict(title="SPL [dB]", position_x=0.88, position_y=0.30, **sb))
    p.add_mesh(parts["drv"], scalars="|u|", cmap="viridis", clim=(0, 1), smooth_shading=True,
               scalar_bar_args={**sb, "fmt": "%.1f", "title": "driver |u| / max", "position_x": 0.03,
                                "position_y": 0.30})
    p.add_mesh(REMOVED_OUTLINE, color="black", line_width=1.2)      # dashed: removed half of the box
    p.camera_position = [cam[0], cam[2] if len(cam) > 2 else FOCUS, cam[1]]
    p.camera.view_angle = cam[3] if len(cam) > 3 else 25
    img = p.screenshot(return_img=True)
    p.close()
    return img


pages = []
for fq in FREQS:
    i = int(np.argmin(np.abs(freqs - fq)))
    parts, e_max, c_max, f = scene(i)
    fig, axs = plt.subplots(2, 3, figsize=(20, 10.5))
    for a in axs.ravel():
        a.axis("off")
    for a, (name, cam) in zip(axs.ravel(), CAMERAS.items()):
        a.imshow(render(parts, e_max, c_max, cam))
        a.set_title(name, fontsize=10)
    E = np.array([np.sum(np.abs(q[i, d["m"] == m]) ** 2) for m in d["orders"]])
    share = "  ".join(f"m{m} {100 * e / E.sum():.0f}%" for m, e in zip(d["orders"], E))
    fig.suptitle(f"HPNEW in its enclosure on a rigid ground — {f:.0f} Hz, 1 N on the coil.  SPL inside "
                 f"(max {c_max:.0f} dB) and outside (max {e_max:.0f} dB beyond {100 * NEAR_FIELD:.0f} cm), same "
                 f"scale, dB re 20 µPa RMS;\ndeformed driver (colour |u|, max deformation 10 % of the diameter), "
                 f"driver kinetic energy {share}.  Box cut open along the driver axis (dashed: removed half); "
                 f"ground and box rigid, DtN hemisphere not drawn.", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    pages.append(fig)
    print(f"  {f:.0f} Hz rendered")

tmp, target = out_dir / "views_3d.pdf.tmp", out_dir / "views_3d.pdf"
with PdfPages(tmp) as pdf:
    for fig in pages:
        pdf.savefig(fig, dpi=110)
        plt.close(fig)
os.replace(tmp, target)
h5.close()
print(f"✓ {target}")
