"""
Plot_3D_vented.py — 3D scenes of the vented enclosure on a rigid ground (Running_vented.py merged
first): one air domain inside + port + outside, the deformed driver, 1 N on the voice coil.

Each scene is cut by the vertical plane through the driver axis and the port axis (the part on
the camera side removed, dashed: the outline of the box):
  - SPL on the ground (outside and the floor inside the box), on the remaining walls (outside
    and inside), on the port tube and on the cut — inside the box, through the port and in
    front of the box. One colour scale: RANGE dB below the max beyond NEAR_FIELD from the
    driver (the near field saturates); dB re 20 µPa RMS per N;
  - the deformed driver, coloured by |u| (max deformation DEFORMATION × diameter).

What this produces (in Results/Figures/)
-----------------------------------------
  - vented_views_3d.pdf : one page per frequency of FREQS, six views
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

from loudpy.Studies.Cavity_coupling import DriverModes, driver_surface
from enclosure import (eig_path, X0, Y0, Z_F, W, H, WALL_T, back_z, PORT_X, PORT_Y)

files = Path("LoudPy_exemples/Coupling_3D/Driver_in_cavity/Results/Files")
out_dir = Path("LoudPy_exemples/Coupling_3D/Driver_in_cavity/Results/Figures")
FREQS = [50, 70, 100, 300, 1000, 1900]        # [Hz] (nearest computed frequency)
DEFORMATION, NEAR_FIELD, RANGE = 0.10, 0.08, 60.0
CONTOUR_DB = 3.0                              # iso-SPL lines every CONTOUR_DB [dB]
IMG = (1000, 760)

d = np.load(files / "vented_merged.npz", allow_pickle=True)
cp = np.load(files / "coupled.npz", allow_pickle=True)
freqs, n_s = d["freqs"], int(d["n_s"])
q = d["y"][:, :n_s]
drv = DriverModes.from_eigen_h5(eig_path, list(cp["orders"]), f_max=float(cp["freq"].max()) + 1.0)
assert len(drv) == n_s
h5 = h5py.File(files / "vented_merged_field.h5", "r", locking=False)

# cut plane through the driver axis and the port axis; cameras on the removed side (+n)
dvec = np.array([X0 - PORT_X, Y0 - PORT_Y, 0.0])
dvec /= np.linalg.norm(dvec)
nrm = np.array([-dvec[1], dvec[0], 0.0])
if nrm[1] < 0:
    nrm = -nrm
o_cut = np.array([X0, Y0, 0.0])
F0 = np.array([0.10, 0.13, -0.06])
mid = np.array([0.5 * (X0 + PORT_X), 0.5 * (Y0 + PORT_Y), 0.0])
ey, ez = np.array([0, 1.0, 0]), np.array([0, 0, 1.0])
on_plane = lambda x: x - np.dot(x - o_cut, nrm) * nrm       # projection on the cut plane
F_sec = on_plane(F0)
F_det = on_plane(np.array([mid[0], mid[1], 0.0]))
# name: (position, focal point, view up, view angle in degrees or ("parallel", half height in m))
F_sec = F_sec + np.array([0.0, 0.03, 0.0])
# name: (position, focal point, view up, view angle or ("parallel", half height), "cut" / "full" box)
CAMERAS = {
    "three-quarter front, cut side": (F0 + 0.9 * nrm + 0.55 * ey + 0.95 * ez, F0, ey, 24, "cut"),
    "section, square to the cut": (F_sec + 1.5 * nrm, F_sec, ey, ("parallel", 0.27), "cut"),
    "section close-up: port and driver": (F_det + 1.0 * nrm, F_det, ey, ("parallel", 0.12), "cut"),
    "front face (whole box)": (np.array([0.10, 0.18, 1.6]), np.array([0.10, 0.18, 0.0]), ey,
                               ("parallel", 0.30), "full"),
    "ground and section from above": (np.array([0.10, 2.0, -0.06]), np.array([0.10, 0.0, -0.06]), ez,
                                      ("parallel", 0.42), "cut"),
    "three-quarter rear (whole box)": (F0 + np.array([0.95, 0.65, -1.05]), F0, ey, 24, "full"),
}

_SUB_TET = np.array([[0, 4, 6, 7], [4, 1, 5, 9], [6, 5, 2, 8], [7, 9, 8, 3],
                     [4, 8, 5, 6], [4, 8, 6, 7], [4, 8, 7, 9], [4, 8, 9, 5]])
_SUB_TRI = np.array([[0, 3, 5], [3, 1, 4], [5, 4, 2], [3, 4, 5]])


def surf(nodes, tris):
    t3 = tris[:, _SUB_TRI].reshape(-1, 3)
    return pv.PolyData(nodes, np.hstack([np.full((len(t3), 1), 3), t3]).ravel())


def dashed(a, b, dash=0.008):
    pts = np.linspace(a, b, max(2, int(np.linalg.norm(np.subtract(b, a)) / dash)) + 1)
    return pv.merge([pv.Line(pts[i], pts[i + 1]) for i in range(0, len(pts) - 1, 2)])


t_ = WALL_T
_tz = [(0.0, Z_F), (0.0, back_z(0.0) - t_), (H + t_, back_z(H + t_) - t_), (H + t_, Z_F)]
_edges = [((x, y0, z0), (x, y1, z1)) for x in (-t_, W + t_) for (y0, z0), (y1, z1) in zip(_tz, _tz[1:] + _tz[:1])]
_edges += [((-t_, y, z), (W + t_, y, z)) for y, z in _tz]
OUTLINE = pv.merge([dashed(a, b) for a, b in _edges])
meshes = {}


def band_mesh(b):
    if b not in meshes:
        g = h5[f"band{b}"]
        nodes, tets = g["nodes"][()], g["tets"][()]
        grid = pv.UnstructuredGrid({pv.CellType.TETRA: tets[:, _SUB_TET].reshape(-1, 4)}, nodes)
        faces = {k: surf(nodes, g[f"faces/{k}"][()]) for k in ("ground", "walls", "port_walls")
                 if f"faces/{k}" in g and len(g[f"faces/{k}"])}
        corner = np.unique(tets[:, :4])
        far = corner[np.linalg.norm(nodes[corner] - np.array([X0, Y0, Z_F]), axis=1) > NEAR_FIELD]
        meshes[b] = (grid, faces, far)
    return meshes[b]


def keep_mask(pts):
    """Cut plane: inside the box (and the port) everywhere, outside only in front of the box."""
    x, y, z = pts.T
    in_box = (x >= -t_) & (x <= W + t_) & (y <= H + t_) & (z >= back_z(y) - t_)
    return ((z > Z_F) | in_box).astype(float)


def scene(i):
    f, b, r = freqs[i], int(d["band_of"][i]), int(d["row_of"][i])
    grid, faces, far = band_mesh(b)
    p = h5[f"band{b}/p"][r].astype(complex)
    L = 20 * np.log10(np.maximum(np.abs(p) / np.sqrt(2), 1e-12) / 2e-5)
    top = L[far].max()
    grid["p"] = L
    cut = grid.slice(normal=nrm, origin=o_cut)
    cut["keep"] = keep_mask(cut.points)
    cut = cut.clip_scalar(scalars="keep", value=0.5, invert=False)
    parts = []
    for k, s in faces.items():
        s["p"] = L
        parts.append(s if k == "ground" else s.clip(normal=nrm, origin=o_cut))
    parts.append(cut)
    full = list(faces.values())                                        # uncut box, no cut plane
    levels = np.arange(np.ceil((top - RANGE) / CONTOUR_DB) * CONTOUR_DB, top + 1e-9, CONTOUR_DB)

    def iso(ms):
        ls = [ln for ln in (m.contour(isosurfaces=levels, scalars="p") for m in ms) if ln.n_points]
        return pv.merge(ls) if ls else None

    def feat(ms):
        return pv.merge([m.extract_feature_edges(feature_angle=35, boundary_edges=False, manifold_edges=False,
                                                 non_manifold_edges=False) for m in ms])
    walls_cut = [s.clip(normal=nrm, origin=o_cut) for k, s in faces.items() if k != "ground"]
    walls_full = [s for k, s in faces.items() if k != "ground"]
    drv_s, _ = driver_surface(drv, q[i], (X0, Y0, 0.0), (0.0, 2 * np.pi), 97, DEFORMATION)
    return dict(cut=dict(parts=parts, lines=iso(parts), edges=feat(walls_cut)),
                full=dict(parts=full, lines=iso(full), edges=feat(walls_full)), drv=drv_s), top, f


BASE = pv.Disc(center=(0.10, -5e-4, -0.06), inner=0.0, outer=0.45, normal=(0, 1, 0), c_res=90)


def render(scene_, top, cam):
    sc = dict(scene_[cam[4]], drv=scene_["drv"])
    pl = pv.Plotter(off_screen=True, window_size=IMG)
    pl.set_background("white")
    sb = dict(vertical=True, height=0.5, width=0.045, title_font_size=15, label_font_size=13, n_labels=7)
    pl.add_mesh(BASE, color="#d9d9d9", lighting=False)                  # under the wall footprints
    for j, m in enumerate(sc["parts"]):
        pl.add_mesh(m, scalars="p", cmap="turbo", clim=(top - RANGE, top), lighting=False,
                    interpolate_before_map=True, show_scalar_bar=(j == 0),
                    scalar_bar_args=dict(title="SPL [dB]", fmt="%.0f", position_x=0.90, position_y=0.25, **sb))
    if sc["lines"] is not None:
        pl.add_mesh(sc["lines"], color="black", line_width=0.7, opacity=0.45)
    if sc["edges"].n_points:
        pl.add_mesh(sc["edges"], color="#202020", line_width=1.6)
    pl.add_mesh(sc["drv"], scalars="|u|", cmap="viridis", clim=(0, 1), smooth_shading=True, specular=0.3,
                scalar_bar_args=dict(title="driver |u| / max", fmt="%.1f", position_x=0.02, position_y=0.25, **sb))
    if cam[4] == "cut":
        pl.add_mesh(OUTLINE, color="black", line_width=1.0)
    pos, foc, up, view = cam[:4]
    pl.camera_position = [tuple(pos), tuple(foc), tuple(up)]
    if isinstance(view, tuple):
        pl.enable_parallel_projection()
        pl.camera.parallel_scale = view[1]
    else:
        pl.camera.view_angle = view
    img = pl.screenshot(return_img=True)
    pl.close()
    return img


pages = []
for fq in FREQS:
    i = int(np.argmin(np.abs(freqs - fq)))
    if abs(freqs[i] - fq) > 15.0:                 # not computed (yet)
        continue
    sc, top, f = scene(i)
    fig, axs = plt.subplots(2, 3, figsize=(20, 10.5))
    for a in axs.ravel():
        a.axis("off")
    for a, (name, cam) in zip(axs.ravel(), CAMERAS.items()):
        a.imshow(render(sc, top, cam))
        a.set_title(name, fontsize=10)
    E = np.array([np.sum(np.abs(q[i, drv.m == m]) ** 2) for m in cp["orders"]])
    share = "  ".join(f"m{m} {100 * e / E.sum():.0f}%" for m, e in zip(cp["orders"], E))
    fig.suptitle(f"Vented HPNEW box on a rigid ground — {f:.0f} Hz (port tuning ≈ {float(d['f_b_estimate']):.0f} Hz), "
                 f"1 N on the coil.  SPL inside, in the port and outside: one air domain, one scale "
                 f"(max {top:.0f} dB beyond {100 * NEAR_FIELD:.0f} cm, {RANGE:.0f} dB range, iso-lines every {CONTOUR_DB:.0f} dB; dB re 20 µPa RMS);\ndeformed driver "
                 f"(colour |u|, max 10 % of the diameter), driver kinetic energy {share}.  Cut through the "
                 f"driver and port axes (dashed: box outline); ground and box rigid.", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    pages.append(fig)
    print(f"  {f:.0f} Hz rendered")

tmp, target = out_dir / "vented_views_3d.pdf.tmp", out_dir / "vented_views_3d.pdf"
with PdfPages(tmp) as pdf:
    for fg in pages:
        pdf.savefig(fg, dpi=110)
        plt.close(fg)
os.replace(tmp, target)
h5.close()
print(f"✓ {target}")
