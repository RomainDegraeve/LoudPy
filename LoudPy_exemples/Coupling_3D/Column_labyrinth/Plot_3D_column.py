"""
Plot_3D_column.py — 3D scenes of the three-driver column with a folded transmission line, on a
rigid ground (Running_column.py first): one air domain inside the line + mouth + outside, and
the three deformed drivers.

The vertical plane through the driver axes (x = X0) cuts the column: it shows the three drivers
and the whole folded line in section, the front radiation and the mouth at the back.
  - SPL on the ground, on the remaining walls and panels and on the cut plane. One colour
    scale: RANGE dB below the max farther than NEAR_FIELD from the drivers, iso-lines every
    CONTOUR_DB; dB re 20 µPa RMS;
  - the three drivers at the same instant and the same deformation scale (the largest
    displacement of the three = DEFORMATION × diameter), coloured by |u| re that maximum.

What this produces (in Results/Figures/)
-----------------------------------------
  - column_views_3d.pdf : one page per frequency of FREQS, four views
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

from loudpy.Studies.Cavity_coupling import DriverModes, MultiDriver
from column import (eig_path, X0, Y0S, Z_F, Z_BACK, W_IN as W, H_IN as H, WALL_T, CENTERS, NAMES,
                    ORDERS, DRV_F_MAX)

files = Path("LoudPy_exemples/Coupling_3D/Column_labyrinth/Results/Files")
out_dir = Path("LoudPy_exemples/Coupling_3D/Column_labyrinth/Results/Figures")
out_dir.mkdir(parents=True, exist_ok=True)
FREQS = [40, 70, 120, 250, 500, 1000]         # [Hz] (nearest computed frequency)
DEFORMATION, NEAR_FIELD, RANGE = 0.10, 0.08, 50.0
CONTOUR_DB = 3.0                              # iso-SPL lines every CONTOUR_DB [dB]
IMG = (900, 900)

d = np.load(files / "column.npz", allow_pickle=True)
freqs, n_s = d["freqs"], int(d["n_s"])
q = d["y"][:, :n_s]
drivers = MultiDriver(DriverModes.from_eigen_h5(eig_path, ORDERS, f_max=DRV_F_MAX), CENTERS, names=NAMES)
assert len(drivers) == n_s, "driver modes changed since Running_column.py"
drv = drivers.modes[0]
h5 = h5py.File(files / "column_field.h5", "r", locking=False)

nrm = np.array([-1.0, 0.0, 0.0])                    # cut x = X0, cameras on the −x side
o_cut = np.array([X0, Y0S[1], 0.0])
ey, ez = np.array([0, 1.0, 0]), np.array([0, 0, 1.0])
mid = np.array([X0, 0.36, 0.5 * (Z_F + Z_BACK)])    # middle of the column
# name: (position, focal point, view up, view angle or ("parallel", half height), "cut" / "full")
CAMERAS = {
    "section through the driver axes, square to the cut": (mid + 1.8 * nrm, mid, ey, ("parallel", 0.62), "cut"),
    "three-quarter front, cut side": (mid + 1.3 * nrm + 0.5 * ey + 1.2 * ez, mid, ey, 30, "cut"),
    "front (whole column)": (mid + 2.2 * ez, mid, ey, ("parallel", 0.45), "full"),
    "three-quarter rear: the mouth": (mid + np.array([-1.5, 0.45, -1.9]), mid - [0, 0.05, 0], ey, 30, "full"),
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
_yz = [(0.0, Z_F), (0.0, Z_BACK - t_), (H + t_, Z_BACK - t_), (H + t_, Z_F)]
_edges = [((x, y0, z0), (x, y1, z1)) for x in (-t_, W + t_) for (y0, z0), (y1, z1) in zip(_yz, _yz[1:] + _yz[:1])]
_edges += [((-t_, y, z), (W + t_, y, z)) for y, z in _yz]
OUTLINE = pv.merge([dashed(a, b) for a, b in _edges])
meshes = {}


def band_mesh(b):
    if b not in meshes:
        g = h5[f"band{b}"]
        nodes, tets = g["nodes"][()], g["tets"][()]
        grid = pv.UnstructuredGrid({pv.CellType.TETRA: tets[:, _SUB_TET].reshape(-1, 4)}, nodes)
        faces = {k: surf(nodes, g[f"faces/{k}"][()]) for k in ("ground", "walls")
                 if f"faces/{k}" in g and len(g[f"faces/{k}"])}
        corner = np.unique(tets[:, :4])
        dist = np.min([np.linalg.norm(nodes[corner] - np.array([X0, y0, Z_F]), axis=1) for y0 in Y0S], axis=0)
        meshes[b] = (grid, faces, corner[dist > NEAR_FIELD])
    return meshes[b]


def driver_surfaces(qi):
    """The three deformed drivers at one instant and one scale (that of the largest displacement)."""
    first = [drivers.surface(j, qi, n_theta=97, deformation=DEFORMATION) for j in range(3)]
    j_big = int(np.argmax([s["|u| [m]"].max() for s, _ in first]))
    ph = first[j_big][1]
    u_max = max(drivers.surface(j, qi, n_theta=13, phase=ph)[0]["|u| [m]"].max() for j in range(3))
    return [drivers.surface(j, qi, n_theta=97, deformation=DEFORMATION, phase=ph, u_max=u_max)[0]
            for j in range(3)]


def scene(i):
    f, b, r = freqs[i], int(d["band_of"][i]), int(d["row_of"][i])
    grid, faces, far = band_mesh(b)
    p = h5[f"band{b}/p"][r].astype(complex)
    L = 20 * np.log10(np.maximum(np.abs(p) / np.sqrt(2), 1e-12) / 2e-5)
    top = L[far].max()
    grid["p"] = L
    cut = grid.slice(normal=nrm, origin=o_cut)
    parts = []
    for k, s in faces.items():
        s["p"] = L
        parts.append(s if k == "ground" else s.clip(normal=nrm, origin=o_cut))
    parts.append(cut)
    full = list(faces.values())
    levels = np.arange(np.ceil((top - RANGE) / CONTOUR_DB) * CONTOUR_DB, top + 1e-9, CONTOUR_DB)

    def iso(ms):
        ls = [ln for ln in (m.contour(isosurfaces=levels, scalars="p") for m in ms) if ln.n_points]
        return pv.merge(ls) if ls else None

    def feat(ms):
        return pv.merge([m.extract_feature_edges(feature_angle=35, boundary_edges=False, manifold_edges=False,
                                                 non_manifold_edges=False) for m in ms])
    walls_cut = [s.clip(normal=nrm, origin=o_cut) for k, s in faces.items() if k != "ground"]
    walls_full = [s for k, s in faces.items() if k != "ground"]
    return dict(cut=dict(parts=parts, lines=iso(parts), edges=feat(walls_cut)),
                full=dict(parts=full, lines=iso(full), edges=feat(walls_full)),
                drv=driver_surfaces(q[i])), top, f


BASE = pv.Disc(center=(X0, -5e-4, -0.08), inner=0.0, outer=0.8, normal=(0, 1, 0), c_res=90)


def render(scene_, top, cam):
    sc = scene_[cam[4]]
    pl = pv.Plotter(off_screen=True, window_size=IMG)
    pl.set_background("white")
    sb = dict(vertical=True, height=0.45, width=0.045, title_font_size=15, label_font_size=13, n_labels=6)
    pl.add_mesh(BASE, color="#d9d9d9", lighting=False)
    for j, m in enumerate(sc["parts"]):
        pl.add_mesh(m, scalars="p", cmap="turbo", clim=(top - RANGE, top), lighting=False,
                    interpolate_before_map=True, show_scalar_bar=(j == 0),
                    scalar_bar_args=dict(title="SPL [dB]", fmt="%.0f", position_x=0.88, position_y=0.27, **sb))
    if sc["lines"] is not None:
        pl.add_mesh(sc["lines"], color="black", line_width=0.7, opacity=0.45)
    if sc["edges"].n_points:
        pl.add_mesh(sc["edges"], color="#202020", line_width=1.6)
    for j, s in enumerate(scene_["drv"]):
        pl.add_mesh(s, scalars="|u|", cmap="viridis", clim=(0, 1), smooth_shading=True, specular=0.3,
                    show_scalar_bar=(j == 0),
                    scalar_bar_args=dict(title="|u| / max", fmt="%.1f", position_x=0.05,
                                         position_y=0.27, **sb))
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
    fig, axs = plt.subplots(2, 2, figsize=(15, 15.5))
    for a in axs.ravel():
        a.axis("off")
    for a, (name, cam) in zip(axs.ravel(), CAMERAS.items()):
        a.imshow(render(sc, top, cam))
        a.set_title(name, fontsize=11)
    shares = []
    for name, qj in zip(NAMES, drivers.split(q[i])):
        E = np.abs(qj) ** 2
        shares.append(f"{name} {100 * E[drv.m >= 1].sum() / E.sum():.2g}%")
    fig.suptitle(f"Column, 3 × HPNEW + folded line on a rigid ground — {f:.0f} Hz (line quarter-wave ≈ "
                 f"{float(d['f_quarter_wave']):.0f} Hz), drivers in phase.\nSPL inside the line, at the mouth and "
                 f"outside, one air domain, one scale (max {top:.0f} dB beyond {100 * NEAR_FIELD:.0f} cm of the drivers, "
                 f"{RANGE:.0f} dB range, iso-lines every {CONTOUR_DB:.0f} dB).\nDrivers at one instant and one scale; "
                 f"m ≥ 1 share of each driver's kinetic energy: {', '.join(shares)}. Rigid walls, no stuffing.",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    pages.append(fig)
    print(f"  {f:.0f} Hz rendered")

tmp, target = out_dir / "column_views_3d.pdf.tmp", out_dir / "column_views_3d.pdf"
with PdfPages(tmp) as pdf:
    for fg in pages:
        pdf.savefig(fg, dpi=110)
        plt.close(fg)
os.replace(tmp, target)
h5.close()
print(f"✓ {target}")
