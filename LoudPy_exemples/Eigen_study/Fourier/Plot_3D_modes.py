"""
Plot_3D_modes.py — 3D views of the azimuthal-order modes (Running_script.py first).

Every mode of the orders ORDERS is drawn on the whole structure (cone, dust cap, surround,
spider, former, coil): the boundary of the meridian mesh is revolved around the axis and
deformed by the 3D field of order m,

    u_r = U_r cos mθ,   u_z = U_z cos mθ,   u_θ = U_θ sin mθ,

coloured by |u|. Two views per mode:
  - 3D view   : revolved over 270° (the cut-away shows the section, spider and former);
  - top view  : full revolution seen from the front along the axis — the circumferential
                pattern on the membrane and the surround.
Only the deformed shape is drawn; the largest displacement is DEFORMATION (10 %) of the
largest dimension of the structure (its outer diameter).

What this produces (in Results/Figures/)
-----------------------------------------
  - modes_3d_m<m>.pdf : MODES_PER_PAGE modes per page, one row per mode (3D view | top view)
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pyvista as pv

import loudpy as lp

lp.plot.style.use()                       # loudpy figure style (fonts, colours, SPL in dB RMS)

in_path = "LoudPy_exemples/Eigen_study/Fourier/Results/Files/eigen_fourier.h5"
out_dir = Path("LoudPy_exemples/Eigen_study/Fourier/Results/Figures")
out_dir.mkdir(parents=True, exist_ok=True)

ORDERS = [0, 1, 2, 3]          # azimuthal orders drawn
DEFORMATION = 0.10             # max |u| = 10 % of the largest dimension of the structure
MODES_PER_PAGE = 3             # rows per page
N_THETA = 73                   # azimuthal resolution of the revolution
IMG = (620, 500)               # rendered image size [px]


def boundary_edges(tris):
    """Edges of the meridian mesh belonging to one triangle only (both faces of thin parts)."""
    e = np.sort(np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
    uniq, count = np.unique(e, axis=0, return_counts=True)
    return uniq[count == 1]


def revolved(xy, edges, U, m, scale, sweep):
    """Surface of the revolved boundary, deformed by scale × (order-m field of U)."""
    nodes = np.unique(edges)
    loc = np.full(xy.shape[0], -1)
    loc[nodes] = np.arange(len(nodes))
    th = np.linspace(0.0, sweep, N_THETA)
    r, z = xy[nodes, 0][:, None], xy[nodes, 1][:, None]
    Ur, Uz = U[nodes, 0][:, None], U[nodes, 1][:, None]
    Ut = U[nodes, 2][:, None] if U.shape[1] > 2 else np.zeros_like(Ur)
    c = np.cos(m * th)[None, :]
    s = np.sin(m * th)[None, :] if m > 0 else np.ones((1, N_THETA))   # m = 0: torsion uniform
    ur, uz, ut = Ur * c, Uz * c, Ut * s
    rr = r + scale * ur
    pts = np.column_stack([(rr * np.cos(th) - scale * ut * np.sin(th)).ravel(),
                           (rr * np.sin(th) + scale * ut * np.cos(th)).ravel(),
                           (z + scale * uz).ravel()])
    ia, ib = loc[edges[:, 0]], loc[edges[:, 1]]
    k = np.arange(N_THETA - 1)
    quads = np.stack([np.full((len(edges), N_THETA - 1), 4),
                      ia[:, None] * N_THETA + k, ib[:, None] * N_THETA + k,
                      ib[:, None] * N_THETA + k + 1, ia[:, None] * N_THETA + k + 1], axis=2)
    surf = pv.PolyData(pts, quads.reshape(-1))
    surf["|u|"] = np.sqrt(ur ** 2 + uz ** 2 + ut ** 2).ravel()
    return surf


def render(surf, view):
    p = pv.Plotter(off_screen=True, window_size=IMG)
    p.set_background("white")
    p.add_mesh(surf, scalars="|u|", cmap="viridis", clim=(0.0, 1.0), show_scalar_bar=False,
               smooth_shading=True)
    if view == "3d":
        p.camera_position = "xz"
        p.camera.azimuth = 35
        p.camera.elevation = 25
    else:                                        # top view: from the front, along the axis
        p.camera_position = "xy"
    p.reset_camera()
    p.camera.zoom(1.3)
    img = p.screenshot(return_img=True)
    p.close()
    return img


modes = lp.load(in_path)

for m in [m for m in ORDERS if m in modes.orders]:
    om = modes.order(m)
    mesh = om.mesh("solid")
    xy = mesh.coords
    edges = boundary_edges(mesh.triangles())
    size = max(2 * xy[:, 0].max(), np.ptp(xy[:, 1]))     # largest dimension (outer diameter)
    target = out_dir / f"modes_3d_m{m}.pdf"
    tmp = out_dir / f"modes_3d_m{m}.tmp.pdf"   # new file then rename (synced folders time out
    with PdfPages(tmp) as pdf:                 # when a large existing PDF is overwritten)
        for start in range(0, len(om), MODES_PER_PAGE):
            page = range(start, min(start + MODES_PER_PAGE, len(om)))
            fig, axs = plt.subplots(MODES_PER_PAGE, 2, figsize=(12, 4.4 * MODES_PER_PAGE))
            for a in axs.ravel():
                a.axis("off")
            for row, k in zip(axs, page):
                mode = om.mode(k)
                U = np.real(mode.values)
                U = U / max(np.linalg.norm(U, axis=1).max(), 1e-30)   # max |u| = 1
                scale = DEFORMATION * size
                row[0].imshow(render(revolved(xy, edges, U, m, scale, 1.5 * np.pi), "3d"))
                row[1].imshow(render(revolved(xy, edges, U, m, scale, 2 * np.pi), "top"))
                row[0].set_title(f"m = {m}   mode {k + 1}   {mode.coordinate:.0f} Hz   "
                                 f"zeta = {mode.extra['zeta']:.3f}   (3D view)", fontsize=10)
                row[1].set_title("top view", fontsize=10)
            fig.suptitle(f"Azimuthal order m = {m}: modes {start + 1}-{start + len(page)} of "
                         f"{len(om)}   (deformed shape, max |u| = {100 * DEFORMATION:.0f} % "
                         f"of the diameter; colour = |u|)", fontsize=11)
            fig.tight_layout()
            pdf.savefig(lp.plot.style.tidy(fig), dpi=110)
            plt.close(fig)
    os.replace(tmp, target)
    print(f"✓ m = {m}: {len(om)} modes → {target}")
