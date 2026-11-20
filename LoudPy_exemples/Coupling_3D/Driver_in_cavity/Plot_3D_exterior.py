"""
Plot_3D_exterior.py — 3D views of the full coupled problem every 10 Hz (Running_exterior.py first):
exterior air (half-space above the rigid ground), enclosure air and driver, 1 N on the coil.

One row per frequency:
  - left  : vertical cut through the driver axis — exterior pressure (cut + ground), enclosure
            pressure, deformed driver in place;
  - middle: horizontal cut at the driver height, seen from above — the wavefronts around the box;
  - right : the deformed driver, front view.
Same instant everywhere (phase of the largest driver displacement). Each pressure is shown
red/blue = ± its own maximum, given in the titles (the enclosure pressure is much larger than
the radiated one at low frequency); outside, the maximum is taken beyond NEAR_FIELD from the
driver and the near field saturates. Deformation max = 10 % of the driver diameter.

What this produces (in Results/Figures/exterior_3d/)
-----------------------------------------------------
  - exterior_3d_<f0>-<f1>Hz.pdf : one file per BAND Hz, ROWS frequencies per page
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
from loudpy.physics.modal import CavityModes
from loudpy.results.surfaces import driver_surface, cavity_grid
from enclosure import X0, Y0, Z_F

lp.plot.style.use()                       # loudpy figure style (fonts, colours, SPL in dB RMS)

files = Path("LoudPy_exemples/Coupling_3D/Driver_in_cavity/Results/Files")
out_dir = Path("LoudPy_exemples/Coupling_3D/Driver_in_cavity/Results/Figures/exterior_3d")
out_dir.mkdir(parents=True, exist_ok=True)

BAND = 500.0                  # frequencies per PDF file [Hz]
ROWS = 2                      # frequencies per page
DEFORMATION = 0.10
NEAR_FIELD = 0.08             # outside colour scale: max |p| beyond this distance from the driver [m]
IMG = (640, 560)
CAM_V = [(-1.25, 0.45, 0.25), (0.125, 0.22, -0.03), (0.0, 1.0, 0.0)]     # vertical cut, from −x
CAM_H = [(0.125, 2.2, -0.03), (0.125, 0.0, -0.03), (0.0, 0.0, 1.0)]       # from above, front up

res = lp.load(files / "exterior.h5")
freqs = res.frequencies
hp = res.driver("main")
drv, q, P_cav = hp.modes, hp.q, res.extra("P_cavity")
cav = CavityModes.load(files / "cavity_modes.h5")
m_of, orders = drv.m, [int(m) for m in np.unique(drv.m)]

cgrid = cavity_grid(cav)
c_corner = np.unique(cav.tets[:, :4])
egrids = {}


def ext_grid(mesh):
    if id(mesh) not in egrids:
        nodes, tets = mesh.coords, mesh.groups["air"].cells
        grid = pv.UnstructuredGrid({pv.CellType.TETRA: tets[:, :4]}, nodes)
        corner = np.unique(tets[:, :4])
        far = corner[np.linalg.norm(nodes[corner] - np.array([X0, Y0, Z_F]), axis=1) > NEAR_FIELD]
        egrids[id(mesh)] = (grid, far)
    return egrids[id(mesh)]


def clip(grid, values, normal, origin):
    grid["p"] = values
    return grid.clip(normal=normal, origin=origin).extract_surface()


def render(meshes, cam, size=IMG):
    p = pv.Plotter(off_screen=True, window_size=size)
    p.set_background("white")
    for mesh, kw in meshes:
        p.add_mesh(mesh, show_scalar_bar=False, **kw)
    p.camera_position = cam
    p.reset_camera()
    p.camera.zoom(1.15)
    img = p.screenshot(return_img=True)
    p.close()
    return img


RB = dict(scalars="p", cmap="RdBu_r", clim=(-1, 1))
UU = dict(scalars="|u|", cmap="viridis", clim=(0, 1), smooth_shading=True)


def row(axs, i):
    view = res.field("p", index=i)
    f = view.coordinate
    grid, e_corner = ext_grid(view.mesh)
    p_ext = np.asarray(view.values, complex)
    p_cav = cav.psi @ P_cav[i]
    drv_v, ph = driver_surface(drv, q[i], (X0, Y0, 0.0), (-np.pi / 2, np.pi / 2), 37, DEFORMATION)
    drv_h, _ = driver_surface(drv, q[i], (X0, Y0, 0.0), (np.pi, 2 * np.pi), 37, DEFORMATION, ph)
    drv_f, _ = driver_surface(drv, q[i], (0.0, 0.0, 0.0), (0.0, 2 * np.pi), 73, DEFORMATION, ph)
    e_max = max(np.abs(p_ext[e_corner]).max(), 1e-30)
    c_max = max(np.abs(p_cav[c_corner]).max(), 1e-30)
    pe = np.clip(np.real(p_ext * np.exp(1j * ph)) / e_max, -1, 1)          # near field saturated
    pc = np.real(p_cav * np.exp(1j * ph)) / c_max
    o_v, o_h = (X0, 0.0, 0.0), (0.0, Y0, 0.0)
    axs[0].imshow(render([(clip(grid, pe, "-x", o_v), RB), (clip(cgrid, pc, "-x", o_v), RB),
                          (drv_v, UU)], CAM_V))
    axs[1].imshow(render([(clip(grid, pe, "y", o_h), RB), (clip(cgrid, pc, "y", o_h), RB),
                          (drv_h, UU)], CAM_H))
    axs[2].imshow(render([(drv_f, UU)], [(0, 0, 1), (0, 0, 0), (0, 1, 0)], (480, 480)))
    E = np.array([np.sum(np.abs(q[i, m_of == m]) ** 2) for m in orders])
    share = "  ".join(f"m{m} {100 * e / E.sum():.0f}%" for m, e in zip(orders, E))
    axs[0].set_title(f"{f:.0f} Hz — vertical cut: outside (beyond {100 * NEAR_FIELD:.0f} cm) max |p| = {e_max:.3g} Pa/N "
                     f"({lp.spl(e_max):.0f} dB RMS), inside {c_max:.3g} Pa/N "
                     f"({lp.spl(c_max):.0f} dB RMS)", fontsize=8)
    axs[1].set_title(f"horizontal cut at the driver height ({Y0:.2f} m), from above", fontsize=8)
    axs[2].set_title(f"driver, front view\nkinetic energy  {share}", fontsize=8)


lo = freqs[0] - freqs[0] % BAND
for b0 in np.arange(lo, freqs[-1], BAND):
    sel = [i for i, f in enumerate(freqs) if b0 <= f < b0 + BAND or (f == freqs[-1] and b0 + BAND >= f)]
    if not sel:
        continue
    name = f"exterior_3d_{freqs[sel[0]]:.0f}-{freqs[sel[-1]]:.0f}Hz.pdf"
    tmp = out_dir / (name + ".tmp")
    with PdfPages(tmp) as pdf:
        for s0 in range(0, len(sel), ROWS):
            fig, axs = plt.subplots(ROWS, 3, figsize=(18, 5.8 * ROWS),
                                    gridspec_kw=dict(width_ratios=[1.15, 1.15, 0.8]))
            for a in axs.ravel():
                a.axis("off")
            for k_, i in enumerate(sel[s0:s0 + ROWS]):
                row(axs[k_], i)
            fig.suptitle("HPNEW in the enclosure on a rigid ground, 1 N on the coil — pressure outside and "
                         "inside (red/blue = ± each one's max), deformation (|u|, max 10 % of the diameter), "
                         "same instant", fontsize=10)
            fig.tight_layout()
            pdf.savefig(lp.plot.style.tidy(fig), dpi=100)
            plt.close(fig)
    os.replace(tmp, out_dir / name)
    print(f"  ✓ {out_dir / name}  ({len(sel)} frequencies)")
