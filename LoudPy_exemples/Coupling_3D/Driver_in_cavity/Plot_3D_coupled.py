"""
Plot_3D_coupled.py — coupled acoustic + mechanical 3D views every DF Hz (Running_script.py first).

At each frequency (1 N on the voice coil) the forced response of the coupled system is
recomputed from the saved modal data, and one row shows:
  - left : the enclosure cut through the driver axis, coloured by the acoustic pressure, with the
           deformed driver in place (half of it, on the same side as the cut);
  - right: the deformed driver seen from the front, along the axis, the whole revolution —
           the circumferential pattern of the non-axisymmetric orders on cone and surround.
Pressure and displacement are shown at the same instant (phase of the largest displacement):
pressure colour ± max |p| of that frequency, max |u| = DEFORMATION × driver diameter.

What this produces (in Results/Figures/coupled_3d/)
----------------------------------------------------
  - coupled_3d_<f0>-<f1>Hz.pdf : one file per BAND Hz, ROWS frequencies per page
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pyvista as pv

from loudpy.Studies.Cavity_coupling import (CavityModes, DriverModes, solve_forced, driver_surface,
                                            cavity_grid, RHO_AIR, C_AIR)

here = Path("LoudPy_exemples/Coupling_3D/Driver_in_cavity")
eig_path = "LoudPy_exemples/Eigen_study/Fourier/Results/Files/eigen_fourier.h5"
out_dir = here / "Results/Figures/coupled_3d"
out_dir.mkdir(parents=True, exist_ok=True)

F_START, F_END, DF = 20.0, 3000.0, 10.0       # frequencies drawn [Hz]
BAND = 500.0                                  # frequencies per PDF file [Hz]
ROWS = 2                                      # frequencies per page
DEFORMATION = 0.10                            # max |u| = 10 % of the driver diameter
IMG_L, IMG_R = (820, 620), (520, 520)
CAM = [(-0.45, 0.05, 0.25), (0.125, 0.15, -0.04), (0.0, 0.0, 1.0)]   # inside the enclosure

# ── data ───────────────────────────────────────────────────────────────────────
d = np.load(here / "Results/Files/coupled.npz", allow_pickle=True)
cav = CavityModes.load(here / "Results/Files/cavity_modes.h5")
drv = DriverModes.from_eigen_h5(eig_path, list(d["orders"]), f_max=float(d["freq"].max()) + 1.0)
assert len(drv) == len(d["F"]), "driver modes changed since Running_script.py"
m_of, orders = d["m"], list(d["orders"])
x0, y0, z_b = d["center"]
freqs = np.arange(F_START, F_END + 0.5 * DF, DF)
q, P = solve_forced(freqs, d["F"], 2 * np.pi * d["freq"], d["zeta"], d["S"], d["R"], cav.kappa2,
                    RHO_AIR, C_AIR, float(d["eta_air"]))
w = 2 * np.pi * freqs
E = np.array([[0.5 * wi ** 2 * np.sum(np.abs(qi[m_of == m]) ** 2) for m in orders] for wi, qi in zip(w, q)])
print(f"{len(freqs)} frequencies, {len(cav.kappa2)} enclosure modes, {len(drv)} driver modes")

grid = cavity_grid(cav)
corner = np.unique(cav.tets[:, :4])                  # grid points carrying a value


def cut_pressure(pn):
    """Pressure on the cut through the driver axis: set on the grid, then clipped (linear interp.)."""
    grid["p"] = pn
    return grid.clip(normal="-x", origin=(x0, 0.0, 0.0)).extract_surface()


def render(meshes, cam, size):
    p = pv.Plotter(off_screen=True, window_size=size)
    p.set_background("white")
    for mesh, kw in meshes:
        p.add_mesh(mesh, show_scalar_bar=False, **kw)
    p.camera_position = cam
    p.reset_camera()
    p.camera.zoom(1.25)
    img = p.screenshot(return_img=True)
    p.close()
    return img


def row(axl, axr, i):
    drv_half, ph = driver_surface(drv, q[i], (x0, y0, 0.0), (-np.pi / 2, np.pi / 2), 37, DEFORMATION)
    drv_full, _ = driver_surface(drv, q[i], (0.0, 0.0, 0.0), (0.0, 2 * np.pi), 73, DEFORMATION, ph)
    p_nodes = cav.psi @ P[i]
    pmax = max(np.abs(p_nodes[corner]).max(), 1e-30)
    half = cut_pressure(np.real(p_nodes * np.exp(1j * ph)) / pmax)
    axl.imshow(render([(half, dict(scalars="p", cmap="RdBu_r", clim=(-1, 1))),
                       (drv_half, dict(scalars="|u|", cmap="viridis", clim=(0, 1), smooth_shading=True))],
                      CAM, IMG_L))
    axr.imshow(render([(drv_full, dict(scalars="|u|", cmap="viridis", clim=(0, 1), smooth_shading=True))],
                      [(0, 0, 1), (0, 0, 0), (0, 1, 0)], IMG_R))
    umax = np.abs(q[i] @ drv.U.reshape(len(drv), -1)).max()
    share = "  ".join(f"m{m} {100 * E[i, k] / E[i].sum():.0f}%" for k, m in enumerate(orders))
    axl.set_title(f"{freqs[i]:.0f} Hz — pressure in the enclosure (max |p| = {pmax:.3g} Pa/N, "
                  f"{20 * np.log10(pmax / 2e-5):.0f} dB) + driver", fontsize=9)
    axr.set_title(f"driver, front view (max |u| ≈ {umax:.2g} m/N)\nkinetic energy  {share}",
                  fontsize=8)


for b0 in np.arange(F_START - F_START % BAND, F_END, BAND):
    sel = [i for i, f in enumerate(freqs) if b0 <= f < b0 + BAND or (f == F_END and b0 + BAND >= F_END)]
    if not sel:
        continue
    name = f"coupled_3d_{freqs[sel[0]]:.0f}-{freqs[sel[-1]]:.0f}Hz.pdf"
    tmp = out_dir / (name + ".tmp")              # new file then rename (synced folder)
    with PdfPages(tmp) as pdf:
        for s0 in range(0, len(sel), ROWS):
            fig, axs = plt.subplots(ROWS, 2, figsize=(14, 5.6 * ROWS),
                                    gridspec_kw=dict(width_ratios=[1.45, 1]))
            for a in axs.ravel():
                a.axis("off")
            for k, i in enumerate(sel[s0:s0 + ROWS]):
                row(axs[k, 0], axs[k, 1], i)
            fig.suptitle("HPNEW in the enclosure, 1 N on the coil — pressure (red/blue = ±, scaled to its "
                         "max) and deformation (colour |u|, max = 10 % of the diameter), same instant",
                         fontsize=10)
            fig.tight_layout()
            pdf.savefig(fig, dpi=100)
            plt.close(fig)
    os.replace(tmp, out_dir / name)
    print(f"  ✓ {out_dir / name}  ({len(sel)} frequencies)")
