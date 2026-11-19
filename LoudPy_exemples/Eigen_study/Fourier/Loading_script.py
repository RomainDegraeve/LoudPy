"""
Loading_script.py — figures of the azimuthal-order modes (Running_script.py first).

What this produces (in Results/Figures/)
-----------------------------------------
  - mode_chart.pdf      : frequency of every mode vs its azimuthal order m (colour = ζ)
  - modes_m<m>.pdf      : meridian view of the modes of each order (deformed by (U_r, U_z),
                          colour = |U| including U_θ), pages of PER_PAGE modes
  (3D views of every mode: Plot_3D_modes.py)
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import PolyCollection
import numpy as np

import loudpy as lp

lp.plot.style.use()                       # loudpy figure style (fonts, colours, SPL in dB RMS)

in_path = "LoudPy_exemples/Eigen_study/Fourier/Results/Files/eigen_fourier.h5"
out_dir = Path("LoudPy_exemples/Eigen_study/Fourier/Results/Figures")
out_dir.mkdir(parents=True, exist_ok=True)
PER_PAGE, SCALE = 12, 0.02     # meridian views: modes per page, deformation (fraction of size)

modes = lp.load(in_path)
print(modes, {m: len(modes.order(m)) for m in modes.orders})

# ── mode chart ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
for m in modes.orders:
    om = modes.order(m)
    sc = ax.scatter(np.full(len(om), m), om.frequencies, c=om.zeta, cmap="viridis", vmin=0, vmax=0.3, s=25)
ax.set_yscale("log")
ax.set_xticks(modes.orders)
ax.set_xlabel("azimuthal order m")
ax.set_ylabel("f [Hz]")
ax.set_title("Modes of the loudspeaker by azimuthal order")
fig.colorbar(sc, label=r"$\zeta$")
ax.grid(True, which="both", alpha=0.3)
fig.savefig(out_dir / "mode_chart.pdf", bbox_inches="tight")
plt.close(fig)

# ── meridian views per order ───────────────────────────────────────────────────
for m in modes.orders:
    om = modes.order(m)
    mesh = om.mesh("solid")
    tri, xy = mesh.triangles(), mesh.coords
    with PdfPages(out_dir / f"modes_m{m}.pdf") as pdf:
        for start in range(0, len(om), PER_PAGE):
            page = range(start, min(start + PER_PAGE, len(om)))
            nc = 4
            nr = int(np.ceil(len(page) / nc))
            fig, axs = plt.subplots(nr, nc, figsize=(4 * nc, 3 * nr), squeeze=False)
            for a in axs.ravel()[len(page):]:
                a.axis("off")
            for a, k in zip(axs.ravel(), page):
                mode = om.mode(k)
                U = np.real(mode.values)
                mag = np.linalg.norm(U, axis=1)
                s = SCALE * np.abs(xy).max() / max(mag.max(), 1e-30)
                d = xy + s * U[:, :2]
                a.add_collection(PolyCollection(xy[tri], facecolor="none", edgecolor="0.85", lw=0.2))
                a.add_collection(PolyCollection(d[tri], array=mag[tri].mean(axis=1), cmap="viridis", lw=0))
                a.autoscale()
                a.set_aspect("equal")
                a.set_title(f"m={m}  {mode.coordinate:.0f} Hz  zeta={mode.extra['zeta']:.3f}", fontsize=8)
                a.tick_params(labelsize=6)
            fig.suptitle(f"Order m = {m}: modes {start + 1}-{start + len(page)} of {len(om)}", fontsize=10)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

print(f"✓ figures → {out_dir}")
