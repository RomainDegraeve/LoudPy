"""
Loading_script.py — plots of the driver–enclosure coupling (Running_script.py first).

What this produces (in Results/Figures/)
-----------------------------------------
  - response.pdf        : kinetic energy of the driver per azimuthal order, share of the
                          non-axisymmetric orders, dust-cap velocity (with / without the
                          enclosure), enclosure pressure at the probe, rocking signature
                          (cone axial displacement at four angles)
  - coupled_modes.pdf   : coupled driver–enclosure modes (f, ζ, dominant component)
  - cavity_modes.pdf    : first enclosure modes on the walls
  - deformation_3d.pdf  : 3D deformed driver (all orders summed) at the frequencies where the
                          non-axisymmetric orders peak: 3D cut-away view + top view,
                          max deformation 10 % of the diameter
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
from enclosure import X0, Z_B

lp.plot.style.use()                       # loudpy figure style (fonts, colours, SPL in dB RMS)

here = Path("LoudPy_exemples/Coupling_3D/Driver_in_cavity")
out_dir = here / "Results/Figures"
out_dir.mkdir(parents=True, exist_ok=True)

N_PEAKS = 6                     # frequencies shown in deformation_3d.pdf
DEFORMATION = 0.10              # max |u| = 10 % of the driver diameter
RIM = (0.040, 0.0265)           # cone point of the rocking signature (r, z) [m]
N_CAV_SHOWN = 8                 # enclosure modes shown
IMG = (620, 500)
MIC = (0.03, 0.04, Z_B - 0.08)  # pressure probe in the enclosure [m]

res = lp.load(here / "Results/Files/coupled.h5")
hp = res.driver("main")
drv = hp.modes
f, q, q0 = res.frequencies, hp.q, hp.in_vacuo()
p_mic = res.cavity_pressure(MIC)
cav_freqs = res.cavity_frequencies
orders, m_of = [int(m) for m in np.unique(drv.m)], drv.m
w = 2 * np.pi * f


def save_pdf(name, figs):
    tmp, target = out_dir / (name + ".tmp"), out_dir / name   # new file then rename (synced folder)
    with PdfPages(tmp) as pdf:
        for fig in figs:
            pdf.savefig(lp.plot.style.tidy(fig), bbox_inches="tight", dpi=130)
            plt.close(fig)
    os.replace(tmp, target)
    print(f"  ✓ {target}")


# ── response.pdf ───────────────────────────────────────────────────────────────
E = {m: 0.5 * w ** 2 * np.sum(np.abs(q[:, m_of == m]) ** 2, axis=1) for m in orders}
E0 = 0.5 * w ** 2 * np.sum(np.abs(q0) ** 2, axis=1)
Etot = sum(E.values())
axis_node = int(np.argmax(drv.xy[:, 1] * (drv.xy[:, 0] < 1e-9)))
rim = int(np.argmin(np.linalg.norm(drv.xy - np.array(RIM), axis=1)))
v_cap = 1j * w * (q @ drv.U[:, axis_node, 1])
v_cap0 = 1j * w * (q0 @ drv.U[:, axis_node, 1])

fig, ax = plt.subplots(2, 2, figsize=(14, 9))
for m in orders:
    ax[0, 0].loglog(f, E[m], label=f"m = {m}")
ax[0, 0].loglog(f, E0, "k--", lw=0.8, label="no enclosure (m = 0 only)")
ax[0, 0].set_ylabel("kinetic energy per order [J / N²]")
ax[0, 0].set_title("Driver kinetic energy by azimuthal order (1 N on the coil)", fontsize=10)
for m in orders[1:]:
    ax[0, 1].semilogx(f, 100 * E[m] / Etot, label=f"m = {m}")
ax[0, 1].semilogx(f, 100 * sum(E[m] for m in orders[1:]) / Etot, "k", lw=1.2, label="all m ≥ 1")
ax[0, 1].set_ylabel("share of kinetic energy [%]")
ax[0, 1].set_title("Non-axisymmetric motion driven through the enclosure", fontsize=10)
ax[1, 0].loglog(f, np.abs(v_cap), label="in the enclosure")
ax[1, 0].loglog(f, np.abs(v_cap0), "k--", lw=0.8, label="no enclosure")
ax[1, 0].set_ylabel("|v| dust cap, on axis [m/s / N]")
ax[1, 0].set_title("Dust-cap axial velocity", fontsize=10)
ax[1, 1].semilogx(f, lp.spl(p_mic))
ax[1, 1].set_ylabel("SPL in the enclosure [dB re 20 µPa RMS / N]")
ax[1, 1].set_title(f"Pressure at the probe {np.round(MIC, 3)} m", fontsize=10)
for a in ax.ravel():
    a.set_xlabel("f [Hz]")
    a.grid(True)
    if a.get_legend_handles_labels()[0]:
        a.legend(fontsize=8)
for fc in cav_freqs[1:]:
    if fc < f[-1]:
        ax[1, 1].axvline(fc, color="0.85", lw=0.6, zorder=0)
fig.suptitle("HPNEW in a non-axisymmetric enclosure (front in vacuo); grey lines: enclosure modes",
             fontsize=11)
fig.tight_layout()

fig2, ax2 = plt.subplots(figsize=(14, 5))
for th in (0, 90, 180, 270):
    a_ = drv.angular(np.deg2rad(th))[:, 0]
    uz = q @ (drv.U[:, rim, 1] * a_)
    ax2.loglog(f, np.abs(uz), label=f"θ = {th}°")
ax2.set_xlabel("f [Hz]")
ax2.set_ylabel("|u_z| [m / N]")
ax2.set_title(f"Rocking signature: axial displacement of the cone at r = {drv.xy[rim, 0] * 1e3:.0f} mm, "
              f"four angles (identical without the enclosure)", fontsize=10)
ax2.grid(True)
ax2.legend(fontsize=8)
save_pdf("response.pdf", [fig, fig2])

# ── coupled_modes.pdf ──────────────────────────────────────────────────────────
modes = lp.load(here / "Results/Files/coupled_modes.h5")
s, Xq, XP = modes.extra("s"), modes.extra("q").T, modes.extra("P").T
fn, zeta = np.abs(s) / (2 * np.pi), -s.real / np.abs(s)
share = np.array([[np.sum(np.abs(Xq[m_of == m, i]) ** 2) for m in orders] for i in range(len(s))])
dom = np.argmax(share, axis=1)
cav_like = np.array([np.argmax(np.abs(XP[:, i])) for i in range(len(s))])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5))
for k, m in enumerate(orders):
    sel = dom == k
    a1.semilogy(fn[sel], zeta[sel], "o", ms=5, label=f"driver m = {m}")
for fc in cav_freqs[1:]:
    if fc < fn.max():
        a1.axvline(fc, color="0.85", lw=0.6, zorder=0)
a1.set_xlabel("f [Hz]")
a1.set_ylabel(r"$\zeta$")
a1.set_title("Coupled modes, coloured by the dominant driver order (grey: rigid-wall enclosure modes)",
             fontsize=9)
a1.grid(True)
a1.legend(fontsize=8)
a2.axis("off")
rows = [[f"{fn[i]:.0f}", f"{zeta[i]:.4f}", f"m = {orders[dom[i]]}",
         " ".join(f"{100 * share[i, k] / share[i].sum():.0f}" for k in range(len(orders))),
         f"{cav_freqs[cav_like[i]]:.0f}"] for i in range(min(len(s), 30))]
tab = a2.table(cellText=rows, colLabels=["f [Hz]", "ζ", "driver", "% m=0 1 2 3", "main cavity mode [Hz]"],
               loc="upper center", cellLoc="center")
tab.auto_set_font_size(False)
tab.set_fontsize(7)
tab.scale(1, 1.05)
a2.set_title("first 30 coupled modes", fontsize=9)
fig.tight_layout()
save_pdf("coupled_modes.pdf", [fig])


# ── helpers for the 3D views ───────────────────────────────────────────────────
def shot(mesh, scalars, view, clim=None, cmap="viridis"):
    p = pv.Plotter(off_screen=True, window_size=IMG)
    p.set_background("white")
    p.add_mesh(mesh, scalars=scalars, cmap=cmap, clim=clim, show_scalar_bar=False, smooth_shading=True)
    if view == "3d":
        p.camera_position = "xz"
        p.camera.azimuth, p.camera.elevation = 35, 25
    elif view == "top":
        p.camera_position = "xy"
    else:
        p.camera_position = view
    p.reset_camera()
    p.camera.zoom(1.3)
    img = p.screenshot(return_img=True)
    p.close()
    return img


# ── cavity_modes.pdf ───────────────────────────────────────────────────────────
cav = CavityModes.load(here / "Results/Files/cavity_modes.h5")
fc_ = cav.tets[:, [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]].reshape(-1, 3)
key = np.sort(fc_, axis=1)
_, idx, cnt = np.unique(key, axis=0, return_index=True, return_counts=True)
bfaces = fc_[idx[cnt == 1]]
surf = pv.PolyData(cav.nodes, np.hstack([np.full((len(bfaces), 1), 3), bfaces]).ravel())
figs = []
for start in range(1, 1 + N_CAV_SHOWN, 4):
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    for j, n in enumerate(range(start, start + 4)):
        psi = cav.psi[:, n] / np.abs(cav.psi[:, n]).max()
        surf["psi"] = psi
        cut_ = surf.clip(normal="-x", origin=(X0, 0, 0))
        cam = [(-0.45, 0.05, 0.25), (0.125, 0.15, -0.04), (0, 0, 1)]
        axs[0, j].imshow(shot(surf, "psi", cam, clim=(-1, 1), cmap="RdBu_r"))
        axs[1, j].imshow(shot(cut_, "psi", cam, clim=(-1, 1), cmap="RdBu_r"))
        axs[0, j].set_title(f"enclosure mode {n}: {cav.freqs[n]:.0f} Hz", fontsize=10)
        axs[1, j].set_title("cut through the driver axis", fontsize=9)
    for a in axs.ravel():
        a.axis("off")
    fig.suptitle("Rigid-wall enclosure modes (pressure, red / blue = ±)", fontsize=11)
    fig.tight_layout()
    figs.append(fig)
save_pdf("cavity_modes.pdf", figs)

# ── deformation_3d.pdf ─────────────────────────────────────────────────────────
tris = drv.tris
e = np.sort(np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
uniq, cnt = np.unique(e, axis=0, return_counts=True)
edges = uniq[cnt == 1]
nodes = np.unique(edges)
loc = np.full(len(drv.xy), -1)
loc[nodes] = np.arange(len(nodes))
size = 2 * drv.xy[:, 0].max()


def driver_surface(qf, sweep, n_theta=73):
    th = np.linspace(0, sweep, n_theta)
    ur, ut, uz = drv.displacement(qf, nodes, th)
    amp = np.sqrt(np.abs(ur) ** 2 + np.abs(ut) ** 2 + np.abs(uz) ** 2)
    j = np.argmax(amp.ravel())
    big = ur.ravel()[j] if abs(ur.ravel()[j]) > abs(uz.ravel()[j]) else uz.ravel()[j]
    ph = np.exp(-1j * np.angle(big))                       # phase of the largest displacement
    ur, ut, uz = (np.real(x * ph) for x in (ur, ut, uz))
    mag = np.sqrt(ur ** 2 + ut ** 2 + uz ** 2)
    sc = DEFORMATION * size / max(mag.max(), 1e-30)
    r, z = drv.xy[nodes, 0][:, None], drv.xy[nodes, 1][:, None]
    rr = r + sc * ur
    pts = np.column_stack([(rr * np.cos(th) - sc * ut * np.sin(th)).ravel(),
                           (rr * np.sin(th) + sc * ut * np.cos(th)).ravel(), (z + sc * uz).ravel()])
    ia, ib = loc[edges[:, 0]], loc[edges[:, 1]]
    k = np.arange(n_theta - 1)
    quads = np.stack([np.full((len(edges), n_theta - 1), 4), ia[:, None] * n_theta + k,
                      ib[:, None] * n_theta + k, ib[:, None] * n_theta + k + 1,
                      ia[:, None] * n_theta + k + 1], axis=2)
    s_ = pv.PolyData(pts, quads.reshape(-1))
    s_["|u|"] = (mag / mag.max()).ravel()
    return s_


frac = sum(E[m] for m in orders[1:]) / Etot
cand = [i for i in range(1, len(f) - 1) if frac[i] >= frac[i - 1] and frac[i] >= frac[i + 1]]
peaks = sorted(sorted(cand, key=lambda i: -frac[i])[:N_PEAKS])
figs = []
for start in range(0, len(peaks), 3):
    sel = peaks[start:start + 3]
    fig, axs = plt.subplots(3, 2, figsize=(12, 13))
    for a in axs.ravel():
        a.axis("off")
    for row, i in zip(axs, sel):
        row[0].imshow(shot(driver_surface(q[i], 1.5 * np.pi), "|u|", "3d", clim=(0, 1)))
        row[1].imshow(shot(driver_surface(q[i], 2 * np.pi), "|u|", "top", clim=(0, 1)))
        parts = "  ".join(f"m{m}: {100 * E[m][i] / Etot[i]:.0f}%" for m in orders)
        row[0].set_title(f"{f[i]:.0f} Hz   (3D view)   kinetic energy  {parts}", fontsize=9)
        row[1].set_title("top view", fontsize=9)
    fig.suptitle("Driver in the enclosure, 1 N on the coil: deformed shape at the peaks of the "
                 "non-axisymmetric motion\n(all orders summed, max |u| = 10 % of the diameter, colour = |u|)",
                 fontsize=10)
    fig.tight_layout()
    figs.append(fig)
save_pdf("deformation_3d.pdf", figs)
print(f"✓ figures → {out_dir}")
