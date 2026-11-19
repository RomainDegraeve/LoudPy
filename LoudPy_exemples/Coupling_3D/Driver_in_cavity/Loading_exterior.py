"""
Loading_exterior.py — radiation of the enclosure standing on a rigid ground
(Running_exterior.py first). Pressures outside the DtN hemisphere come from the DtN expansion
(exact, ground reflection included); 1 N on the voice coil.

What this produces (in Results/Figures/)
-----------------------------------------
  - spl_1m.pdf          : SPL at 1 m from the driver, on axis and at several angles
  - polars.pdf          : horizontal and vertical directivity maps (SPL re on-axis vs angle and
                          frequency) and polar curves at a few frequencies, at DIST m
  - response_change.pdf : driver kinetic energy per order and dust-cap velocity, enclosure only
                          (front in vacuo) vs enclosure + exterior air
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

import loudpy as lp
from enclosure import X0, Y0, Z_F, sealed_scene

lp.plot.style.use()                       # loudpy figure style (fonts, colours, SPL in dB RMS)

files = Path("LoudPy_exemples/Coupling_3D/Driver_in_cavity/Results/Files")
out_dir = Path("LoudPy_exemples/Coupling_3D/Driver_in_cavity/Results/Figures")
DIST = 1.0                                   # measurement distance from the driver centre [m]
H_ANGLES = [0, 30, 60, 90, 180]              # horizontal angles of spl_1m.pdf [deg]
POLAR_F = [100, 300, 1000, 2000, 3000]       # frequencies of the polar curves [Hz]

res = lp.load(files / "exterior.h5")
f = res.frequencies
hp = res.driver("main")
q = hp.q
# the same driver and enclosure, front in vacuo (cavity modes only: fast)
encl = lp.FrequencyResponse(sealed_scene(), drive=lp.Force(1.0)).solve(f, log=lambda *a, **k: None)
q_encl = encl.driver("main").q
src = np.array([X0, Y0, Z_F])
pressure = res.far_field
spl = lp.spl                                 # dB re 20 µPa RMS
h_pts = lambda a: src + DIST * np.column_stack([np.sin(a), 0 * a, np.cos(a)])    # horizontal plane
v_pts = lambda a: src + DIST * np.column_stack([0 * a, np.sin(a), np.cos(a)])    # vertical plane


def save_pdf(name, figs):
    tmp, target = out_dir / (name + ".tmp"), out_dir / name
    with PdfPages(tmp) as pdf:
        for fig in figs:
            pdf.savefig(fig, bbox_inches="tight", dpi=130)
            plt.close(fig)
    os.replace(tmp, target)
    print(f"  ✓ {target}")


# ── spl_1m.pdf ─────────────────────────────────────────────────────────────────
P_h = pressure(h_pts(np.deg2rad(H_ANGLES)))
fig, ax = plt.subplots(figsize=(13, 6))
for j, a in enumerate(H_ANGLES):
    ax.semilogx(f, spl(P_h[:, j]), lw=1.4 if a == 0 else 0.9, label=f"{a}° (horizontal)")
ax.set_xlabel("f [Hz]")
ax.set_ylabel(f"SPL at {DIST:.0f} m [dB re 20 µPa RMS / N]")
ax.set_title(f"HPNEW in the enclosure, standing on a rigid ground: SPL at {DIST:.0f} m from the driver "
             f"(height {Y0:.2f} m), 1 N on the coil", fontsize=10)
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8)
save_pdf("spl_1m.pdf", [fig])

# ── polars.pdf ─────────────────────────────────────────────────────────────────
ah = np.deg2rad(np.arange(-180, 181, 2))
av = np.deg2rad(np.arange(-10, 191, 2))                   # from just above the ground, over the top
Ph, Pv = pressure(h_pts(ah)), pressure(v_pts(av))
ref = spl(pressure(h_pts(np.array([0.0])))[:, 0])[:, None]
figs = []
fig, axs = plt.subplots(1, 2, figsize=(15, 5.5))
for a, P, ang, name in ((axs[0], Ph, ah, "horizontal (around the vertical axis)"),
                        (axs[1], Pv, av, "vertical (driver axis → top → back; −10° near the ground)")):
    im = a.pcolormesh(f, np.rad2deg(ang), (spl(P) - ref).T, cmap="viridis", vmin=-30, vmax=6, rasterized=True,
                      shading="auto")
    a.set_xscale("log")
    a.set_xlabel("f [Hz]")
    a.set_ylabel("angle [deg]")
    a.set_title(f"Directivity, {name}, SPL re on axis at {DIST:.0f} m", fontsize=9)
    plt.colorbar(im, ax=a, label="dB")
fig.tight_layout()
figs.append(fig)
fig, axs = plt.subplots(1, 2, figsize=(14, 7), subplot_kw=dict(projection="polar"))
for fp in POLAR_F:
    i = int(np.argmin(np.abs(f - fp)))
    axs[0].plot(ah, spl(Ph[i]) - ref[i, 0], label=f"{f[i]:.0f} Hz")
    axs[1].plot(av, spl(Pv[i]) - ref[i, 0], label=f"{f[i]:.0f} Hz")
for a, t in zip(axs, ("horizontal", "vertical (0° = axis, 90° = up)")):
    a.set_theta_zero_location("N")
    a.set_rlim(-30, 6)
    a.set_title(f"{t}, dB re on axis", fontsize=9)
    a.legend(fontsize=7, loc="lower left", bbox_to_anchor=(-0.1, -0.12))
fig.tight_layout()
figs.append(fig)
save_pdf("polars.pdf", figs)

# ── response_change.pdf ────────────────────────────────────────────────────────
drv = hp.modes
m_of, orders = drv.m, [int(m) for m in np.unique(drv.m)]
U = drv.U
w = 2 * np.pi * f
axis_node = int(np.argmax(drv.xy[:, 1] * (drv.xy[:, 0] < 1e-9)))
fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
for k, m in enumerate(orders):
    for qq, ls, lab in ((q_encl, "--", "enclosure only"), (q, "-", "enclosure + exterior")):
        E = 0.5 * w ** 2 * np.sum(np.abs(qq[:, m_of == m]) ** 2, axis=1)
        ax[0].loglog(f, E, ls, color=f"C{k}", lw=0.9, label=f"m = {m}, {lab}")
for qq, ls, lab in ((q_encl, "--", "enclosure only (front in vacuo)"), (q, "-", "enclosure + exterior")):
    ax[1].loglog(f, np.abs(1j * w * (qq @ U[:, axis_node, 1])), ls, label=lab)
ax[0].set_ylabel("kinetic energy [J / N²]")
ax[0].set_title("Driver kinetic energy per azimuthal order", fontsize=10)
ax[1].set_ylabel("|v| dust cap [m/s / N]")
ax[1].set_title("Dust-cap axial velocity", fontsize=10)
for a in ax:
    a.set_xlabel("f [Hz]")
    a.grid(True, which="both", alpha=0.3)
    a.legend(fontsize=7, ncol=2 if a is ax[0] else 1)
fig.suptitle("Effect of the exterior air (radiation loading of the front) on the driver", fontsize=11)
fig.tight_layout()
save_pdf("response_change.pdf", [fig])
print(f"✓ figures → {out_dir}")
