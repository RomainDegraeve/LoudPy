"""
Loading_vented.py — vented enclosure on a rigid ground vs the sealed one (Running_vented.py merged
and Running_exterior.py first). 1 N on the voice coil; pressures outside the hemisphere from the
DtN expansion (exact, ground reflection included).

What this produces (in Results/Figures/)
-----------------------------------------
  - vented_response.pdf : SPL at 1 m (on the driver axis, in front of the port, off axis) and
                          dust-cap excursion, vented vs sealed; directivity maps of the vented box;
                          polar curves at a few frequencies, vented vs sealed
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

import loudpy as lp
from enclosure import X0, Y0, Z_F, PORT_X, PORT_Y, port_tuning, vented_volume

lp.plot.style.use()                       # loudpy figure style (fonts, colours, SPL in dB RMS)

files = Path("LoudPy_exemples/Coupling_3D/Driver_in_cavity/Results/Files")
out_dir = Path("LoudPy_exemples/Coupling_3D/Driver_in_cavity/Results/Figures")
DIST = 1.0

spl = lp.spl                                                  # dB re 20 µPa RMS
vented, sealed = lp.load(files / "vented_merged.h5"), lp.load(files / "exterior.h5")
vent, seal = dict(f=vented.frequencies), dict(f=sealed.frequencies)
f_b_estimate = port_tuning(vented_volume())
src = np.array([X0, Y0, Z_F])
pts = {"driver axis": src + [0, 0, DIST], "in front of the port": np.array([PORT_X, PORT_Y, Z_F + DIST]),
       "30° horizontal": src + DIST * np.array([np.sin(np.pi / 6), 0, np.cos(np.pi / 6)]),
       "90° horizontal": src + DIST * np.array([1.0, 0, 0])}
P = np.array(list(pts.values()))
Pv, Ps = vented.far_field(P), sealed.far_field(P)
x_v = np.abs(vented.driver("main").excursion())
x_s = np.abs(sealed.driver("main").excursion())
fb = vent["f"][np.argmin(np.where(vent["f"] < 200, x_v, np.inf))]

figs = []
fig, ax = plt.subplots(2, 1, figsize=(13, 10), sharex=True)
for j, (name, c) in enumerate(zip(pts, ["C0", "C3", "C1", "C2"])):
    ax[0].semilogx(vent["f"], spl(Pv[:, j]), color=c, lw=1.3 if j == 0 else 0.9, label=f"vented, {name}")
    ax[0].semilogx(seal["f"], spl(Ps[:, j]), color=c, ls="--", lw=0.8, label=f"sealed, {name}")
ax[0].set_ylabel(f"SPL at {DIST:.0f} m [dB re 20 µPa RMS / N]")
ax[0].set_title("Vented vs sealed enclosure on a rigid ground, 1 N on the coil", fontsize=10)
ax[1].loglog(vent["f"], 1e6 * x_v, label="vented")
ax[1].loglog(seal["f"], 1e6 * x_s, "--", label="sealed")
ax[1].set_ylabel("dust-cap excursion [µm / N]")
ax[1].set_xlabel("f [Hz]")
ax[1].set_title(f"Excursion minimum at {fb:.0f} Hz (port tuning; Helmholtz estimate "
                f"{f_b_estimate:.0f} Hz)", fontsize=10)
for a in ax:
    a.axvline(fb, color="0.6", lw=0.8, ls=":")
    a.grid(True)
    a.legend(fontsize=7, ncol=2)
fig.tight_layout()
figs.append(fig)

ah = np.deg2rad(np.arange(-180, 181, 2))
av = np.deg2rad(np.arange(-10, 191, 2))
Ph = vented.far_field(src + DIST * np.column_stack([np.sin(ah), 0 * ah, np.cos(ah)]))
Pvv = vented.far_field(src + DIST * np.column_stack([0 * av, np.sin(av), np.cos(av)]))
ref = spl(Pv[:, 0])[:, None]
fig, axs = plt.subplots(1, 2, figsize=(15, 5.5))
for a, Pm, ang, name in ((axs[0], Ph, ah, "horizontal"), (axs[1], Pvv, av, "vertical (−10° near the ground)")):
    im = a.pcolormesh(vent["f"], np.rad2deg(ang), (spl(Pm) - ref).T, cmap="viridis", vmin=-30, vmax=6,
                      shading="auto", rasterized=True)
    a.set_xscale("log")
    a.set_xlabel("f [Hz]")
    a.set_ylabel("angle [deg]")
    a.set_title(f"Vented box: directivity, {name}, SPL re on axis at {DIST:.0f} m", fontsize=9)
    plt.colorbar(im, ax=a, label="dB")
fig.tight_layout()
figs.append(fig)

# polar curves, vented (solid) vs sealed (dashed), dB re on axis
POLAR_F = [50, 70, 100, 300, 1000, 2000, 3000]
Ph_s = sealed.far_field(src + DIST * np.column_stack([np.sin(ah), 0 * ah, np.cos(ah)]))
Pv_s = sealed.far_field(src + DIST * np.column_stack([0 * av, np.sin(av), np.cos(av)]))
ref_s = spl(Ps[:, 0])[:, None]
fig, axs = plt.subplots(1, 2, figsize=(15, 7.5), subplot_kw=dict(projection="polar"))
for k_, fp in enumerate(POLAR_F):
    iv, is_ = int(np.argmin(np.abs(vent["f"] - fp))), int(np.argmin(np.abs(seal["f"] - fp)))
    if abs(vent["f"][iv] - fp) > 15:                    # not computed (yet)
        continue
    col = f"C{k_}"
    axs[0].plot(ah, spl(Ph[iv]) - ref[iv, 0], color=col, label=f"{vent['f'][iv]:.0f} Hz vented")
    axs[0].plot(ah, spl(Ph_s[is_]) - ref_s[is_, 0], color=col, ls="--", lw=0.8, label=f"{seal['f'][is_]:.0f} Hz sealed")
    axs[1].plot(av, spl(Pvv[iv]) - ref[iv, 0], color=col)
    axs[1].plot(av, spl(Pv_s[is_]) - ref_s[is_, 0], color=col, ls="--", lw=0.8)
for a, ttl in zip(axs, ("horizontal (0° = driver axis)", "vertical (0° = driver axis, 90° = up, −10° near the ground)")):
    a.set_theta_zero_location("N")
    a.set_theta_direction(-1)
    a.set_rlim(-30, 6)
    a.set_rticks([-30, -20, -10, 0])
    a.set_title(f"Directivity at {DIST:.0f} m, {ttl}\ndB re on axis — solid: vented, dashed: sealed", fontsize=9)
axs[0].legend(fontsize=7, loc="lower left", bbox_to_anchor=(-0.15, -0.2), ncol=2)
fig.tight_layout()
figs.append(fig)

tmp, target = out_dir / "vented_response.pdf.tmp", out_dir / "vented_response.pdf"
with PdfPages(tmp) as pdf:
    for fg in figs:
        pdf.savefig(lp.plot.style.tidy(fg), bbox_inches="tight", dpi=130)
        plt.close(fg)
os.replace(tmp, target)
print(f"✓ {target}   (tuning from the excursion minimum: {fb:.0f} Hz)")
