"""
Loading_labyrinth.py — response of the labyrinth cabinet on a rigid ground (Running_labyrinth.py
first), compared with the sealed and vented boxes of ../Driver_in_cavity when their results exist.
1 N on the voice coil; pressures outside the hemisphere from the DtN expansion (ground included).

What this produces (in Results/Figures/)
-----------------------------------------
  - labyrinth_response.pdf : SPL at 1 m (driver axis, in front of the mouth, off axis) and
                             dust-cap excursion, labyrinth vs sealed vs vented; directivity
                             maps; polar curves at a few frequencies
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from loudpy.Studies.Cavity_coupling import DriverModes, C_AIR
from loudpy.Studies.Exterior_dtn import HalfSpaceDtN, _h2
from labyrinth import eig_path, X0, Y0, Z_F, W_IN, ORDERS, DRV_F_MAX, mouth

files = Path("LoudPy_exemples/Coupling_3D/Labyrinth/Results/Files")
other = Path("LoudPy_exemples/Coupling_3D/Driver_in_cavity/Results/Files")
out_dir = Path("LoudPy_exemples/Coupling_3D/Labyrinth/Results/Figures")
out_dir.mkdir(parents=True, exist_ok=True)
DIST = 1.0
POLAR_F = [50, 70, 100, 200, 500, 1000, 2000]

drv = DriverModes.from_eigen_h5(eig_path, ORDERS, f_max=DRV_F_MAX)
axis_node = int(np.argmax(drv.xy[:, 1] * (drv.xy[:, 0] < 1e-9)))


def load(path):
    d = np.load(path, allow_pickle=True)
    q = d["y"][:, :int(d["n_s"])] if "y" in d else d["q"]
    if q.shape[1] != len(drv):                       # results of another driver basis: skip
        return None
    return dict(f=d["freqs"], q=q, C=d["coefs"], dtn=HalfSpaceDtN(d["center"], float(d["radius"]), int(d["up"])))


def pressure(m, points):
    f, C, dtn = m["f"], m["C"], m["dtn"]
    n_max = int(round((np.sqrt(8 * C.shape[1] + 1) - 3) / 2))
    n_of = dtn.harmonics(n_max)[:, 0]
    r, th, ph = dtn.local(points)
    Y = dtn.Y(th, ph, n_max)
    out = np.zeros((len(f), len(points)), complex)
    for i, fi in enumerate(f):
        k = 2 * np.pi * fi / C_AIR
        nh = np.count_nonzero(C[i])
        ratio = _h2(n_of[None, :nh], k * r[:, None]) / _h2(n_of[:nh], k * dtn.R)[None, :]
        out[i] = (Y[:, :nh] * ratio) @ C[i, :nh]
    return out


spl = lambda p: 20 * np.log10(np.maximum(np.abs(p) / np.sqrt(2), 1e-12) / 2e-5)       # RMS
lab = load(files / "labyrinth.npz")
d = np.load(files / "labyrinth.npz")
f_qw = float(d["f_quarter_wave"])
refs = {k: load(other / n) for k, n in (("sealed", "exterior.npz"), ("vented", "vented_merged.npz"))
        if (other / n).exists()}
refs = {k: v for k, v in refs.items() if v is not None}
src = np.array([X0, Y0, Z_F])
yb, yt = mouth()
pts = {"driver axis": src + [0, 0, DIST], "in front of the mouth": np.array([X0, 0.5 * (yb + yt), Z_F + DIST]),
       "30° horizontal": src + DIST * np.array([np.sin(np.pi / 6), 0, np.cos(np.pi / 6)]),
       "90° horizontal": src + DIST * np.array([1.0, 0, 0])}
P = np.array(list(pts.values()))
Pl = pressure(lab, P)
x_l = np.abs(lab["q"] @ drv.U[:, axis_node, 1])
lo = lab["f"] < 200
f_min_x = lab["f"][lo][np.argmin(x_l[lo])]

figs = []
fig, ax = plt.subplots(2, 1, figsize=(13, 10), sharex=True)
cols = ["C0", "C3", "C1", "C2"]
for j, (name, c) in enumerate(zip(pts, cols)):
    ax[0].semilogx(lab["f"], spl(Pl[:, j]), color=c, lw=1.4 if j == 0 else 0.9, label=f"labyrinth, {name}")
for k, (name, m) in enumerate(refs.items()):
    ax[0].semilogx(m["f"], spl(pressure(m, P[:1])[:, 0]), color="0.45", ls=["--", ":"][k], lw=0.9,
                   label=f"{name} box, driver axis")
    ax[1].loglog(m["f"], 1e6 * np.abs(m["q"] @ drv.U[:, axis_node, 1]), color="0.45", ls=["--", ":"][k],
                 lw=0.9, label=f"{name} box")
ax[1].loglog(lab["f"], 1e6 * x_l, "C0", lw=1.4, label="labyrinth")
ax[0].set_ylabel(f"SPL at {DIST:.0f} m [dB re 20 µPa RMS / N]")
ax[0].set_title("Labyrinth (folded transmission line, no stuffing) on a rigid ground, 1 N on the coil",
                fontsize=10)
ax[1].set_ylabel("dust-cap excursion [µm / N]")
ax[1].set_xlabel("f [Hz]")
ax[1].set_title(f"Excursion minimum at {f_min_x:.0f} Hz (line quarter-wave estimate {f_qw:.0f} Hz)", fontsize=10)
for a in ax:
    a.axvline(f_qw, color="0.7", lw=0.8, ls=":")
    a.grid(True, which="both", alpha=0.3)
    a.legend(fontsize=7, ncol=2)
fig.tight_layout()
figs.append(fig)

ah = np.deg2rad(np.arange(-180, 181, 2))
av = np.deg2rad(np.arange(-10, 191, 2))
Ph = pressure(lab, src + DIST * np.column_stack([np.sin(ah), 0 * ah, np.cos(ah)]))
Pv = pressure(lab, src + DIST * np.column_stack([0 * av, np.sin(av), np.cos(av)]))
ref = spl(Pl[:, 0])[:, None]
fig, axs = plt.subplots(1, 2, figsize=(15, 5.5))
for a, Pm, ang, name in ((axs[0], Ph, ah, "horizontal"), (axs[1], Pv, av, "vertical (−10° near the ground)")):
    im = a.pcolormesh(lab["f"], np.rad2deg(ang), (spl(Pm) - ref).T, cmap="viridis", vmin=-30, vmax=6,
                      shading="auto", rasterized=True)
    a.set_xscale("log")
    a.set_xlabel("f [Hz]")
    a.set_ylabel("angle [deg]")
    a.set_title(f"Labyrinth: directivity, {name}, SPL re on axis at {DIST:.0f} m", fontsize=9)
    plt.colorbar(im, ax=a, label="dB")
fig.tight_layout()
figs.append(fig)

fig, axs = plt.subplots(1, 2, figsize=(15, 7.5), subplot_kw=dict(projection="polar"))
for k, fp in enumerate(POLAR_F):
    i = int(np.argmin(np.abs(lab["f"] - fp)))
    if abs(lab["f"][i] - fp) > 15:
        continue
    axs[0].plot(ah, spl(Ph[i]) - ref[i, 0], color=f"C{k}", label=f"{lab['f'][i]:.0f} Hz")
    axs[1].plot(av, spl(Pv[i]) - ref[i, 0], color=f"C{k}")
for a, ttl in zip(axs, ("horizontal (0° = driver axis)", "vertical (0° = axis, 90° = up, −10° near the ground)")):
    a.set_theta_zero_location("N")
    a.set_theta_direction(-1)
    a.set_rlim(-30, 6)
    a.set_title(f"Labyrinth directivity at {DIST:.0f} m, {ttl}, dB re on axis", fontsize=9)
axs[0].legend(fontsize=7, loc="lower left", bbox_to_anchor=(-0.15, -0.2), ncol=2)
fig.tight_layout()
figs.append(fig)

tmp, target = out_dir / "labyrinth_response.pdf.tmp", out_dir / "labyrinth_response.pdf"
with PdfPages(tmp) as pdf:
    for fg in figs:
        pdf.savefig(fg, bbox_inches="tight", dpi=130)
        plt.close(fg)
os.replace(tmp, target)
print(f"✓ {target}   (excursion minimum {f_min_x:.0f} Hz, quarter-wave estimate {f_qw:.0f} Hz)")
