"""
Loading_column.py — response of the three-driver column with a folded transmission line, on a
rigid ground (Running_column.py first). Pressures outside the hemisphere from the DtN expansion
(ground included); all drivers driven with the DRIVE amplitudes of Running_column.py.

What this produces (in Results/Figures/)
-----------------------------------------
  - column_response.pdf :
      1. SPL at 1 m (front axis of the middle driver, behind the column at the mouth, 30° up and
         down) and the dust-cap excursion of each driver (top, middle, bottom)
      2. directivity maps, vertical (the column effect) and horizontal, re the front axis
      3. vertical and horizontal polar curves at a few frequencies
      4. per driver: share of the kinetic energy in m ≥ 1 (rocking, ovalisation — driven only by
         the non-uniform air load, the coil drives m = 0) and excursion phase re the middle driver
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from loudpy.Studies.Cavity_coupling import DriverModes, MultiDriver, C_AIR
from loudpy.Studies.Exterior_dtn import HalfSpaceDtN, _h2
from column import eig_path, X0, Y0S, Z_F, Z_BACK, WALL_T, MOUTH_H, CENTERS, NAMES, ORDERS, DRV_F_MAX

files = Path("LoudPy_exemples/Coupling_3D/Column_labyrinth/Results/Files")
out_dir = Path("LoudPy_exemples/Coupling_3D/Column_labyrinth/Results/Figures")
out_dir.mkdir(parents=True, exist_ok=True)
DIST = 1.0                                       # distance of the probes from the middle driver [m]
POLAR_F = [50, 100, 200, 400, 700, 1000]         # [Hz] (nearest computed frequency)

d = np.load(files / "column.npz", allow_pickle=True)
drivers = MultiDriver(DriverModes.from_eigen_h5(eig_path, ORDERS, f_max=DRV_F_MAX), CENTERS, names=NAMES)
n_s = int(d["n_s"])
assert len(drivers) == n_s, "driver modes changed since Running_column.py"
f, q, C = d["freqs"], d["y"][:, :n_s], d["coefs"]
dtn = HalfSpaceDtN(d["center"], float(d["radius"]), int(d["up"]))
f_qw = float(d["f_quarter_wave"])
drive = d["drive"]
drv = drivers.modes[0]
axis_node = int(np.argmax(drv.xy[:, 1] * (drv.xy[:, 0] < 1e-9)))         # dust-cap centre


def pressure(points):
    """Pressure (n_f, n_points) outside the hemisphere, from the DtN coefficients."""
    n_max = int(round((np.sqrt(8 * C.shape[1] + 1) - 3) / 2))
    n_of = dtn.harmonics(n_max)[:, 0]
    r, th, ph = dtn.local(points)
    if np.any(r < dtn.R):
        raise ValueError("probe inside the DtN hemisphere: increase DIST")
    Y = dtn.Y(th, ph, n_max)
    out = np.zeros((len(f), len(points)), complex)
    for i, fi in enumerate(f):
        k = 2 * np.pi * fi / C_AIR
        nh = np.count_nonzero(C[i])
        ratio = _h2(n_of[None, :nh], k * r[:, None]) / _h2(n_of[:nh], k * dtn.R)[None, :]
        out[i] = (Y[:, :nh] * ratio) @ C[i, :nh]
    return out


spl = lambda p: 20 * np.log10(np.maximum(np.abs(p) / np.sqrt(2), 1e-12) / 2e-5)       # RMS
src = np.array([X0, Y0S[1], Z_F])                                                     # middle driver
up30 = np.array([0, np.sin(np.pi / 6), np.cos(np.pi / 6)])
pts = {"front axis (middle driver)": src + [0, 0, DIST],
       "30° up": src + DIST * up30,
       "30° down": src + DIST * up30 * [1, -1, 1],
       "behind, at the mouth": np.array([X0, 0.5 * MOUTH_H, Z_BACK - WALL_T - DIST])}
P = np.array(list(pts.values()))
Pp = pressure(P)
qs = drivers.split(q)                                                  # per-driver amplitudes
x = np.array([qj @ drv.U[:, axis_node, 1] for qj in qs])                # (3, n_f) dust-cap excursion
drive_txt = ", ".join(f"{n} {abs(a):.2g} N∠{np.degrees(np.angle(a)):.0f}°" for n, a in zip(NAMES, drive))

figs = []
fig, ax = plt.subplots(2, 1, figsize=(13, 10), sharex=True)
for j, (name, c) in enumerate(zip(pts, ["C0", "C1", "C2", "C3"])):
    ax[0].semilogx(f, spl(Pp[:, j]), color=c, lw=1.4 if j == 0 else 0.9, label=name)
for j, (name, c) in enumerate(zip(NAMES, ["C3", "C0", "C2"])):
    ax[1].loglog(f, 1e6 * np.abs(x[j]), color=c, lw=1.2, label=f"{name} driver")
lo = f < 2 * f_qw                                       # line tuning: excursion minimum near f_qw
f_min = f[lo][np.argmin(np.abs(x[1, lo]))] if lo.any() else np.nan
ax[0].set_ylabel(f"SPL at {DIST:.0f} m [dB re 20 µPa RMS]")
ax[0].set_title(f"Column, 3 × HPNEW + folded line (no stuffing) on a rigid ground; drive: {drive_txt}",
                fontsize=10)
ax[1].set_ylabel("dust-cap excursion [µm]")
ax[1].set_xlabel("f [Hz]")
ax[1].set_title(f"Excursion of each driver; middle-driver minimum at {f_min:.0f} Hz "
                f"(line quarter-wave estimate {f_qw:.0f} Hz)", fontsize=10)
for a in ax:
    a.axvline(f_qw, color="0.7", lw=0.8, ls=":")
    a.grid(True, which="both", alpha=0.3)
    a.legend(fontsize=8)
fig.tight_layout()
figs.append(fig)

# directivity: vertical plane x = X0 (0° = front axis, 90° = up, 180° = behind), horizontal plane
av = np.deg2rad(np.arange(-20, 201, 2))
ah = np.deg2rad(np.arange(-180, 181, 2))
Pv = pressure(src + DIST * np.column_stack([0 * av, np.sin(av), np.cos(av)]))
Ph = pressure(src + DIST * np.column_stack([np.sin(ah), 0 * ah, np.cos(ah)]))
ref = spl(Pp[:, 0])[:, None]
fig, axs = plt.subplots(1, 2, figsize=(15, 5.5))
for a, Pm, ang, name in ((axs[0], Pv, av, "vertical (90° = up, 180° = behind)"),
                         (axs[1], Ph, ah, "horizontal")):
    im = a.pcolormesh(f, np.rad2deg(ang), (spl(Pm) - ref).T, cmap="viridis", vmin=-30, vmax=6,
                      shading="auto", rasterized=True)
    a.set_xscale("log")
    a.set_xlabel("f [Hz]")
    a.set_ylabel("angle [deg]")
    a.set_title(f"Column directivity, {name}, SPL re front axis at {DIST:.0f} m", fontsize=9)
    plt.colorbar(im, ax=a, label="dB")
fig.tight_layout()
figs.append(fig)

fig, axs = plt.subplots(1, 2, figsize=(15, 7.5), subplot_kw=dict(projection="polar"))
for k, fp in enumerate(POLAR_F):
    i = int(np.argmin(np.abs(f - fp)))
    if abs(f[i] - fp) > 15:
        continue
    axs[0].plot(av, spl(Pv[i]) - ref[i, 0], color=f"C{k}", label=f"{f[i]:.0f} Hz")
    axs[1].plot(ah, spl(Ph[i]) - ref[i, 0], color=f"C{k}")
for a, ttl in zip(axs, ("vertical (0° = front axis, 90° = up)", "horizontal (0° = front axis)")):
    a.set_theta_zero_location("E" if a is axs[0] else "N")
    a.set_theta_direction(1 if a is axs[0] else -1)
    a.set_rlim(-30, 6)
    a.set_title(f"Directivity at {DIST:.0f} m, {ttl}, dB re front axis", fontsize=9)
axs[0].legend(fontsize=8, loc="lower left", bbox_to_anchor=(-0.15, -0.2), ncol=3)
fig.tight_layout()
figs.append(fig)

# per driver: m ≥ 1 share of the kinetic energy (unit modal mass: E ∝ Σ |ω q|²), phase re middle
fig, ax = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
for j, (name, c) in enumerate(zip(NAMES, ["C3", "C0", "C2"])):
    E = np.abs(qs[j]) ** 2
    ax[0].semilogx(f, 100 * E[:, drv.m >= 1].sum(1) / E.sum(1), color=c, lw=1.2, label=f"{name} driver")
    ax[1].semilogx(f, np.degrees(np.angle(x[j] / x[1])), color=c, lw=1.2, label=f"{name} driver")
ax[0].set_ylabel("m ≥ 1 share of the driver's kinetic energy [%]")
ax[0].set_title("Non-axisymmetric motion (rocking m = 1, …): only driven by the non-uniform air load", fontsize=10)
ax[1].set_ylabel("excursion phase re middle driver [deg]")
ax[1].set_xlabel("f [Hz]")
for a in ax:
    a.axvline(f_qw, color="0.7", lw=0.8, ls=":")
    a.grid(True, which="both", alpha=0.3)
    a.legend(fontsize=8)
fig.tight_layout()
figs.append(fig)

tmp, target = out_dir / "column_response.pdf.tmp", out_dir / "column_response.pdf"
with PdfPages(tmp) as pdf:
    for fg in figs:
        pdf.savefig(fg, bbox_inches="tight", dpi=130)
        plt.close(fg)
os.replace(tmp, target)
print(f"✓ {target}   (middle-driver excursion minimum {f_min:.0f} Hz, quarter-wave estimate {f_qw:.0f} Hz)")
