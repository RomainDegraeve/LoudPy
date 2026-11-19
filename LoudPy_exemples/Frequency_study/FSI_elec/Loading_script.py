"""
Loading_script.py — the voltage-driven coupled FSI sweep (Running_script.py first).

The sweep has one block per mesh; lp.load returns them as one result and the probes find the
nearest node on each block's mesh.

What this produces (in Results/Figures/)
-----------------------------------------
  - spl_sweep_angles.pdf            : SPL and phase at R = 0.3 m, a few angles, with the
                                      electrical impedance |Z| = |U / I_coil|
  - spl_angle_map(_norm).pdf        : directivity map (frequency × angle), absolute / normalised
  - meca_sweep.pdf                  : |u|, |v|, |a| vs frequency at the cone tip
  - fields_<f>Hz.pdf                : SPL and |u| maps at a few frequencies
  - surround_<f>Hz.pdf, interface_<f>Hz.pdf : surround field and interface profile (last of them)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import loudpy as lp
from loudpy.Plotter import (plot_fields_grid, plot_meca_sweep, plot_spl_angle_map, plot_spl_sweep_multi,
                            STYLE)

lp.plot.style.use()                       # loudpy figure style (fonts, colours, SPL in dB RMS)

in_path = "LoudPy_exemples/Frequency_study/FSI_elec/Results/Files/sweep.h5"
out_dir = Path("LoudPy_exemples/Frequency_study/FSI_elec/Results/Figures")
out_dir.mkdir(parents=True, exist_ok=True)

res = lp.load(in_path)
print(res)
f = res.frequencies

# ── SPL at R = 0.3 m (angles from the axis) ────────────────────────────────────
R = 0.3


def point(theta_deg):
    """(r, z) at distance R from the origin, theta from the z axis."""
    th = np.radians(theta_deg)
    return R * np.sin(th), R * np.cos(th)


curves = {rf"$\theta$ = {a}$^\circ$": res.probe(point(a), "p") for a in (0, 10, 30, 45, 60)}
Z = res.extra("U") / res.extra("I_coil")                 # drive voltage / coil current, per frequency
fig = plot_spl_sweep_multi(f, curves, title=rf"SPL and phase at $R$ = {R} m", Z_elec=Z)
lp.plot.style.tidy(fig).savefig(out_dir / "spl_sweep_angles.pdf", bbox_inches="tight")

# ── directivity map: frequency × angle → SPL ──────────────────────────────────
angles = np.linspace(0, 80, 50)
p_map = np.array([res.probe(point(a), "p") for a in angles])           # (n_angle, n_freq)
fig = plot_spl_angle_map(f, angles, p_map, title=rf"Directivity at $R$ = {R} m")
lp.plot.style.tidy(fig).savefig(out_dir / "spl_angle_map.pdf", bbox_inches="tight")
fig = plot_spl_angle_map(f, angles, p_map, normalize=True, dynamic_range=30,
                         levels=[-24, -18, -12, -9, -6, -3, -1],
                         title=rf"Directivity at $R$ = {R} m (normalised to on-axis)")
lp.plot.style.tidy(fig).savefig(out_dir / "spl_angle_map_norm.pdf", bbox_inches="tight")

# ── cone tip: u, v = jωu, a = −ω²u ─────────────────────────────────────────────
u = res.probe((0.0, 0.0), "u")[:, 1]
w = 2 * np.pi * f
fig = plot_meca_sweep(f, u, 1j * w * u, -w ** 2 * u, title="Cone tip - mechanical sweep")
lp.plot.style.tidy(fig).savefig(out_dir / "meca_sweep.pdf", bbox_inches="tight")

# ── field maps at a few frequencies (SPL, RMS as everywhere) ──
for f_target in (20, 1000, 5000, 15000):
    p, u = res.field("p", f=f_target), res.field("u", f=f_target)
    fk = p.coordinate
    fig = plot_fields_grid([
        dict(coords=p.mesh.coords, tris=p.triangles(), values=p.spl(),
             title=f"SPL @ {fk:.0f} Hz", **STYLE["acou"], vmin=20, xlim=(0, 0.30), ylim=(-0.1, 0.5)),
        dict(coords=u.mesh.coords, tris=u.triangles(), values=u.magnitude(),
             title=f"|u| @ {fk:.0f} Hz", **STYLE["meca"]),
    ], ncols=2)
    lp.plot.style.tidy(fig).savefig(out_dir / f"fields_{fk:.0f}Hz.pdf", bbox_inches="tight")

# ── surround and interface at the last of them ────────────────────────────────
ax = lp.plot.field(u.restrict("surround"), title=f"Surround |u| @ {fk:.0f} Hz")
lp.plot.style.tidy(ax.figure).savefig(out_dir / f"surround_{fk:.0f}Hz.pdf", bbox_inches="tight")
ax = lp.plot.deformed_boundary(u, res.boundary("interface_acou_meca_front"), scale=1e4,
                               title=f"Interface deformation @ {fk:.0f} Hz")
lp.plot.style.tidy(ax.figure).savefig(out_dir / f"interface_{fk:.0f}Hz.pdf", bbox_inches="tight")

plt.show()
