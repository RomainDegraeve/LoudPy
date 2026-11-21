"""
plot.py — the coupled structure + air sweep, 2.83 V on the coil (run.py first).

The sweep has one result block per mesh (the air mesh follows the wavelength); lp.load returns
them as one result and every evaluation runs over all of them.

What this produces
------------------
Results/Figures/
  - spl_angles.pdf            : SPL and phase at R = 0.3 m for a few angles from the axis
  - directivity_map(_norm).pdf: SPL over frequency × angle at R = 0.3 m, absolute / re on axis
  - directivity_polar.pdf     : polar SPL (re on axis) at a few frequencies
  - cone_tip.pdf              : |u_z|, |v_z|, |a_z| at the cone tip
  - impedance.pdf             : electrical input impedance |Z| = |U / I_coil| and its phase
  - fields_<f>Hz.pdf          : SPL in the air and |u| on the deformed structure at a few frequencies
  - surround / interface_<f>Hz.pdf : the surround and the interface profile (last of them)
Results/Files/
  - directivity_0.3m.csv      : the directivity map as a table (frequency × angle)
  - impedance.csv             : Z over frequency
  - thiele_small.csv          : the small-signal (Thiele–Small) parameters fitted on the impedance
  - u_<f>Hz.html              : the deformed structure in 3D (revolved), interactive, one file
"""
from pathlib import Path

import numpy as np

import loudpy as lp

HERE = Path(__file__).resolve().parent               # this example's folder
EXAMPLES = HERE.parent                               # LoudPy_exemples/

lp.plot.style.use()                       # the loudpy figure style for this script

here = HERE / "Results"
res = lp.load(here / "Files/sweep.h5")
print(res)
R = 0.3                                                   # [m] from the origin

# ── SPL and phase at R for a few angles ────────────────────────────────────────
few = lp.Arc(R, [0, 10, 30, 45, 60])
fig, ax = lp.plot.figure("tall", 2, 1, sharex=True)
res.spl(at=few).plot(ax=ax[0])
res.pressure(at=few).phase(deg=True).plot(ax=ax[1], legend=False)
ax[0].set_xlabel("")
lp.plot.save(fig, here / "Figures/spl_angles.pdf")

# ── directivity: frequency × angle ─────────────────────────────────────────────
directivity = res.spl(at=lp.Arc(R, np.linspace(0, 80, 41)))
directivity.export(here / "Files/directivity_0.3m.csv")
lp.plot.save(directivity.plot(kind="map").figure, here / "Figures/directivity_map.pdf")
lp.plot.save(directivity.normalize(angle=0).plot(kind="map", contours=[-12, -6, -3]).figure,
             here / "Figures/directivity_map_norm.pdf")
lp.plot.save(lp.plot.directivity(res, f=[1000, 3000, 6000, 12000], radius=R, angles=np.arange(0, 91, 2)).figure,
             here / "Figures/directivity_polar.pdf")

# ── cone tip: u, v = jωu, a = −ω²u ─────────────────────────────────────────────
u = res.displacement(at=(0.0, 0.0), component="z")
fig, ax = lp.plot.figure("tall", 3, 1, sharex=True)
for a, d in zip(ax, (u, u.velocity(), u.acceleration())):
    d.plot(ax=a)
    a.set_xlabel("")
ax[-1].set_xlabel(lp.plot.style.LABELS["f"])
lp.plot.save(fig, here / "Figures/cone_tip.pdf")

# ── electrical input impedance ─────────────────────────────────────────────────
Z = res.impedance()
Z.export(here / "Files/impedance.csv")
fig, ax = lp.plot.figure("tall", 2, 1, sharex=True)
Z.plot(ax=ax[0])
Z.phase(deg=True, unwrap=False).plot(ax=ax[1])
ax[0].set_xlabel("")
lp.plot.save(fig, here / "Figures/impedance.pdf")

# ── Thiele–Small parameters (impedance fit; Sd and Vas from the radiating boundary) ──
ts = res.thiele_small(on="interface_acou_meca_front")
units = dict(Re="Ω", Le="H·s^(1−n)", n="", Bl="T·m", fs="Hz", Mms="kg", Cms="m/N", Rms="kg/s", Qms="", Qes="",
             Qts="", Sd="m²", Vas="m³")
(here / "Files/thiele_small.csv").write_text("parameter,value,unit\n" +
                                            "".join(f"{k},{v:.6g},{units.get(k, '')}\n" for k, v in ts.items()))
print("Thiele–Small: " + ", ".join(f"{k} = {v:.4g}" for k, v in ts.items()))

# ── fields at a few frequencies ────────────────────────────────────────────────
for f in (20, 1000, 5000, 15000):
    p, u = res.field("p", f=f), res.field("u", f=f)
    fig, ax = lp.plot.figure("wide", 1, 2)
    p.plot(ax=ax[0], xlim=(0, 0.3), ylim=(-0.1, 0.5))
    u.plot(ax=ax[1], deformed=True)
    lp.plot.save(fig, here / f"Figures/fields_{p.coordinate:.0f}Hz.pdf")
    u.view3d(sweep=1.5 * np.pi).save(here / f"Files/u_{u.coordinate:.0f}Hz.html",
                                     title=f"|u| at {u.coordinate:.0f} Hz (2.83 V)")

# ── surround and interface at the last of them ────────────────────────────────
lp.plot.save(u.restrict("surround").plot(deformed=True).figure, here / f"Figures/surround_{u.coordinate:.0f}Hz.pdf")
lp.plot.save(lp.plot.boundary(u, res.boundary("interface_acou_meca_front")).figure,
             here / f"Figures/interface_{u.coordinate:.0f}Hz.pdf")
print(f"✓ figures → {here / 'Figures'}")
