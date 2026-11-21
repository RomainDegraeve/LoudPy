"""
Loading_script.py — the multitone time simulation (Running_script.py first).

What this produces
------------------
Results/Figures/
  - uva_time.pdf        : axial displacement, velocity, acceleration at the cone tip over time
  - uva_spectrum.pdf    : their spectra over the steady-state block (after the ramp), tones marked
  - field_t<t>s.pdf     : |u| on the deformed structure (true proportions) at the end of the ramp
  - field_membrane.pdf  : the same on the membrane
  - animation.mp4       : the moving structure over the steady-state block
Results/Files/
  - tip.csv             : u_z, v_z, a_z at the cone tip over time
"""
from pathlib import Path

import numpy as np

import loudpy as lp

lp.plot.style.use()                       # the loudpy figure style for this script

here = Path("LoudPy_exemples/Time_study/Multitone/Results")
res = lp.load(here / "Files/time.h5")
print(res)
t_ramp = res.meta.get("t_ramp")                          # steady state after the ramp
tones = np.asarray(res.meta.get("freqs_tones", []), float)
u, v, a = (res.probe_signal("tip", f, "z") for f in ("u", "v", "a"))    # full rate (lp.Output probe)
np.savetxt(here / "Files/tip.csv", np.column_stack([u.coords["t"], u, v, a]), delimiter=",",
           header="t [s],u_z [m],v_z [m/s],a_z [m/s²]")

fig, ax = lp.plot.figure("tall", 3, 1, sharex=True)
for axi, d in zip(ax, (u, v, a)):
    d.plot(ax=axi)
    if t_ramp is not None:
        axi.axvline(t_ramp, color="0.5", lw=0.6, ls="--")
    axi.set_xlabel("")
ax[-1].set_xlabel(lp.plot.style.LABELS["t"])
lp.plot.save(fig, here / "Figures/uva_time.pdf")

fig, ax = lp.plot.figure("tall", 3, 1, sharex=True)
for axi, d in zip(ax, (u, v, a)):
    d.spectrum(start=t_ramp, window=None if len(tones) else "hann").db().plot(ax=axi, logx=False)
    for f in tones:
        axi.axvline(f, color="0.6", lw=0.5, ls=":")
    axi.set_xlim(0, 3 * tones.max() if len(tones) else None)
    axi.set_xlabel("")
ax[-1].set_xlabel(lp.plot.style.LABELS["f"])
lp.plot.save(fig, here / "Figures/uva_spectrum.pdf")

view = res.field("u", t=t_ramp if t_ramp is not None else res.times[-1])
lp.plot.save(view.plot(deformed=True, factor=1.0).figure, here / f"Figures/field_t{view.coordinate:.3f}s.pdf")
lp.plot.save(view.restrict("membranne").plot(deformed=True, factor=1.0).figure, here / "Figures/field_membrane.pdf")
lp.plot.animate(res, here / "Figures/animation.mp4", start=t_ramp, factor=1.0, duration=10.0)
print(f"✓ figures → {here / 'Figures'}")
