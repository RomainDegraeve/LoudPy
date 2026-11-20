"""
Loading_script.py — the structural frequency sweep (Running_script.py first).

What this produces
------------------
Results/Figures/
  - uva.pdf                    : |u_z|, |v_z|, |a_z| at the cone tip over frequency
  - impulse_response.pdf       : impulse response of u_z (inverse FFT of the sweep), waveform and decay
  - deformed_<f>Hz.pdf         : |u| on the deformed structure at 1 kHz
  - membrane / surround_<f>Hz  : the same on one part
  - interface_<f>Hz.pdf        : deformed profile of the fluid-structure interface
Results/Files/
  - tip_displacement.csv       : u_z at the cone tip (re, im) over frequency
"""
from pathlib import Path

import loudpy as lp

here = Path("LoudPy_exemples/Frequency_study/Meca/Results")
res = lp.load(here / "Files/sweep.h5")
print(res)

# ── cone tip: u, v = jωu, a = −ω²u ─────────────────────────────────────────────
u = res.displacement(at=(0.0, 0.0), component="z")
u.export(here / "Files/tip_displacement.csv")
fig, ax = lp.plot.figure("tall", 3, 1, sharex=True)
u.plot(ax=ax[0])
u.velocity().plot(ax=ax[1])
u.acceleration().plot(ax=ax[2])
for a in ax[:2]:
    a.set_xlabel("")
lp.plot.save(fig, here / "Figures/uva.pdf")

h = u.impulse_response(df=1.0)
fig, ax = lp.plot.figure("wide", 1, 2)
h.sel(t=slice(0, 0.05)).plot(ax=ax[0])                        # waveform, first 50 ms
h.sel(t=slice(0, 0.6)).db(ref=abs(h.values).max()).plot(ax=ax[1])     # decay [dB re peak]
ax[1].set_ylim(-120, 5)
lp.plot.save(fig, here / "Figures/impulse_response.pdf")

# ── fields at 1 kHz ────────────────────────────────────────────────────────────
view = res.field("u", f=1000.0)
fk = view.coordinate
lp.plot.save(view.plot(deformed=True).figure, here / f"Figures/deformed_{fk:.0f}Hz.pdf")
for region, name in (("membranne", "membrane"), ("surround", "surround")):
    lp.plot.save(view.restrict(region).plot(deformed=True).figure, here / f"Figures/{name}_{fk:.0f}Hz.pdf")
lp.plot.save(lp.plot.boundary(view, res.boundary("interface_acou_meca_front")).figure,
             here / f"Figures/interface_{fk:.0f}Hz.pdf")
print(f"✓ figures → {here / 'Figures'}")
