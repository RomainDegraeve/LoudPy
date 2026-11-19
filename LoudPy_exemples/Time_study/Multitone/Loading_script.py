"""
Loading_script.py — the multitone time simulation (Running_script.py first).

What this produces (in Results/Figures/)
-----------------------------------------
  - uva_time.pdf       : axial displacement, velocity, acceleration vs time at a probe
  - uva_fft.pdf        : their spectra over the steady-state block (after the ramp)
  - field_t<t>s.pdf    : |u| at the end of the ramp
  - field_membrane.pdf : the same on the membrane only
  - animation.mp4      : the deforming structure over the steady-state block
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import loudpy as lp
from loudpy.Plotter import plot_uva_time, plot_uva_fft, animate_field

lp.plot.style.use()                       # loudpy figure style (fonts, colours, SPL in dB RMS)

# ── Paths ──────────────────────────────────────────────────────────────────────
in_path = "LoudPy_exemples/Time_study/Multitone/Results/Files/time.h5"
out_dir = Path("LoudPy_exemples/Time_study/Multitone/Results/Figures")
out_dir.mkdir(parents=True, exist_ok=True)

res = lp.load(in_path)
print(res)
t_ramp = res.meta.get("t_ramp")                  # end of the ramp: steady state after it
freqs_tones = np.asarray(res.meta.get("freqs_tones", []), float)

# ── Axial response at a probe (cone tip) ───────────────────────────────────────
probe = (0.0, 0.0)
t = res.times
u, v, a = (np.real(res.probe(probe, name)[:, 1]) for name in ("u", "v", "a"))

fig = plot_uva_time(t, u, v, a, title="Cone tip — time response", t_ramp=t_ramp)
lp.plot.style.tidy(fig).savefig(out_dir / "uva_time.pdf", bbox_inches="tight")

# FFT over the steady-state block only, the tone frequencies marked
spectra = plot_uva_fft(t, u, v, a, t_start=t_ramp, excitation_freqs=freqs_tones, title="Cone tip — FFT")
spectra.fig.savefig(out_dir / "uva_fft.pdf", bbox_inches="tight")

# ── |u| at the end of the ramp, whole structure and membrane ──────────────────
view = res.field("u", t=t_ramp if t_ramp is not None else t[-1])
ax = lp.plot.field(view, title=f"|u| at t = {view.coordinate:.4f} s")
lp.plot.style.tidy(ax.figure).savefig(out_dir / f"field_t{view.coordinate:.3f}s.pdf", bbox_inches="tight")
ax = lp.plot.field(view.restrict("membranne"), title=f"Membrane |u| at t = {view.coordinate:.4f} s")
lp.plot.style.tidy(ax.figure).savefig(out_dir / "field_membrane.pdf", bbox_inches="tight")

# ── Animation (pyvista + imageio) ──────────────────────────────────────────────
# scale_factor 1 = true proportions (the large-signal motion is already visible);
# starting at t_ramp animates the steady-state response only.
mesh = res.mesh("solid")
animate_field(mesh.coords, mesh.triangles(), U_xy=np.real(res.blocks[0].fields["u"]), time=t,
              scale_factor=1, start_time=t_ramp or 0.0, fps=30, target_duration=10.0,
              save_path=out_dir / "animation.mp4")

plt.show()
