"""
Loading_script.py — the structural frequency sweep (Running_script.py first).

What this produces (in Results/Figures/)
-----------------------------------------
  - impulse_response(_db).pdf : impulse response of the axial displacement at a probe
  - meca_sweep.pdf            : |u|, |v|, |a| vs frequency at the probe
  - deformed_<f>Hz.pdf        : displacement magnitude on the deformed mesh
  - membrane_<f>Hz.pdf        : the same on the membrane only
  - surround_<f>Hz.pdf        : the same on the surround only
  - interface_<f>Hz.pdf       : deformed profile of the fluid-structure interface
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import loudpy as lp
from loudpy.Plotter import plot_field, plot_meca_sweep, plot_impulse_response, STYLE

lp.plot.style.use()                       # loudpy figure style (fonts, colours, SPL in dB RMS)

# ── Paths ──────────────────────────────────────────────────────────────────────
in_path = "LoudPy_exemples/Frequency_study/Meca/Results/Files/sweep.h5"
out_dir = Path("LoudPy_exemples/Frequency_study/Meca/Results/Figures")
out_dir.mkdir(parents=True, exist_ok=True)

res = lp.load(in_path)
print(res)

# ── Axial displacement at a probe: u, v = jωu, a = −ω²u ────────────────────────
probe = (0.0, 0.0)                              # (r, z) [m]
f = res.frequencies
u = res.probe(probe, "u")[:, 1]                 # u_z
w = 2 * np.pi * f
v, a = 1j * w * u, -w ** 2 * u

fig, _, _ = plot_impulse_response(f, u, t_max=6000e-3, title="Cone tip impulse response")
fig.savefig(out_dir / "impulse_response.pdf", bbox_inches="tight")
fig, _, _ = plot_impulse_response(f, u, t_max=600e-3, db=True, title="Cone tip impulse response — decay")
fig.savefig(out_dir / "impulse_response_db.pdf", bbox_inches="tight")

fig = plot_meca_sweep(f, u, v, a, title="Cone tip mechanical sweep")
fig.savefig(out_dir / "meca_sweep.pdf", bbox_inches="tight")

# ── Fields at one frequency ────────────────────────────────────────────────────
view = res.field("u", f=1000.0)                 # the entry nearest 1 kHz
fk = view.coordinate
mesh = view.mesh

# deformed mesh: displaced by scale × Re(u), undeformed mesh in grey
scale = 1e4
deformed = mesh.coords + scale * view.values.real
ax = plot_field(deformed, mesh.triangles(), view.magnitude(),
                title=f"Deformed |u| @ {fk:.0f} Hz  (scale ×{scale:g})", **STYLE["meca"])
ax.triplot(mesh.coords[:, 0], mesh.coords[:, 1], mesh.triangles(), color="k", lw=0.2, alpha=0.3)
ax.figure.savefig(out_dir / f"deformed_{fk:.0f}Hz.pdf", bbox_inches="tight")

for region in ("membranne", "surround"):
    ax = lp.plot.field(view.restrict(region), title=f"{region} |u| @ {fk:.0f} Hz")
    ax.figure.savefig(out_dir / f"{region.replace('membranne', 'membrane')}_{fk:.0f}Hz.pdf", bbox_inches="tight")

# rest shape (dots) and scaled deformed shape (red) of the cone / air boundary
ax = lp.plot.deformed_boundary(view, res.boundary("interface_acou_meca_front"), scale=scale,
                               title=f"Interface deformation @ {fk:.0f} Hz")
ax.figure.savefig(out_dir / f"interface_{fk:.0f}Hz.pdf", bbox_inches="tight")

plt.show()
