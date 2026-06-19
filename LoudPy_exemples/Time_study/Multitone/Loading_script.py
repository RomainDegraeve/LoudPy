import sys
sys.path.insert(0, "/Users/romaindegraeve/Documents/Master_IMDEA/LoudPy_GITUB/LoudPy/src")

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from loudpy.Files_Loader import TimeReader, Domain
from loudpy.Plotter import (plot_field, plot_uva_time, plot_uva_fft,
                             animate_field, STYLE)

# ── paths ─────────────────────────────────────────────────────────────────────
h5_path = Path("LoudPy_exemples/Time_study/Multitone/Results/time_study_large_signal.h5")
out_dir = Path("LoudPy_exemples/Time_study/Multitone/Results/Figures")
out_dir.mkdir(parents=True, exist_ok=True)

# ── inspect file ──────────────────────────────────────────────────────────────
with TimeReader(h5_path) as r:
    r.inspect()

# ── load run ──────────────────────────────────────────────────────────────────
with TimeReader(h5_path) as r:
    kind   = r.kinds()[0]          # "meca"
    run    = r.load_run(kind, 0)  # TimeRun: .time .U .V .A .mesh_id
    mesh   = r.mesh(run.mesh_id, Domain.MECA)
    t_ramp      = float(r._h5.attrs["t_ramp"])       if "t_ramp"       in r._h5.attrs else None
    freqs_tones = np.array(r._h5.attrs["freqs_tones"]) if "freqs_tones" in r._h5.attrs else None

time = run.time    # (n_t,)
U    = run.U       # (n_t, n_nodes, 2)
V    = run.V
A    = run.A

# ── probe node (cone tip — adjust coords) ─────────────────────────────────────
target    = np.array([0.0, 0.0])
probe_row = int(np.argmin(np.linalg.norm(mesh.coords - target, axis=1)))
print(f"Probe node {probe_row} at {mesh.coords[probe_row]}")

u_probe = U[:, probe_row, 1]   # uy axial
v_probe = V[:, probe_row, 1]
a_probe = A[:, probe_row, 1]

# ── UVA time signals ──────────────────────────────────────────────────────────
fig = plot_uva_time(time, u_probe, v_probe, a_probe,
                    title="Cone tip - time response",
                    t_ramp=t_ramp)
fig.savefig(out_dir / "uva_time.pdf", bbox_inches="tight")

# ── FFT analysis (steady-state block only) ────────────────────────────────────
result = plot_uva_fft(time, u_probe, v_probe, a_probe,
                      t_start=t_ramp,
                      excitation_freqs=freqs_tones,
                      title="Cone tip - FFT")
fig_fft = result[0]
fig_fft.savefig(out_dir / "uva_fft.pdf", bbox_inches="tight")

# ── field snapshot at end of ramp ─────────────────────────────────────────────
t_snap = t_ramp if t_ramp is not None else time[-1]
t_idx  = int(np.argmin(np.abs(time - t_snap)))
u_mag  = np.linalg.norm(np.abs(U[t_idx]), axis=1)   # (n_nodes,)

ax = plot_field(mesh.coords, mesh.tris, u_mag,
                title=f"|u| at t = {time[t_idx]:.4f} s", **STYLE["meca"])
ax.figure.savefig(out_dir / f"field_t{time[t_idx]:.3f}s.pdf", bbox_inches="tight")

# ── subdomain snapshot (membrane only) ────────────────────────────────────────
with TimeReader(h5_path) as r:
    sub_mesh = r.subdomain_mesh("submeca_membranne")

t2r      = {int(t): i for i, t in enumerate(mesh.tags)}
sub_rows = np.array([t2r[int(t)] for t in sub_mesh.tags if int(t) in t2r])
u_sub    = np.linalg.norm(np.abs(U[t_idx][sub_rows]), axis=1)

ax = plot_field(sub_mesh.coords, sub_mesh.tris, u_sub,
                title=f"Membrane |u| at t = {time[t_idx]:.4f} s", **STYLE["meca"])
ax.figure.savefig(out_dir / "field_membrane.pdf", bbox_inches="tight")

# ── animation (requires pyvista + imageio) ────────────────────────────────────
animate_field(
    mesh.coords, mesh.tris,
    U_xyz           = U,        # (n_t, n_nodes, 2)
    time            = time,
    scale_factor    = 1e5,           # adjust to make deformation visible
    start_time      = t_ramp or 0.0,
    fps             = 30,
    target_duration = 10.0,
    save_path       = out_dir / "animation.mp4",
)

plt.show()
