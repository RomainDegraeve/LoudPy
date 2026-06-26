"""
Loading_script.py — Post-process a multitone large-signal time simulation.

Reads the HDF5 file produced by Running_script.py and produces:
  - uva_time.pdf       : displacement, velocity, acceleration vs time at a probe node
  - uva_fft.pdf        : spectra of u, v, a over the steady-state block (after the ramp)
  - field_<t>s.pdf     : |u| field on the full mesh at the end of the ramp
  - field_membrane.pdf : same field restricted to the membrane sub-domain
  - animation.mp4      : animation of the deforming mesh over the steady-state block
"""
import sys
sys.path.insert(0, "/Users/romaindegraeve/Documents/Master_IMDEA/LoudPy_GITUB/LoudPy/src")

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from loudpy.Files_Loader import TimeReader, Domain
from loudpy.Plotter import (plot_field, plot_uva_time, plot_uva_fft,
                             animate_field, STYLE)

# ── Paths ────────────────────────────────────────────────────────────────────────
h5_path = Path("LoudPy_exemples/Time_study/Multitone/Results/time_study_large_signal_1N.h5")
out_dir = Path("LoudPy_exemples/Time_study/Multitone/Results/Figures")
out_dir.mkdir(parents=True, exist_ok=True)

# ── Inspect ──────────────────────────────────────────────────────────────────────
with TimeReader(h5_path) as r:
    r.inspect()   # prints the stored runs, mesh ids and available fields

# ── Load the time run ─────────────────────────────────────────────────────────────
# A TimeRun bundles the full time histories on the mechanical mesh:
#   .time (n_t,)   .U / .V / .A (n_t, n_nodes, 2)   displacement / velocity / acceleration
# t_ramp marks the end of the excitation ramp — everything after it is steady state.
with TimeReader(h5_path) as r:
    kind   = r.kinds()[0]         # "meca"
    run    = r.load_run(kind, 0)  # first (and only) run
    mesh   = r.mesh(run.mesh_id, Domain.MECA)
    t_ramp      = float(r._h5.attrs["t_ramp"])         if "t_ramp"       in r._h5.attrs else None
    freqs_tones = np.array(r._h5.attrs["freqs_tones"]) if "freqs_tones" in r._h5.attrs else None

time = run.time    # (n_t,)              time vector  [s]
U    = run.U       # (n_t, n_nodes, 2)   displacement [m]
V    = run.V       # (n_t, n_nodes, 2)   velocity     [m/s]
A    = run.A       # (n_t, n_nodes, 2)   acceleration [m/s²]

# ── Probe node ────────────────────────────────────────────────────────────────────
# Pick the mesh node closest to the requested coordinates and read its axial (uy)
# response.  (0, 0) is the cone tip — adjust target to probe elsewhere.
target    = np.array([0.0, 0.0])   # (x, y) of the probe point [m]
probe_row = int(np.argmin(np.linalg.norm(mesh.coords - target, axis=1)))
print(f"Probe node {probe_row} at {mesh.coords[probe_row]} m")

u_probe = U[:, probe_row, 1]   # uy: axial displacement
v_probe = V[:, probe_row, 1]   # uy: axial velocity
a_probe = A[:, probe_row, 1]   # uy: axial acceleration

# ── U/V/A time signals ─────────────────────────────────────────────────────────────
# t_ramp is passed so the ramp interval can be shaded on the plot.
fig = plot_uva_time(time, u_probe, v_probe, a_probe,
                    title="Cone tip — time response",
                    t_ramp=t_ramp)
fig.savefig(out_dir / "uva_time.pdf", bbox_inches="tight")

# ── U/V/A spectra ──────────────────────────────────────────────────────────────────
# The FFT is taken on the steady-state block only (t >= t_ramp); the excitation
# tone frequencies are marked on the plot for reference.
result = plot_uva_fft(time, u_probe, v_probe, a_probe,
                      t_start=t_ramp,
                      excitation_freqs=freqs_tones,
                      title="Cone tip — FFT")
fig_fft = result[0]
fig_fft.savefig(out_dir / "uva_fft.pdf", bbox_inches="tight")

# ── Field snapshot at the end of the ramp ──────────────────────────────────────────
# Locate the time index closest to t_ramp and map the displacement magnitude
# |u| = sqrt(ux² + uy²) over every mesh node.
t_snap = t_ramp if t_ramp is not None else time[-1]
t_idx  = int(np.argmin(np.abs(time - t_snap)))
u_mag  = np.linalg.norm(np.abs(U[t_idx]), axis=1)   # (n_nodes,) [m]

ax = plot_field(mesh.coords, mesh.tris, u_mag,
                title=f"|u| at t = {time[t_idx]:.4f} s", **STYLE["meca"])
ax.figure.savefig(out_dir / f"field_t{time[t_idx]:.3f}s.pdf", bbox_inches="tight")

# ── Membrane sub-domain snapshot ───────────────────────────────────────────────────
# Restrict the same snapshot to the membrane physical group.  sub_mesh.tags are
# global node tags, so map them back to rows of the full-mesh arrays.
with TimeReader(h5_path) as r:
    sub_mesh = r.subdomain_mesh("submeca_membranne")

t2r      = {int(t): i for i, t in enumerate(mesh.tags)}        # node tag -> row index
sub_rows = np.array([t2r[int(t)] for t in sub_mesh.tags if int(t) in t2r])
u_sub    = np.linalg.norm(np.abs(U[t_idx][sub_rows]), axis=1)  # (n_sub_nodes,) [m]

ax = plot_field(sub_mesh.coords, sub_mesh.tris, u_sub,
                title=f"Membrane |u| at t = {time[t_idx]:.4f} s", **STYLE["meca"])
ax.figure.savefig(out_dir / "field_membrane.pdf", bbox_inches="tight")

# ── Animation (requires pyvista + imageio) ─────────────────────────────────────────
# Each frame displaces the mesh by scale_factor × U.  The large-signal
# displacements already reach a visible fraction of the mesh size, so scale_factor
# ≈ 1 shows true proportions — raise it only to exaggerate the motion (too large a
# value throws the mesh off-screen).  Starting at t_ramp skips the ramp and animates
# the steady-state response only.
animate_field(
    mesh.coords, mesh.tris,
    U_xy           = U,        # (n_t, n_nodes, 2)
    time            = time,
    scale_factor    = 1,        # display scale factor (1 = true proportions)
    start_time      = 1.8,
    fps             = 30,
    target_duration = 10.0,
    save_path       = out_dir / "animation.mp4",
)

plt.show()
