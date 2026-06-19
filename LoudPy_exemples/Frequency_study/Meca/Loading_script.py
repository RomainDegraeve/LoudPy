"""
Loading_script.py — Post-process a mechanical frequency sweep.

Reads the HDF5 file produced by Running_script.py and produces:
  - meca_sweep.pdf         : |u|, |v|, |a| vs frequency at a probe node
  - deformed_<f>Hz.pdf     : displacement field on the deformed mesh
  - membrane_<f>Hz.pdf     : same field restricted to the membrane sub-domain
  - surround_<f>Hz.pdf     : same field restricted to the surround sub-domain
  - interface_<f>Hz.pdf    : deformed profile of the fluid-structure interface
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from loudpy.Files_Loader import FreqReader, Domain
from loudpy.Plotter import (
    plot_field, plot_meca_sweep, plot_interface_deformed, STYLE,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
h5_path = Path("LoudPy_exemples/Frequency_study/Meca/Results/Files/snap_f_20-20000Hz.h5")
out_dir = Path("LoudPy_exemples/Frequency_study/Meca/Results/Figures")
out_dir.mkdir(parents=True, exist_ok=True)

# ── Inspect ────────────────────────────────────────────────────────────────────
with FreqReader(h5_path) as r:
    r.inspect()   # prints the list of stored snapshots and available fields

# ── Mechanical sweep at a probe node ──────────────────────────────────────────
# The mesh is fixed across all snapshots, so we load it once.
# Velocity and acceleration are derived from the harmonic displacement via:
#   v = jω u      a = -ω² u      (with ω = 2π f)
target_coords = np.array([0.0, 0.0])   # (x, y) of the probe point [m] — adjust as needed
freqs_sweep, u_sweep, v_sweep, a_sweep = [], [], [], []

with FreqReader(h5_path) as r:
    snaps     = r.snapshots()
    mesh_meca = r.mesh(snaps[0].mesh_id, Domain.MECA)

    # Find the mesh node closest to the requested probe coordinates
    probe_row = int(np.argmin(np.linalg.norm(mesh_meca.coords - target_coords, axis=1)))
    print(f"Probe node {probe_row} at coords {mesh_meca.coords[probe_row]} m")

    for snap in snaps:
        omega = 2 * np.pi * snap.f
        u     = snap.fields["u_meca"][probe_row, 1]   # uy: axial displacement
        freqs_sweep.append(snap.f)
        u_sweep.append(u)
        v_sweep.append(1j * omega * u)
        a_sweep.append(-omega**2 * u)

freqs_sweep = np.array(freqs_sweep)

fig = plot_meca_sweep(freqs_sweep,
                      np.array(u_sweep), np.array(v_sweep), np.array(a_sweep),
                      title="Cone tip — mechanical sweep")
fig.savefig(out_dir / "meca_sweep.pdf", bbox_inches="tight")

# ── Field map at a specific frequency ─────────────────────────────────────────
target_f = 1000.0   # [Hz] — the snapshot closest to this value will be used

with FreqReader(h5_path) as r:
    snaps = r.snapshots()
    idx   = int(np.argmin(np.abs(freqs_sweep - target_f)))
    snap  = snaps[idx]

    mesh_meca = r.mesh(snap.mesh_id, Domain.MECA)
    u_field   = snap.fields["u_meca"]                     # (n_nodes, 2) complex
    u_mag     = np.linalg.norm(np.abs(u_field), axis=1)  # displacement magnitude [m]

    # ── Deformed full mesh ─────────────────────────────────────────────────────
    # The displayed mesh is displaced by scale × Re(u) to make the deformation
    # visible.  The undeformed mesh is overlaid in grey for reference.
    scale      = 1e4   # display scale factor (physical deformations are tiny)
    coords_def = np.column_stack([
        mesh_meca.coords[:, 0] + scale * u_field[:, 0].real,
        mesh_meca.coords[:, 1] + scale * u_field[:, 1].real,
    ])

    ax = plot_field(coords_def, mesh_meca.tris, u_mag,
                    title=f"Deformed |u| @ {snap.f:.0f} Hz  (scale ×{scale:g})",
                    **STYLE["meca"])
    ax.triplot(mesh_meca.coords[:, 0], mesh_meca.coords[:, 1],
               mesh_meca.tris[:, :3], color="k", lw=0.2, alpha=0.3)
    ax.figure.savefig(out_dir / f"deformed_{snap.f:.0f}Hz.pdf", bbox_inches="tight")

    # ── Membrane sub-domain ────────────────────────────────────────────────────
    # extract_subdomain returns a mesh restricted to the named physical group
    # and the field values interpolated onto those nodes only.
    sub_mesh, u_sub = r.extract_subdomain(snap, "submeca_membranne", "u_meca")
    u_sub_mag = np.linalg.norm(np.abs(u_sub), axis=1)

    ax = plot_field(sub_mesh.coords, sub_mesh.tris, u_sub_mag,
                    title=f"Membrane |u| @ {snap.f:.0f} Hz", **STYLE["meca"])
    ax.figure.savefig(out_dir / f"membrane_{snap.f:.0f}Hz.pdf", bbox_inches="tight")

    # ── Surround sub-domain ────────────────────────────────────────────────────
    sub_sur, u_sur = r.extract_subdomain(snap, "submeca_surround", "u_meca")
    u_sur_mag = np.linalg.norm(np.abs(u_sur), axis=1)

    ax = plot_field(sub_sur.coords, sub_sur.tris, u_sur_mag,
                    title=f"Surround |u| @ {snap.f:.0f} Hz", **STYLE["meca"])
    ax.figure.savefig(out_dir / f"surround_{snap.f:.0f}Hz.pdf", bbox_inches="tight")

    # ── Fluid-structure interface ──────────────────────────────────────────────
    # Shows the rest shape (dots) and the scaled deformed shape (red) of the
    # coupling boundary between the cone and the acoustic cavity.
    coords_iface, u_iface = r.extract_interface(
        snap, "interface_acou_meca_front", "u_meca")

    ax = plot_interface_deformed(
        coords_iface[:, 0], coords_iface[:, 1],
        u_iface[:, 0], u_iface[:, 1],
        scale=scale,
        title=f"Interface deformation @ {snap.f:.0f} Hz")
    ax.figure.savefig(out_dir / f"interface_{snap.f:.0f}Hz.pdf", bbox_inches="tight")

plt.show()
