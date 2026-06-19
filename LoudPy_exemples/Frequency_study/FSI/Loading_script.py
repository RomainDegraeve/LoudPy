"""
Loading_script.py — Post-process a coupled FSI frequency sweep.

The FSI running script saves one HDF5 file per mesh block.  This loader
stitches all blocks together transparently via the all_snapshots() generator.

What this produces (in Results/Figures/)
-----------------------------------------
  - spl_sweep.pdf          : SPL vs frequency at an acoustic probe point
  - meca_sweep.pdf         : |u|, |v|, |a| vs frequency at the cone tip
  - fields_<f>Hz.pdf       : side-by-side SPL and |u| maps at key frequencies
  - surround_<f>Hz.pdf     : surround displacement field at the last target freq
  - interface_<f>Hz.pdf    : deformed fluid-structure interface at that frequency
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from loudpy.Files_Loader import FreqReader, Domain
from loudpy.Plotter import (
    plot_field, plot_fields_grid, plot_spl_sweep, plot_meca_sweep,
    plot_interface_deformed, STYLE,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
results_dir = Path("LoudPy_exemples/Frequency_study/FSI/Results/Files")
out_dir     = Path("LoudPy_exemples/Frequency_study/FSI/Results/Figures")
out_dir.mkdir(parents=True, exist_ok=True)

files = sorted(results_dir.glob("snap_*.h5"))   # all block files in frequency order
if not files:
    raise FileNotFoundError(f"No snap_*.h5 files found in {results_dir}")

# ── Inspect one block ──────────────────────────────────────────────────────────
with FreqReader(files[0]) as r:
    r.inspect()

# ── Helper generator ───────────────────────────────────────────────────────────
def all_snapshots(files):
    """
    Yield (file_path, open_reader, FreqSnapshot) for every snapshot in every
    block file.  The reader context stays open while each snapshot is yielded,
    so r.mesh() can be called safely inside the loop.
    """
    for fpath in files:
        with FreqReader(fpath) as r:
            for snap in r.snapshots():
                yield fpath, r, snap

# ── Build a flat frequency index ───────────────────────────────────────────────
# One pass to collect (freq, fpath, snap) so we can quickly find the snapshot
# closest to any target frequency without re-reading every file.
all_snaps_index = [(snap.f, fpath, snap)
                   for fpath, r, snap in all_snapshots(files)]
all_snaps_index.sort(key=lambda x: x[0])
freqs_all = np.array([x[0] for x in all_snaps_index])

# ── SPL sweep at an acoustic probe point ──────────────────────────────────────
# The acoustic mesh changes between blocks, so the probe row is re-located
# for every snapshot using a nearest-node search.
target_acou  = np.array([0.3, 0.1])   # probe location [m] — adjust as needed
freqs_sweep, p_sweep = [], []

for _fpath, r, snap in all_snapshots(files):
    mesh_acou = r.mesh(snap.mesh_id, Domain.ACOU)
    probe_row = int(np.argmin(np.linalg.norm(
        mesh_acou.coords - target_acou, axis=1)))
    freqs_sweep.append(snap.f)
    p_sweep.append(snap.fields["p_acou"][probe_row])

freqs_sweep = np.array(freqs_sweep)
p_sweep     = np.array(p_sweep)

fig = plot_spl_sweep(freqs_sweep, p_sweep,
                     title=f"SPL at probe {target_acou} m (front field)")
fig.savefig(out_dir / "spl_sweep.pdf", bbox_inches="tight")

# ── Mechanical sweep at the cone tip ──────────────────────────────────────────
# Velocity and acceleration derived from the harmonic displacement:
#   v = jω u,   a = -ω² u   (with ω = 2πf)
target_meca = np.array([0.0, 0.0])   # cone apex [m]
u_sweep, v_sweep, a_sweep = [], [], []

for _fpath, r, snap in all_snapshots(files):
    mesh_meca = r.mesh(snap.mesh_id, Domain.MECA)
    probe_row = int(np.argmin(np.linalg.norm(
        mesh_meca.coords - target_meca, axis=1)))
    omega = 2 * np.pi * snap.f
    u     = snap.fields["u_meca"][probe_row, 1]   # uy: axial displacement
    u_sweep.append(u)
    v_sweep.append(1j * omega * u)
    a_sweep.append(-omega**2 * u)

fig = plot_meca_sweep(freqs_sweep,
                      np.array(u_sweep), np.array(v_sweep), np.array(a_sweep),
                      title="Cone tip - mechanical sweep")
fig.savefig(out_dir / "meca_sweep.pdf", bbox_inches="tight")

# ── Field maps at selected frequencies ────────────────────────────────────────
# For each target, we find the nearest available snapshot and plot
# the acoustic SPL and structural displacement side by side.
target_freqs = [20, 1000, 5000, 15000]   # [Hz]

for f_target in target_freqs:
    _, fpath, snap = min(all_snaps_index, key=lambda x: abs(x[0] - f_target))

    with FreqReader(fpath) as r:
        mesh_acou = r.mesh(snap.mesh_id, Domain.ACOU)
        mesh_meca = r.mesh(snap.mesh_id, Domain.MECA)

        # SPL: 20·log₁₀(|p| / p_ref) with p_ref = 20 µPa
        p_spl = 20 * np.log10(np.abs(snap.fields["p_acou"]) / 20e-6 + 1e-30)
        u_mag = np.linalg.norm(np.abs(snap.fields["u_meca"]), axis=1)

    fig = plot_fields_grid([
        dict(coords=mesh_acou.coords, tris=mesh_acou.tris, values=p_spl,
             title=f"SPL @ {snap.f:.0f} Hz", **STYLE["acou"],
             vmin=20, xlim=(0, 0.30), ylim=(-0.1, 0.5)),
        dict(coords=mesh_meca.coords, tris=mesh_meca.tris, values=u_mag,
             title=f"|u| @ {snap.f:.0f} Hz", **STYLE["meca"]),
    ], ncols=2)
    fig.savefig(out_dir / f"fields_{snap.f:.0f}Hz.pdf", bbox_inches="tight")

# ── Subdomain and interface at the last target frequency ───────────────────────
# fpath and snap still point to the last entry of the loop above.
with FreqReader(fpath) as r:
    # Surround sub-domain displacement
    sub_mesh, u_sub = r.extract_subdomain(snap, "submeca_surround", "u_meca")
    u_sub_mag = np.linalg.norm(np.abs(u_sub), axis=1)

    ax = plot_field(sub_mesh.coords, sub_mesh.tris, u_sub_mag,
                    title=f"Surround |u| @ {snap.f:.0f} Hz", **STYLE["meca"])
    ax.figure.savefig(out_dir / f"surround_{snap.f:.0f}Hz.pdf", bbox_inches="tight")

    # Fluid-structure interface deformed profile
    coords_iface, u_iface = r.extract_interface(
        snap, "interface_acou_meca_front", "u_meca")

ax = plot_interface_deformed(
    coords_iface[:, 0], coords_iface[:, 1],
    u_iface[:, 0], u_iface[:, 1],
    scale=1e4,
    title=f"Interface deformation @ {snap.f:.0f} Hz")
ax.figure.savefig(out_dir / f"interface_{snap.f:.0f}Hz.pdf", bbox_inches="tight")

plt.show()
