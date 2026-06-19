import sys
sys.path.insert(0, "/Users/romaindegraeve/Documents/Master_IMDEA/LoudPy_GITUB/LoudPy/src")

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from loudpy.Files_Loader import FreqReader, Domain
from loudpy.Plotter import (plot_field, plot_fields_grid, plot_meca_sweep, STYLE)

# ── paths ─────────────────────────────────────────────────────────────────────
h5_path = Path("LoudPy_exemples/Frequency_study/Meca/Results/Files/snap_f_20.000000000000004-20000.000000000004Hz.h5")
out_dir = Path("LoudPy_exemples/Frequency_study/Meca/Results/Figures")
out_dir.mkdir(parents=True, exist_ok=True)

# ── inspect ───────────────────────────────────────────────────────────────────
with FreqReader(h5_path) as r:
    r.inspect()

# ── load mesh + probe once, then sweep all snapshots ─────────────────────────
freqs_sweep, u_sweep, v_sweep, a_sweep = [], [], [], []
probe_row = None

with FreqReader(h5_path) as r:
    snaps = r.snapshots()   # list of all 400 FreqSnapshots

    # mesh is fixed — load once from first snapshot
    mesh_meca = r.mesh(snaps[0].mesh_id, Domain.MECA)
    target    = np.array([0.0, 0.0])   # cone tip — adjust
    probe_row = int(np.argmin(np.linalg.norm(mesh_meca.coords - target, axis=1)))
    print(f"Probe node {probe_row} at {mesh_meca.coords[probe_row]}")

    for snap in snaps:
        omega = 2 * np.pi * snap.f
        u     = snap.fields["u_meca"][probe_row, 1]   # uy axial
        freqs_sweep.append(snap.f)
        u_sweep.append(u)
        v_sweep.append(1j * omega * u)
        a_sweep.append(-omega**2 * u)

freqs_sweep = np.array(freqs_sweep)

# ── mechanical sweep plot ─────────────────────────────────────────────────────
fig = plot_meca_sweep(freqs_sweep, np.array(u_sweep),
                      np.array(v_sweep), np.array(a_sweep),
                      title="Cone tip - mechanical sweep")
fig.savefig(out_dir / "meca_sweep.pdf", bbox_inches="tight")

# ── field map at a specific frequency ────────────────────────────────────────
target_f = 1000   # Hz
with FreqReader(h5_path) as r:
    snaps = r.snapshots()
    idx   = int(np.argmin(np.abs(freqs_sweep - target_f)))
    snap  = snaps[idx]
    mesh_meca = r.mesh(snap.mesh_id, Domain.MECA)
    u_mag = np.linalg.norm(np.abs(snap.fields["u_meca"]), axis=1)

ax = plot_field(mesh_meca.coords, mesh_meca.tris, u_mag,
                title=f"|u| @ {snap.f:.0f} Hz", **STYLE["meca"])
ax.figure.savefig(out_dir / f"field_{snap.f:.0f}Hz.pdf", bbox_inches="tight")

plt.show()