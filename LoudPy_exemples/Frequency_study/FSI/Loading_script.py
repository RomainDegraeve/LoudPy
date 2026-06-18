import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from loudpy.Files_Loader import FreqReader, Domain
from loudpy.Plotter import (plot_field, plot_fields_grid, plot_spl_sweep,
                             plot_interface_deformed, STYLE, plot_meca_sweep)

results_dir = Path("LoudPy_exemples/Frequency_study/FSI/Results/Files")
out_dir     = Path("LoudPy_exemples/Frequency_study/FSI/Results/Figures")

files = sorted(results_dir.glob("snap_*.h5"))

# ── inspect one file ──────────────────────────────────────────────────────────
with FreqReader(files[0]) as r:
    r.inspect()

# ── helper: iterate all (fpath, snap) pairs across all files ─────────────────
def all_snapshots(files):
    for fpath in files:
        with FreqReader(fpath) as r:
            for snap in r.snapshots():
                yield fpath, snap

# ── SPL sweep ─────────────────────────────────────────────────────────────────
target_acou   = np.array([0.3, 0.1])
freqs_sweep, p_sweep = [], []

for fpath, snap in all_snapshots(files):
    with FreqReader(fpath) as r:
        
        mesh_acou = r.mesh(snap.mesh_id, Domain.ACOU)
        probe_row = int(np.argmin(np.linalg.norm(
                mesh_acou.coords - target_acou, axis=1)))
        freqs_sweep.append(snap.f)
        p_sweep.append(snap.fields["p_acou"][probe_row])

freqs_sweep = np.array(freqs_sweep)
p_sweep     = np.array(p_sweep)

fig = plot_spl_sweep(freqs_sweep, p_sweep, title=f"SPL at probe {target_acou} (front)")
fig.savefig(out_dir / "spl_sweep.pdf", bbox_inches="tight")

# ── mechanical sweep ──────────────────────────────────────────────────────────
target_meca    = np.array([0.0, 0.0])
probe_row_meca = None
u_sweep, v_sweep, a_sweep = [], [], []

for fpath, snap in all_snapshots(files):
    with FreqReader(fpath) as r:
        mesh_meca = r.mesh(snap.mesh_id, Domain.MECA)
        probe_row_meca = int(np.argmin(np.linalg.norm(
                mesh_meca.coords - target_meca, axis=1)))
        omega = 2 * np.pi * snap.f
        u = snap.fields["u_meca"][probe_row_meca, 1]
        u_sweep.append(u)
        v_sweep.append(1j * omega * u)
        a_sweep.append(-omega**2 * u)

fig = plot_meca_sweep(freqs_sweep, np.array(u_sweep),
                      np.array(v_sweep), np.array(a_sweep),
                      title="Cone tip - mechanical sweep")
fig.savefig(out_dir / "meca_sweep.pdf", bbox_inches="tight")

# ── field maps at key frequencies ─────────────────────────────────────────────
target_freqs = [20, 1000, 5000, 15000]

# build index: freq -> (fpath, snap)
all_snaps_index = []
for fpath, snap in all_snapshots(files):
    all_snaps_index.append((snap.f, fpath, snap))

for f_target in target_freqs:
    f_val, fpath, snap = min(all_snaps_index, key=lambda x: abs(x[0] - f_target))

    with FreqReader(fpath) as r:
        mesh_acou = r.mesh(snap.mesh_id, Domain.ACOU)
        mesh_meca = r.mesh(snap.mesh_id, Domain.MECA)
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

# ── subdomain & interface at last target freq ─────────────────────────────────
with FreqReader(fpath) as r:
    sub_mesh, u_sub = r.extract_subdomain(snap, "submeca_surround", "u_meca")
    u_sub_mag = np.linalg.norm(np.abs(u_sub), axis=1)

ax = plot_field(sub_mesh.coords, sub_mesh.tris, u_sub_mag,
                title=f"Surround |u| @ {snap.f:.0f} Hz", **STYLE["meca"])
ax.figure.savefig(out_dir / f"surround_{snap.f:.0f}Hz.pdf", bbox_inches="tight")

with FreqReader(fpath) as r:
    coords_iface, u_iface = r.extract_interface(
        snap, "interface_acou_meca_front", "u_meca")

ax = plot_interface_deformed(
    coords_iface[:, 0], coords_iface[:, 1],
    u_iface[:, 0], u_iface[:, 1],
    scale=1e4,
    title=f"Interface deformation @ {snap.f:.0f} Hz")
ax.figure.savefig(out_dir / f"interface_{snap.f:.0f}Hz.pdf", bbox_inches="tight")

plt.show()