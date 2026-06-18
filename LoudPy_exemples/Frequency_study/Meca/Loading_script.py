import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from loudpy.Files_Loader import FreqReader, Domain
from loudpy.Plotter import (plot_field, plot_fields_grid, plot_spl_sweep,
                             plot_interface_deformed, STYLE, plot_meca_sweep)

# ── paths ─────────────────────────────────────────────────────────────────────
results_dir = Path("LoudPy_exemples/Frequency_study/Meca/Results/Files")
out_dir     = Path("LoudPy_exemples/Frequency_study/Meca/Results/Figures")


files = sorted(results_dir.glob("snap_*.h5"))


# ── inspect one file ──────────────────────────────────────────────────────────
with FreqReader(files[0]) as r:
    r.inspect()



probe_row_meca = None
u_sweep, v_sweep, a_sweep = [], [], []
freqs_sweep = []
for fpath in files:
    with FreqReader(fpath) as r:
        snap = r.load()
        if probe_row_meca is None:
            mesh_meca = r.mesh(snap.mesh_id, Domain.MECA)
            target = np.array([0.0, 0.0])   # adjust to cone tip coords
            probe_row_meca = int(np.argmin(np.linalg.norm(
                mesh_meca.coords - target, axis=1)))
        omega = 2 * np.pi * snap.f
        freqs_sweep.append(snap.f)
        u = snap.fields["u_meca"][probe_row_meca, 1]   # uy (axial)
        u_sweep.append(u)
        v_sweep.append(1j * omega * u)
        a_sweep.append(-omega**2 * u)

fig = plot_meca_sweep(freqs_sweep, np.array(u_sweep),
                      np.array(v_sweep), np.array(a_sweep),
                      title="Cone tip - mechanical sweep")

fig.savefig(out_dir / "meca_sweep.pdf", bbox_inches="tight")


# ── field maps at a few key frequencies ──────────────────────────────────────
target_freqs = [20, 1000, 5000, 15000]   # Hz — adjust to your sweep range

for f_target in target_freqs:
    idx   = int(np.argmin(np.abs(np.array([freqs_sweep]) - f_target)))
    fpath = files[idx]

    with FreqReader(fpath) as r:
        snap      = r.load()
       
        mesh_meca = r.mesh(snap.mesh_id, Domain.MECA)

        u_mag = np.linalg.norm(np.abs(snap.fields["u_meca"]), axis=1)

    fig = plot_fields_grid([
    dict(coords=mesh_meca.coords, tris=mesh_meca.tris, values=u_mag,
         title=f"|u| @ {snap.f:.0f} Hz", **STYLE["meca"]),], ncols=2)
    
    fig.savefig(out_dir / f"fields_{snap.f:.0f}Hz.pdf", bbox_inches="tight")

# ── subdomain field (surround only) at one frequency ─────────────────────────
with FreqReader(files[idx]) as r:
    snap = r.load()
    sub_mesh, u_sub = r.extract_subdomain(snap, "submeca_surround", "u_meca")
    u_sub_mag = np.linalg.norm(np.abs(u_sub), axis=1)

ax = plot_field(sub_mesh.coords, sub_mesh.tris, u_sub_mag,
                title=f"Surround |u| @ {snap.f:.0f} Hz", **STYLE["meca"])
ax.figure.savefig(out_dir / f"surround_{snap.f:.0f}Hz.pdf", bbox_inches="tight")

# ── interface deformation (acou-meca front) at one frequency ─────────────────
with FreqReader(files[idx]) as r:
    snap = r.load()
    coords_iface, u_iface = r.extract_interface(
        snap, "interface_acou_meca_front", "u_meca")

ax = plot_interface_deformed(
    coords_iface[:, 0], coords_iface[:, 1],
    u_iface[:, 0], u_iface[:, 1],
    scale=1e4,
    title=f"Interface deformation @ {snap.f:.0f} Hz")
ax.figure.savefig(out_dir / f"interface_{snap.f:.0f}Hz.pdf", bbox_inches="tight")

plt.show()