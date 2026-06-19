"""
Loading_script.py — Visualise the results of a modal analysis.

Reads eigen_study.h5 and produces the following figures in Results/Figures/:
  - modes_grid.pdf        : all computed mode shapes on the full mesh
  - modes_selection.pdf   : a hand-picked subset of mode shapes
  - mode_k_full.pdf       : one mode on the full mesh
  - mode_k_surround.pdf   : that mode restricted to the surround subdomain
  - mode_k_interface.pdf  : deformed profile of the fluid-structure interface
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from loudpy.Files_Loader import EigenReader, Domain
from loudpy.Plotter import (
    plot_modes_grid, plot_field, plot_interface_deformed, STYLE,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
in_path = "LoudPy_exemples/Eigen_study/Results/Files/eigen_study.h5"
out_dir = Path("LoudPy_exemples/Eigen_study/Results/Figures")
out_dir.mkdir(parents=True, exist_ok=True)

# ── Inspect file contents ──────────────────────────────────────────────────────
with EigenReader(in_path) as r:
    r.inspect()   # prints a summary of stored modes, meshes, and sub-domains

# ── Load all modal data ────────────────────────────────────────────────────────
with EigenReader(in_path) as r:
    # load() returns one EigenSnapshot per mode, the full mesh, and dpn
    # (degrees of freedom per node — 2 for 2-D mechanics: ux, uy).
    snaps, mesh, dpn = r.load(Domain.MECA)

    freqs  = np.array([s.freq           for s in snaps])        # (n_modes,)  [Hz]
    zetas  = np.array([s.zeta           for s in snaps])        # (n_modes,)  damping ratios
    shapes = np.stack([s.fields["u_meca"] for s in snaps])      # (n_modes, n_nodes, 2)

    print(f"Loaded {len(freqs)} modes,  "
          f"f = [{freqs[0]:.1f}, {freqs[-1]:.1f}] Hz")
    print("Available sub-domains:", r.subdomain_names())
    print("Available interfaces :", r.interface_names())

    # Choose mode index k to examine in detail (0-based: k=2 → mode 3)
    k = 2

    # Extract displacement field restricted to the surround sub-domain
    # Returns: surround_mesh (coords, tris for surround nodes only)
    #          u_surround    (n_surround_nodes, 2) complex displacement
    surround_mesh, u_surround = r.extract_subdomain(
        snaps[k], "submeca_surround", "u_meca")

    # Extract displacement along the fluid-structure interface boundary
    # Returns: coords_iface (n_iface_nodes, 2) node coordinates
    #          u_iface      (n_iface_nodes, 2) complex displacement
    coords_iface, u_iface = r.extract_interface(
        snaps[k], interface="interface_acou_meca_front", field="u_meca")

# ── Mode-shape grid — all modes ────────────────────────────────────────────────
# Each panel shows the displacement magnitude |u| with the undeformed mesh
# overlaid in grey.  deform_scale controls the visible deformation amplitude
# relative to the characteristic mesh size.
fig = plot_modes_grid(
    mesh.coords, mesh.tris, shapes, freqs,
    zetas=zetas, n_plot=42, ncols=4, deform_scale=0.05)
fig.savefig(out_dir / "modes_grid.pdf", bbox_inches="tight")

# ── Mode-shape grid — user-selected modes ─────────────────────────────────────
# Pass mode_indices to choose any subset of modes by 0-based index.
fig = plot_modes_grid(
    mesh.coords, mesh.tris, shapes, freqs,
    zetas=zetas, mode_indices=[0, 2, 4, 16, 20], ncols=3)
fig.savefig(out_dir / "modes_selection.pdf", bbox_inches="tight")

# ── Single mode — full mesh ────────────────────────────────────────────────────
u_mag = np.linalg.norm(np.abs(shapes[k]), axis=1)   # scalar |u| per node
ax = plot_field(mesh.coords, mesh.tris, u_mag,
                title=f"Mode {k+1} - {freqs[k]:.1f} Hz", **STYLE["meca"])
ax.figure.savefig(out_dir / f"mode_{k+1}_full.pdf", bbox_inches="tight")

# ── Single mode — surround subdomain only ─────────────────────────────────────
u_sur_mag = np.linalg.norm(np.abs(u_surround), axis=1)
ax = plot_field(surround_mesh.coords, surround_mesh.tris, u_sur_mag,
                title=f"Surround - mode {k+1} - {freqs[k]:.1f} Hz",
                **STYLE["meca"])
ax.figure.savefig(out_dir / f"mode_{k+1}_surround.pdf", bbox_inches="tight")

# ── Fluid-structure interface — deformed profile ───────────────────────────────
# Compares the rest position (dots) against the scaled deformed position (red)
# along the coupling boundary between the cone and the acoustic cavity.
ax = plot_interface_deformed(
    coords_iface[:, 0], coords_iface[:, 1],
    u_iface[:, 0], u_iface[:, 1],
    scale=0.1,
    title=f"Interface - mode {k+1} - {freqs[k]:.1f} Hz")
ax.figure.savefig(out_dir / f"mode_{k+1}_interface.pdf", bbox_inches="tight")

plt.show()
