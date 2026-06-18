from loudpy.Files_Loader.Readers import EigenReader
from loudpy.Files_Loader.Load_Domains import Domain
from loudpy.Plotter import plot_modes_grid, plot_field, plot_interface_deformed, STYLE

import matplotlib.pyplot as plt
import numpy as np

mat_path = "src/loudpy/Materials_Bank/materials.json"
in_path = "LoudPy_exemples/Eigen_study/Results/Files/eigen_study.h5"



with EigenReader(in_path) as r:
    r.inspect()

    snaps, mesh, dpn = r.load(Domain.MECA)

    freqs = np.array([s.freq for s in snaps])
    zetas =  np.array([s.zeta for s in snaps])
    shapes = np.stack([s.fields["u_meca"] for s in snaps])  # (n_modes, n_nodes, 2)

    print(np.shape(shapes))

    mesh_id = snaps[0].mesh_id
    print(r.subdomain_names())
    print(r.interface_names()) 

    surround_mesh, u_vals = r.extract_subdomain(snaps[2], "submeca_surround", "u_meca")
    interface_mesh, u_interface = r.extract_interface(snaps[2],interface="interface_acou_meca_front", field ="u_meca" )
           


from pathlib import Path
out = Path("LoudPy_exemples/Eigen_study/Results/figures")


# ── plot all modes grid ───────────────────────────────────────────
fig_grid = plot_modes_grid(mesh.coords, mesh.tris, shapes, freqs,
                           zetas=zetas, n_plot=42, ncols=4, deform_scale=0.05)


fig_grid.savefig(out / "modes_grid.pdf", bbox_inches="tight")

# ── plot one mode on full mesh ────────────────────────────────────
k = 2
u_mag = np.linalg.norm(np.abs(shapes[k]), axis=1)
ax = plot_field(mesh.coords, mesh.tris, u_mag,
                title=f"Mode {k+1} - {freqs[k]:.1f} Hz", **STYLE["meca"])
ax.figure.savefig(out / f"mode_{k+1}_full.pdf", bbox_inches="tight")

fig = plot_modes_grid(mesh.coords, mesh.tris, shapes, freqs,
                      zetas=zetas, mode_indices=[0, 2, 4, 16, 20], ncols=3)


# ── plot one mode restricted to subdomain ────────────────────────
u_cone_mag = np.linalg.norm(np.abs(u_vals), axis=1)
ax = plot_field(surround_mesh.coords, surround_mesh.tris, u_cone_mag,
                title=f"Cone - mode {k+1} - {freqs[k]:.1f} Hz", **STYLE["meca"])
ax.figure.savefig(out / f"mode_{k+1}_surround.pdf", bbox_inches="tight")

# ── plot deformed interface ───────────────────────────────────────
ax = plot_interface_deformed(
    interface_mesh[:, 0], interface_mesh[:, 1],
    u_interface[:, 0], u_interface[:, 1],
    scale=0.1,
    title=f"Interface - mode {k+1} - {freqs[k]:.1f} Hz")
ax.figure.savefig(out / f"mode_{k+1}_interface.pdf", bbox_inches="tight")

plt.show()




