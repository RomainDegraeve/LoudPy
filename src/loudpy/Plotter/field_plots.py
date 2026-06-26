"""
Spatial field plotting — all functions take plain numpy arrays.

No Snapshot, Mesh, Reader or study objects are imported here.
The user extracts data from the file and passes arrays directly.

Typical workflow
----------------
with FreqReader("run.h5") as r:
    snap      = r.load()
    mesh      = r.mesh(snap.mesh_id, Domain.ACOU)
    p_spl     = 20 * np.log10(np.abs(snap.fields["p_acou"]) / 20e-6)

fig = plot_field(mesh.coords, mesh.tris, p_spl, title="SPL [dB]", cmap="magma")
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ── style presets the user can reference ──────────────────────────────────────
STYLE = {
    "acou":  dict(cmap="magma",    label="SPL [dB]"),
    "meca":  dict(cmap="viridis",  label="|u| [m]"),
    "phase": dict(cmap="twilight", label="phase [rad]", vmin=-np.pi, vmax=np.pi),
}

# ── global style (applied at import) ──────────────────────────────────────────
plt.rc("lines",  linewidth=2)
plt.rc("font",   size=14)
plt.rc("axes",   linewidth=1.5, labelsize=14)
plt.rc("legend", fontsize=14)
plt.rcParams["font.family"]                  = "serif"
plt.rcParams["font.serif"]                   = "cmr10"
plt.rcParams["axes.formatter.use_mathtext"]  = True
plt.rcParams["mathtext.fontset"]             = "stix"

# ── core primitive ─────────────────────────────────────────────────────────────

def plot_field(coords: np.ndarray, tris: np.ndarray, values: np.ndarray, *,
               cmap: str = "viridis", shading: str = "gouraud",
               vmin=None, vmax=None,
               xlim=None, ylim=None,
               label: str = "", title: str = "",
               ax=None) -> plt.Axes:
    """
    Draw a scalar field on a triangular mesh.

    Parameters
    ----------
    coords : (n_nodes, 2)
    tris   : (n_tris, 3)  row indices into coords
    values : (n_nodes,)   scalar field (real)
    """
    if ax is None:
        _, ax = plt.subplots()
    tpc = ax.tripcolor(coords[:, 0], coords[:, 1], tris[:, :3], values,
                       cmap=cmap, shading=shading, vmin=vmin, vmax=vmax)
    ax.figure.colorbar(tpc, ax=ax, label=label)
    if xlim: ax.set_xlim(*xlim)
    if ylim: ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    if title: ax.set_title(title)
    return ax


# ── multi-panel layout ─────────────────────────────────────────────────────────

def plot_fields_grid(panels: list[dict], *,
                     ncols: int = 2,
                     figsize: tuple | None = None) -> plt.Figure:
    """
    Draw several fields in a grid.  Each panel is a dict with keys:
        coords, tris, values  — required
        title, cmap, label, vmin, vmax, xlim, ylim  — optional

    Example
    -------
    panels = [
        dict(coords=mesh_a.coords, tris=mesh_a.tris, values=p_spl,
             title="SPL", **STYLE["acou"]),
        dict(coords=mesh_m.coords, tris=mesh_m.tris, values=u_mag,
             title="|u|", **STYLE["meca"]),
    ]
    fig = plot_fields_grid(panels, ncols=2)
    """
    n     = len(panels)
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)
    if figsize is None:
        figsize = (5.5 * ncols, 4.5 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for i, p in enumerate(panels):
        ax = axes[i // ncols][i % ncols]
        plot_field(p["coords"], p["tris"], p["values"],
                   cmap   = p.get("cmap",   "viridis"),
                   shading= p.get("shading","gouraud"),
                   vmin   = p.get("vmin"),
                   vmax   = p.get("vmax"),
                   xlim   = p.get("xlim"),
                   ylim   = p.get("ylim"),
                   label  = p.get("label",  ""),
                   title  = p.get("title",  ""),
                   ax     = ax)

    for i in range(n, nrows * ncols):
        axes[i // ncols][i % ncols].axis("off")

    fig.tight_layout()
    return fig


# ── mode shapes ───────────────────────────────────────────────────────────────

def plot_modes_grid(coords: np.ndarray, tris: np.ndarray,
                    shapes: np.ndarray, freqs: np.ndarray, *,
                    zetas: np.ndarray | None = None,
                    mode_indices: list[int] | None = None,
                    n_plot: int = 10, ncols: int = 5,
                    cmap: str = "viridis",
                    log_scale: bool = False,
                    deform_scale: float = 0.05) -> plt.Figure:
    """
    Grid of mode-shape plots.

    Parameters
    ----------
    coords       : (n_nodes, 2)
    tris         : (n_tris, 3)
    shapes       : (n_modes, n_nodes)      scalar field per mode, or
                   (n_modes, n_nodes, 2)   vector field (ux, uy) per mode
    freqs        : (n_modes,)  natural frequencies in Hz
    zetas        : (n_modes,)  damping ratios, optional
    mode_indices : list of int, optional
                   explicit 0-based indices to plot; overrides n_plot
    deform_scale : float
        Scale deformed shape overlay (0 = no deformation).
        Only used when shapes is (n_modes, n_nodes, 2).
    """
    if mode_indices is not None:
        indices = list(mode_indices)
    else:
        indices = list(range(min(n_plot, len(freqs))))

    n_plot = len(indices)
    ncols  = min(ncols, n_plot)
    nrows  = math.ceil(n_plot / ncols)
    L      = float(np.ptp(coords, axis=0).max())

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4 * ncols, 4 * nrows), squeeze=False)

    vector = shapes.ndim == 3   # (n_modes, n_nodes, 2)

    for plot_pos, k in enumerate(indices):
        row, col = plot_pos // ncols, plot_pos % ncols
        ax       = axes[row][col]

        shape_k = shapes[k]                              # (n_nodes,) or (n_nodes, 2)
        if vector:
            mag = np.linalg.norm(np.abs(shape_k), axis=1)   # (n_nodes,)
        else:
            mag = np.abs(shape_k)

        if log_scale:
            ref = mag.max() * 1e-6 + 1e-30
            mag = 20 * np.log10(np.maximum(mag, ref))

        x, y = coords[:, 0].copy(), coords[:, 1].copy()
        if deform_scale and vector:
            ux, uy = shape_k[:, 0].real, shape_k[:, 1].real
            umax   = max(np.abs(ux).max(), np.abs(uy).max(), 1e-30)
            s      = deform_scale * L / umax
            x += s * ux
            y += s * uy
            ax.triplot(coords[:, 0], coords[:, 1], tris[:, :3],
                       color="k", lw=0.2, alpha=0.3)

        ax.tripcolor(x, y, tris[:, :3], mag, shading="gouraud", cmap=cmap)
        ax.set_aspect("equal")

        sub = rf" $\zeta$={zetas[k]:.3g}" if zetas is not None and zetas[k] is not None else ""
        ax.set_title(f"Mode {k+1}  {freqs[k]:.1f} Hz{sub}")

        is_bottom = (row == nrows - 1) or (
            row == nrows - 2 and col >= (n_plot - (nrows - 1) * ncols))
        if is_bottom:   ax.set_xlabel("x [m]")
        else:           ax.tick_params(labelbottom=False)
        if col == 0:    ax.set_ylabel("y [m]")
        else:           ax.tick_params(labelleft=False)

    for k in range(n_plot, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")

    fig.tight_layout()
    return fig


# ── interface deformation ──────────────────────────────────────────────────────

def plot_interface_deformed(x: np.ndarray, y: np.ndarray,
                             ux: np.ndarray, uy: np.ndarray, *,
                             scale: float = 1.0,
                             title: str = "",
                             ax=None) -> plt.Axes:
    """
    Plot rest shape and deformed shape of an interface.

    Parameters
    ----------
    x, y   : (n_nodes,)  rest coordinates along the interface
    ux, uy : (n_nodes,)  complex displacements — real part is used
    scale  : displacement scale factor
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, y, "ko", alpha=0.3, ms=3, label="rest")
    ax.plot(x + scale * ux.real, y + scale * uy.real,
            "ro", ms=3, label=rf"deformed ($\times${scale:g})")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.legend()
    if title: ax.set_title(title)
    return ax


# ── PyVista animation ──────────────────────────────────────────────────────────

def animate_field(coords: np.ndarray, tris: np.ndarray,
                  U_xy: np.ndarray, time: np.ndarray, *,
                  scale_factor: float = 1.0,
                  start_time: float = 0.0,
                  fps: int = 30,
                  target_duration: float = 10.0,
                  zoom_factor: float = 1.4,
                  cmap: str = "inferno",
                  window_size: tuple = (1920, 1080),
                  show_rest: bool = True,
                  rest_color: str = "lightgrey",
                  rest_opacity: float = 0.35,
                  save_path: str | Path = "anim.mp4") -> None:
    """
    Off-screen PyVista animation of a vector displacement field.

    Parameters
    ----------
    coords  : (n_nodes, 2)
    tris    : (n_tris, 3)
    U_xyz   : (n_timesteps, n_nodes, 2)  — node-indexed displacement (real)
    time    : (n_timesteps,)
    """
    try:
        import pyvista as pv
        import imageio
    except ImportError as e:
        raise ImportError("pip install pyvista imageio imageio-ffmpeg") from e

    N         = len(coords)
    start_idx = int(np.searchsorted(time, start_time))
    stride    = max(1, (len(time) - start_idx) // max(1, int(fps * target_duration)))
    sel       = np.arange(start_idx, len(time), stride)
    n_frames  = len(sel)
    t_sub     = time[sel]

    print(f"[animate_field] {n_frames} frames, stride={stride}, "
          f"t=[{t_sub[0]:.4f}, {t_sub[-1]:.4f}] s")

    U_sel  = np.real(U_xy[sel])               # (n_frames, n_nodes, 2)
    U_mag  = np.linalg.norm(U_sel, axis=2)     # (n_frames, n_nodes)
    vmax   = float(U_mag.max()) or 1.0

    nodes_3d = np.column_stack([coords, np.zeros(N)])
    cells_pv = np.column_stack(
        [np.full(len(tris), 3, dtype=np.int64), tris[:, :3]]
    ).ravel()

    rest_mesh = pv.PolyData(nodes_3d.copy(), cells_pv)
    live_mesh = pv.PolyData(nodes_3d.copy(), cells_pv)
    live_mesh.point_data["Disp"] = U_mag[0]

    pv.global_theme.multi_samples = 8
    plotter = pv.Plotter(off_screen=True, window_size=list(window_size))
    plotter.set_background("white")

    if show_rest:
        plotter.add_mesh(rest_mesh, color=rest_color, style="wireframe",
                         line_width=1.0, opacity=rest_opacity,
                         lighting=False, name="rest_wire")
        plotter.add_mesh(rest_mesh, color=rest_color, opacity=0.15,
                         show_edges=False, lighting=False, name="rest_fill")

    plotter.add_mesh(live_mesh, scalars="Disp",
                     cmap=cmap, clim=[0.0, vmax],
                     show_edges=False, smooth_shading=True,
                     scalar_bar_args=dict(title="|u| [m]", color="black"),
                     name="deformed")
    plotter.view_xy()
    plotter.reset_camera()
    plotter.camera.zoom(zoom_factor)

    save_path = str(save_path)
    with imageio.get_writer(save_path, fps=fps, quality=9) as writer:
        for fi in range(n_frames):
            pts = nodes_3d.copy()
            pts[:, :2] += scale_factor * U_sel[fi]
            live_mesh.points = pts
            live_mesh.point_data["Disp"] = U_mag[fi]
            plotter.add_text(f"t = {t_sub[fi]:.4f} s",
                             position="upper_right", color="black",
                             name="timer", font_size=14)
            plotter.render()
            writer.append_data(plotter.screenshot())
            if (fi + 1) % 50 == 0:
                print(f"  {fi+1}/{n_frames} frames done")

    plotter.close()
    print(f"→ {save_path}")
