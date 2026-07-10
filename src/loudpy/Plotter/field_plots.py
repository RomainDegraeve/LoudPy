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

"""
Animate the FSI frequency sweep: acoustic pressure field + deformed
mechanical structure, one clip per snapshot file, stitched into an .mp4.

Two layers per frame
--------------------
  * acoustic domain : instantaneous pressure  Re(p e^{j phi})  (diverging cmap)
  * mechanical mesh : deformed by             Re(u e^{j phi})  colored by |u|

With n_phase > 1 the membrane visibly oscillates at each frequency before
the sweep moves on. Because the mesh is regenerated at every frequency
(PML thickness and acoustic size depend on lambda), the PyVista meshes are
rebuilt for every snapshot -- nothing is assumed constant across frames.

Follows the same convention as the plotting module: the animation function
takes plain numpy arrays; all file/reader handling lives in the loader.
"""


from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1) Loader — adapt the three FIELD/DOMAIN names here if yours differ
# ─────────────────────────────────────────────────────────────────────────────

P_FIELD = "p_acou"     # complex nodal pressure in the acoustic domain
U_FIELD = "u_meca"     # complex nodal displacement (n_nodes, 2) in the meca domain

def load_sweep_frames(run_dir: str | Path, pattern: str = "snap_*.h5",
                      stride: int = 1) -> list[dict]:
    """
    Read every snapshot in every block file and return a list of dicts:
        dict(freq, coords_a, tris_a, p, coords_m, tris_m, u)
    stride : keep every Nth snapshot (applied across the whole sweep).
    """
    from loudpy.Files_Loader import FreqReader, Domain

    files = sorted(Path(run_dir).glob(pattern))
    if not files:
        raise FileNotFoundError(f"No '{pattern}' in {run_dir}")

    frames, k = [], 0
    for fp in files:
        with FreqReader(fp) as r:
            for snap in r.snapshots():
                if k % stride:
                    k += 1
                    continue
                k += 1
                mesh_a = r.mesh(snap.mesh_id, Domain.ACOU)
                mesh_m = r.mesh(snap.mesh_id, Domain.MECA)
                frames.append(dict(
                    freq     = float(snap.f),
                    coords_a = np.asarray(mesh_a.coords, float),
                    tris_a   = np.asarray(mesh_a.tris, np.int64)[:, :3],
                    p        = np.asarray(snap.fields["p_acou"]),
                    coords_m = np.asarray(mesh_m.coords, float),
                    tris_m   = np.asarray(mesh_m.tris, np.int64)[:, :3],
                    u        = np.asarray(snap.fields["u_meca"]),
                ))
                print(f"loaded {fp.name}  f = {snap.f:.1f} Hz")

    frames.sort(key=lambda d: d["freq"])
    return frames


# ─────────────────────────────────────────────────────────────────────────────
def animate_pressure_displacement(frames: list[dict], *,
                                  n_phase: int = 8,
                                  n_cycles: int = 1,
                                  fps: int = 25,
                                  show_grid: bool = True,
                                  deform_scale: float = 0.05,
                                  max_deform: float | None = None,
                                  spl_range: float = 60.0,
                                  per_frame_norm: bool = True,
                                  pressure_mode: str = "mag_phase",   # "mag_phase" | "real"
                                  p_clim: tuple | None = None,        # (pmin, pmax) [Pa] for Re(p)
                                  clip_margin: float | tuple = 3e-3,  # scalar or (l, r, b, t) [m]
                                  xlim: tuple | None = None,
                                  ylim: tuple | None = None,
                                  cmap_phase: str = "twilight",
                                  cmap_spl: str = "magma",
                                  cmap_real: str = "RdBu_r",
                                  cmap_u: str = "viridis",
                                  window_size: tuple = (1088, 1088),
                                  show_meca_rest: bool = True,
                                  save_path: str | Path = "sweep_anim.mp4") -> None:
    """
    Mirrored axisymmetric animation.

    pressure_mode :
        "mag_phase" : left = pressure phase [rad], right = SPL [dB] (static per f)
        "real"      : both halves = Re(p e^{j phi}) [Pa], oscillating in sync
                      with the structure (same phi).
    In both cases the mechanical field oscillates as Re(u e^{j phi}).

    n_cycles    : number of full 2pi periods rendered per frequency.
    p_clim      : fixed colour range for Re(p); None -> symmetric auto range.
    clip_margin : shave [m] off each side of the visible box.  Cells are kept
                  or dropped whole (centroid test), so no interpolated nodal
                  values ever appear at the boundary.
    max_deform  : caps the displayed peak deformation in metres.
    xlim, ylim  : visible window in metres.  Dimensions are rounded up to a
                  multiple of 16 for ffmpeg.
    """
    try:
        import pyvista as pv
        import imageio
    except ImportError as e:
        raise ImportError("pip install pyvista imageio imageio-ffmpeg") from e

    if pressure_mode not in ("mag_phase", "real"):
        raise ValueError("pressure_mode must be 'mag_phase' or 'real'")

    def _pd(coords, tris, mirror=False, z=0.0):
        pts = np.column_stack([coords, np.full(len(coords), z)])
        if mirror:
            pts[:, 0] *= -1.0
        cells = np.column_stack(
            [np.full(len(tris), 3, dtype=np.int64), tris]).ravel()
        return pv.PolyData(pts, cells)

    def _mask_tris(coords, tris, box):
        """
        Keep triangles whose centroid lies inside box = (x0, x1, y0, y1).
        Returns (sub_coords, sub_tris, orig_ids) with exact node indices —
        no interpolation, so scalars stay true nodal values.
        """
        c = coords[tris].mean(axis=1)                       # (n_tri, 2)
        keep = ((c[:, 0] >= box[0]) & (c[:, 0] <= box[1]) &
                (c[:, 1] >= box[2]) & (c[:, 1] <= box[3]))
        t = tris[keep]
        if len(t) == 0:
            raise ValueError("clip_margin removed the entire mesh")
        used = np.unique(t)
        remap = np.full(len(coords), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        return coords[used], remap[t], used

    # ── amplitude scaling ─────────────────────────────────────────────────────
    L = max(float(np.ptp(f["coords_a"], axis=0).max()) for f in frames)
    disp_peak = deform_scale * L
    if max_deform is not None:
        disp_peak = min(disp_peak, max_deform)
    z_off = 1e-3 * L

    spl_gmax = max(
        20 * np.log10(np.abs(f["p"]).max() / 20e-6 + 1e-30) for f in frames)
    p_gmax = max(float(np.abs(f["p"]).max()) for f in frames) or 1e-30

    # ── window / camera ───────────────────────────────────────────────────────
    window_size = tuple(int(np.ceil(w / 16)) * 16 for w in window_size)   # ffmpeg

    pv.global_theme.multi_samples = 8
    plotter = pv.Plotter(off_screen=True, window_size=list(window_size))
    plotter.set_background("white")

    if xlim is None:
        xlim = (-L, L)
    if ylim is None:
        ys = frames[0]["coords_a"][:, 1]
        ylim = (float(ys.min()), float(ys.max()))

    cx, cy  = 0.5 * (xlim[0] + xlim[1]), 0.5 * (ylim[0] + ylim[1])
    aspect  = window_size[0] / window_size[1]
    half_h  = 0.5 * (ylim[1] - ylim[0])
    half_w  = 0.5 * (xlim[1] - xlim[0])
    pad     = 1.30 if show_grid else 1.02      # room for ticks and labels
    p_scale = pad * max(half_h, half_w / aspect)

    # ── clip box (centroid test, applied to the r >= 0 half) ─────────────────
    if np.isscalar(clip_margin):
        ml = mr = mb = mt = float(clip_margin)
    else:
        ml, mr, mb, mt = (float(v) for v in clip_margin)

    # mesh is axisymmetric (r >= 0); mirror is exact, so mask once symmetrically
    r_max = min(abs(xlim[0]) - ml, xlim[1] - mr)
    box_a = (0.0, r_max, ylim[0] + mb, ylim[1] - mt)

    phis = np.linspace(0.0, 2 * np.pi * n_cycles, n_phase * n_cycles,
                       endpoint=False)
    save_path = str(save_path)
    total, done = len(frames) * len(phis), 0

    bar_phase = dict(title="phase [rad]", color="black", vertical=True,
                     position_x=0.02, position_y=0.30, width=0.04, height=0.45)
    bar_spl   = dict(title="SPL [dB]",   color="black", vertical=True,
                     position_x=0.93, position_y=0.30, width=0.04, height=0.45)
    bar_real  = dict(title="Re(p) [Pa]", color="black", vertical=True,
                     position_x=0.93, position_y=0.30, width=0.04, height=0.45)
    bar_u     = dict(title="|u| [m]",    color="black", vertical=False,
                     position_x=0.30, position_y=0.91, width=0.40, height=0.04)

    with imageio.get_writer(save_path, fps=fps, quality=9,
                            macro_block_size=1) as writer:
        for f in frames:
            p      = f["p"]
            spl    = 20 * np.log10(np.abs(p) / 20e-6 + 1e-30)
            phase  = np.angle(p)
            u_mag  = np.linalg.norm(np.abs(f["u"]), axis=1)
            u_amp  = float(np.abs(f["u"]).max()) or 1e-30
            s_geom = disp_peak / u_amp

            spl_top  = float(spl.max()) if per_frame_norm else spl_gmax
            clim_spl = [spl_top - spl_range, spl_top]

            if p_clim is not None:
                clim_real = [float(p_clim[0]), float(p_clim[1])]
            else:
                p_top = (float(np.abs(p).max()) or 1e-30) if per_frame_norm else p_gmax
                clim_real = [-p_top, p_top]

            # exact-node masking: identical for both halves (mirror is exact)
            cA, tA, idxA = _mask_tris(f["coords_a"], f["tris_a"], box_a)
            acou_R = _pd(cA, tA)                 # right half
            acou_L = _pd(cA, tA, mirror=True)    # left half

            if pressure_mode == "mag_phase":
                acou_R.point_data["spl"]   = spl[idxA]
                acou_L.point_data["phase"] = phase[idxA]

            meca_R = _pd(f["coords_m"], f["tris_m"], z=z_off)
            meca_L = _pd(f["coords_m"], f["tris_m"], mirror=True, z=z_off)

            for phi in phis:
                e  = np.exp(1j * phi)
                du = np.real(f["u"] * e) * s_geom      # (n_m, 2)

                plotter.clear()

                if pressure_mode == "mag_phase":
                    plotter.add_mesh(acou_L, scalars="phase", cmap=cmap_phase,
                                     clim=[-np.pi, np.pi], show_edges=False,
                                     name="acouL", scalar_bar_args=bar_phase)
                    plotter.add_mesh(acou_R, scalars="spl", cmap=cmap_spl,
                                     clim=clim_spl, show_edges=False,
                                     name="acouR", scalar_bar_args=bar_spl)
                else:
                    p_re = np.real(p * e)[idxA]       # exact nodal values
                    acou_R.point_data["Re(p)"] = p_re
                    acou_L.point_data["Re(p)"] = p_re
                    plotter.add_mesh(acou_L, scalars="Re(p)", cmap=cmap_real,
                                     clim=clim_real, show_edges=False,
                                     name="acouL", show_scalar_bar=False)
                    plotter.add_mesh(acou_R, scalars="Re(p)", cmap=cmap_real,
                                     clim=clim_real, show_edges=False,
                                     name="acouR", scalar_bar_args=bar_real)

                if show_meca_rest:
                    for m, nm in ((meca_R, "restR"), (meca_L, "restL")):
                        plotter.add_mesh(m, color="grey", style="wireframe",
                                         line_width=0.6, opacity=0.3,
                                         lighting=False, name=nm)

                for base, sign, nm, bar in (
                        (meca_R, +1.0, "mecaR", bar_u),
                        (meca_L, -1.0, "mecaL", None)):
                    dm  = base.copy()
                    pts = dm.points.copy()
                    pts[:, 0] += sign * du[:, 0]      # mirror flips u_r
                    pts[:, 1] += du[:, 1]
                    pts[:, 2] += 0.5 * z_off
                    dm.points = pts
                    dm.point_data["|u|"] = u_mag
                    plotter.add_mesh(dm, scalars="|u|", cmap=cmap_u,
                                     show_edges=False, name=nm,
                                     show_scalar_bar=bar is not None,
                                     scalar_bar_args=bar or {})

                hud = (f"f = {f['freq']:8.1f} Hz   SPLmax = {spl.max():5.1f} dB   "
                       f"deform x{s_geom:.3g}")
                if pressure_mode == "real":
                    hud += f"   phi = {np.degrees(phi) % 360.0:5.1f} deg"
                plotter.add_text(hud, position="upper_edge", color="black",
                                 name="hud", font_size=13)

                plotter.view_xy()

                if show_grid:
                    plotter.show_grid(
                        bounds=(xlim[0], xlim[1], ylim[0], ylim[1], 0.0, 0.0),
                        location="outer",
                        xtitle="r [m]", ytitle="z [m]",
                        show_zaxis=False,
                        color="black", font_size=12, fmt="%.2f",
                        grid="back", ticks="outside",
                        n_xlabels=9, n_ylabels=9,
                        padding=0.0,
                        use_3d_text=False,
                    )

                plotter.enable_parallel_projection()
                plotter.camera.focal_point    = (cx, cy, 0.0)
                plotter.camera.position       = (cx, cy, 1.0)
                plotter.camera.parallel_scale = p_scale

                plotter.render()
                writer.append_data(plotter.screenshot())

                done += 1
                if done % 50 == 0:
                    print(f"  {done}/{total} frames")

    plotter.close()
    print(f"→ {save_path}")
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# 3) Driver
# ─────────────────────────────────────────────────────────────────────────────


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
    U_xy    : (n_timesteps, n_nodes, 2)  — node-indexed displacement (ux, uy), real
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
