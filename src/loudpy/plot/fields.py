"""
Plots of fields on meshes (loudpy.results.Field) and of mode sets.

    field(f)                    colour map on the r–z section (smooth: P2 triangles split through
                                their mid-side nodes), region outlines, colour bar with units;
                                deformed=True draws a displacement on the deformed structure;
                                mirror=True shows the full section (−r … r); 3D fields are rendered
    boundary(f, tags)           rest and deformed positions of boundary nodes (profile view)
    modes(modes)                grid of mode shapes of one mode set (deformed, coloured by |u|)
    coupled_modes(modes, idx)   coupled modes: relative pressure level in the air + deformed structure
"""
from __future__ import annotations

import numpy as np

from loudpy.plot import style
from loudpy.results.quantities import Quantity, quantity


# ── helpers ──────────────────────────────────────────────────────────────────
def _instant(values) -> np.ndarray:
    """Real field at the phase of the largest amplitude (harmonic), or the values themselves (real)."""
    v = np.asarray(values)
    if not np.iscomplexobj(v):
        return v
    j = np.argmax(np.abs(v))
    return np.real(v * np.exp(-1j * np.angle(v.ravel()[j])))


def _outline_segments(mesh, rows_xy):
    """Boundary edges of every 2D group (edges used by one corner triangle of the group): (n, 2, 2)."""
    segs = []
    for name in mesh.names(2):
        cells = mesh.groups[name].cells
        if cells.shape[1] == 6:          # follow the mid-side nodes: 2 segments per P2 edge
            e = [(0, 3), (3, 1), (1, 4), (4, 2), (2, 5), (5, 0)]
        else:
            e = [(0, 1), (1, 2), (2, 0)]
        rows = mesh.rows_of(cells)
        E = np.sort(np.vstack([rows[:, list(p)] for p in e]), axis=1)
        uniq, cnt = np.unique(E, axis=0, return_counts=True)
        segs.append(rows_xy[uniq[cnt == 1]])
    return np.vstack(segs) if segs else np.zeros((0, 2, 2))


def _title(f) -> str:
    q = quantity(f.coord)
    head = f"${q.symbol}$ = {f.coordinate:.4g} {q.unit}".strip()
    if "index" in f.meta and f.coord == "f" and "zeta" in f.extra:
        z = np.real(f.extra["zeta"])
        head = f"mode {f.meta['index']}: {head}" + (rf", $\zeta$ = {z:.3g}" if np.isfinite(z) else "")
    return head


def _scalar(f, show):
    """(values, quantity, cmap, symmetric) of what is shown of a field."""
    if show is None:
        show = "spl" if f.name == "p" and np.iscomplexobj(f.values) else "abs"
    if show == "spl":
        s = f.spl()
        return s.values, s.quantity, style.CMAPS["spl"], False
    if show in ("r", "z", "theta"):
        c = f.component(show)
        return np.abs(c.values), c.quantity.part("abs"), style.CMAPS["displacement"], False
    if show == "abs":
        m = f.magnitude() if (f.is_vector or np.iscomplexobj(f.values)) else f
        kind = "pressure" if f.name == "p" else "displacement"
        return np.asarray(m.values, float), m.quantity, style.CMAPS.get(kind if kind != "pressure" else "spl"), False
    if show in ("real", "imag"):
        v = getattr(f, show)
        vals = v.values if not f.is_vector else v.values[:, np.argmax(np.abs(v.values).max(axis=0))]
        return vals, v.quantity, style.DIVERGING, True
    if show == "instant":
        v = _instant(f.values)
        vals = v if v.ndim == 1 else v[:, np.argmax(np.abs(v).max(axis=0))]
        return vals, f.quantity.part("real"), style.DIVERGING, True
    if show == "phase":
        ph = f.phase()
        return ph.values, ph.quantity, style.CMAPS["phase"], False
    raise ValueError(f"unknown show={show!r}: 'spl', 'abs', 'real', 'imag', 'instant', 'phase', 'r', 'z', 'theta'")


# ── one field ────────────────────────────────────────────────────────────────
@style.styled
def field(f, show: str | None = None, ax=None, deformed=None, scale: float | None = None, factor: float | None = None,
          mirror: bool = False,
          outline: bool = True, mesh: bool = False, cmap=None, vmin=None, vmax=None, range_db: float | None = 50.0,
          xlim=None, ylim=None, title=None, colorbar: bool = True, **kwargs):
    """
    A Field on its mesh; returns the axes.

    show      : "spl" (pressure default), "abs" (default otherwise), "real", "imag", "instant"
                (real field at the phase of the peak), "phase", or a component "r", "z", "theta"
    deformed  : displacement fields: draw on the deformed structure (True, or a Field "u" to deform
                another field's mesh with); scale: peak deformation as a fraction of the mesh size (0.05);
                factor: an absolute displacement multiplier instead (one scale for several plots)
    mirror    : full section −r … r (axisymmetric fields)
    outline   : region outlines; mesh: element edges
    range_db  : SPL maps: colour range below the top (top = 99.5th percentile; None: full range)
    """
    if f.mesh.dim == 3:
        from loudpy.plot.view3d import field3d
        return field3d(f, show=show, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, title=title, colorbar=colorbar, **kwargs)
    if ax is None:
        _, ax = style.figure("field")
    vals, q, cm, symmetric = _scalar(f, show)
    cmap = cmap or cm
    if q.name == "spl" and vmin is None and range_db is not None:
        top = float(np.nanpercentile(vals, 99.5)) if vmax is None else vmax
        vmin, vmax = top - range_db, top
    if symmetric and vmin is None and vmax is None:
        vmax = float(np.nanmax(np.abs(vals))) or 1.0
        vmin = -vmax
    xy = np.asarray(f.mesh.coords, float).copy()
    rest = xy.copy()
    if deformed is not None and deformed is not False:
        u = f if deformed is True else deformed
        if u.name not in ("u", "v", "a") or not u.is_vector:
            raise ValueError("deformed= needs a displacement field (u)")
        ui = _instant(u.values)[:, :2]
        size = float(np.ptp(xy, axis=0).max())
        s = factor if factor is not None else \
            (0.05 if scale is None else scale) * size / max(float(np.abs(ui).max()), 1e-300)
        if u.mesh is not f.mesh:
            rows = u.mesh.rows_of(f.mesh.node_tags)
            ui = ui[rows]
        xy = xy + s * ui
    tri = f.mesh.triangles(refine=True)
    sides = [1.0, -1.0] if mirror else [1.0]
    for sgn in sides:
        tp = ax.tripcolor(sgn * xy[:, 0], xy[:, 1], tri, vals, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax,
                          rasterized=True, **kwargs)
        if mesh:
            ax.triplot(sgn * xy[:, 0], xy[:, 1], tri, color="k", lw=0.15, alpha=0.4)
        if outline:
            from matplotlib.collections import LineCollection
            if deformed is not None and deformed is not False:
                seg0 = _outline_segments(f.mesh, rest * [sgn, 1])
                ax.add_collection(LineCollection(seg0, colors="0.6", linewidths=0.4, linestyles=(0, (2, 2))))
            ax.add_collection(LineCollection(_outline_segments(f.mesh, xy * [sgn, 1]), colors="0.15", linewidths=0.4))
    if mirror:
        ax.axvline(0.0, color="0.3", lw=0.4, ls="-.")
    ax.set_aspect("equal")
    ax.grid(False, which="both")
    ax.set_xlabel(style.LABELS["r"])
    ax.set_ylabel(style.LABELS["z"])
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if colorbar:
        ax.figure.colorbar(tp, ax=ax, label=q.axis_label(), pad=0.02, shrink=0.9)
    ax.set_title(_title(f) if title is None else title)
    return ax


@style.styled
def boundary(f, tags, scale: float | None = None, ax=None, title=None):
    """Rest and deformed positions of the nodes `tags` (e.g. res.boundary(name)) of a displacement field."""
    if ax is None:
        _, ax = style.figure("wide")
    coords, u = f.at_nodes(tags)
    ui = _instant(u)[:, :2]
    order = np.lexsort((coords[:, 1], coords[:, 0]))
    coords, ui = coords[order], ui[order]
    size = float(np.ptp(coords, axis=0).max())
    s = (0.05 if scale is None else scale) * size / max(float(np.abs(ui).max()), 1e-300)
    ax.plot(coords[:, 0], coords[:, 1], ".", color="0.6", ms=2, label="rest")
    ax.plot(coords[:, 0] + s * ui[:, 0], coords[:, 1] + s * ui[:, 1], ".", ms=2.5, label=f"deformed (×{s:.3g})")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(style.LABELS["r"])
    ax.set_ylabel(style.LABELS["z"])
    ax.legend()
    ax.set_title(_title(f) if title is None else title)
    return ax


# ── mode sets ────────────────────────────────────────────────────────────────
def _one_set(modes):
    if len(modes.blocks) != 1:
        raise ValueError(f"pick one mode set first (sets: {modes.sets}), e.g. modes.order(0) or modes.set('fsi')")
    return modes


@style.styled
def modes(modes, indices=None, n: int | None = None, ncols: int = 4, scale: float = 0.05, show: str = "abs",
          name: str = "u", size=None):
    """
    Grid of mode shapes of one mode set (e.g. modes.order(0)): each mode on the deformed structure,
    coloured by its normalised |u|; titles with frequency and damping ratio. Returns the figure.
    """
    import matplotlib.pyplot as plt
    _one_set(modes)
    idx = list(indices) if indices is not None else list(range(min(n or 12, len(modes))))
    nrows = int(np.ceil(len(idx) / ncols))
    w = style.SIZES["wide"][0]
    fig, axes = plt.subplots(nrows, ncols, figsize=size or (w, w / ncols * 0.95 * nrows), squeeze=False,
                             layout="constrained")
    for ax, k in zip(axes.ravel(), idx):
        f = modes.mode(k, name)
        v, _, cmap, _ = _scalar(f, show)
        vmax = float(np.max(np.abs(v))) or 1.0
        f_norm = f._scalar(v / vmax, Quantity("shape", "Normalised amplitude"))
        field(f_norm, ax=ax, deformed=f, scale=scale, colorbar=False, cmap=cmap, vmin=0, vmax=1,
              title=f"{f.coordinate:.1f} Hz" + (rf", $\zeta$ = {np.real(f.extra['zeta']):.2g}"
                                                 if "zeta" in f.extra and np.isfinite(np.real(f.extra["zeta"])) else ""))
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelsize=6)
    for ax in axes.ravel()[len(idx):]:
        ax.axis("off")
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    fig.colorbar(ScalarMappable(Normalize(0, 1), cmap), ax=axes, shrink=0.6, label="Normalised amplitude $|u|/|u|_{max}$")
    fig.supxlabel(style.LABELS["r"], fontsize=9)
    fig.supylabel(style.LABELS["z"], fontsize=9)
    return fig


@style.styled
def coupled_modes(modes, indices, labels=None, range_db: float = 60.0, scale: float = 0.05, xlim=None, ylim=None):
    """
    Coupled structure + air modes of one set (modes.set("fsi")): per mode, the relative pressure
    level in the air (dB re its 99th percentile) with the deformed structure drawn on top.
    The modal amplitude is arbitrary: only relative levels are meaningful. Returns the figure.
    """
    import matplotlib.pyplot as plt
    _one_set(modes)
    idx = list(indices)
    ncols = min(3, len(idx))
    nrows = int(np.ceil(len(idx) / ncols))
    w = style.SIZES["wide"][0]
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, w / ncols * 1.15 * nrows), squeeze=False, layout="constrained")
    for j, (ax, k) in enumerate(zip(axes.ravel(), idx)):
        p, u = modes.mode(k, "p"), modes.mode(k, "u")
        L = p.spl().values
        top = float(np.percentile(L, 99.0))
        rel = p._scalar(L - top, "level")
        name = labels[j] if labels is not None else f"mode {k}"
        zeta = np.real(p.extra.get("zeta", np.nan))
        head = f"{name}\n{p.coordinate:.0f} Hz" + (rf", $\zeta$ = {zeta:.2g}" if np.isfinite(zeta) else "")
        field(rel, ax=ax, cmap=style.CMAPS["spl"], vmin=-range_db, vmax=0, colorbar=False, outline=False,
              xlim=xlim, ylim=ylim, title=head)
        ax.title.set_fontsize(8)
        solid = u._scalar(np.full(u.mesh.n_nodes, 0.55), Quantity("solid", "Structure"))
        field(solid, ax=ax, deformed=u, scale=scale, colorbar=False, cmap="Greys", vmin=0, vmax=1, xlim=xlim,
              ylim=ylim, title=head)
        ax.title.set_fontsize(8)
        ax.set_xlabel("")
        ax.set_ylabel("")
    for ax in axes.ravel()[len(idx):]:
        ax.axis("off")
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    fig.colorbar(ScalarMappable(Normalize(-range_db, 0), style.CMAPS["spl"]), ax=axes, shrink=0.6,
                 label=r"Relative pressure level $\Delta L_p$ [dB]")
    fig.supxlabel(style.LABELS["r"], fontsize=9)
    fig.supylabel(style.LABELS["z"], fontsize=9)
    return fig


@style.styled
def mode_chart(modes, y: str = "zeta", ax=None, color=None, label=None, **kwargs):
    """
    Modes as points: frequency (log axis) against the damping ratio (y="zeta") or the azimuthal
    order (y="order", coloured by ζ); one colour per mode set otherwise. Returns the axes.
    """
    from loudpy.plot.curves import _x_axis
    if ax is None:
        _, ax = style.figure("wide")
    if y == "order":
        f, m, z = modes.frequencies, np.concatenate([np.full(len(b), b.meta.get("order", -1)) for b in modes.blocks]), \
            np.real(modes.zeta)
        sc = ax.scatter(f, m, c=z, cmap=style.CMAPS["level"], s=kwargs.pop("s", 14), **kwargs)
        ax.figure.colorbar(sc, ax=ax, label=quantity("zeta").axis_label(), pad=0.015)
        ax.set_yticks(sorted(set(m.astype(int))))
        ax.set_ylabel(quantity("order").axis_label())
    else:
        for k, b in enumerate(modes.blocks):
            ax.plot(b.values, np.real(b.extra.get("zeta", np.full(len(b), np.nan))), "o",
                    ms=kwargs.get("ms", 3.5), color=color or style.COLORS[k % len(style.COLORS)],
                    label=label if label is not None else (b.label if len(modes.blocks) > 1 else None))
        ax.set_ylabel(quantity("zeta").axis_label())
        if label is not None or len(modes.blocks) > 1:
            ax.legend()
    _x_axis(ax, "f", modes.frequencies, None)
    return ax
