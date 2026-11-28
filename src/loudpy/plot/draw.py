"""
A resolved plot (lp.plot.PlotSpec.resolve) drawn in matplotlib with its style options (loudpy.plot.options): every
view of a spec, the same options whatever the view.

    prepare(data, style)         the data as shown: show (abs, real, imag, phase, dB, SPL), normalize, mirror
    draw(resolved, ax, style)    the figure: curve, map, polar, table, field, bode, nyquist, balloon
    table_rows(data, digits)     (header, rows) of a table view

Views: curve (y over one axis; 2D data: one curve per value of the other axis), map (colour over two axes), polar
(over an angle), table, field (a field on its mesh), bode (modulus and phase over the frequency, two axes), nyquist
(imaginary against real part along the curve: an impedance, a transfer), balloon (a directivity over the angle from
the axis revolved around it: radius and colour the level, a 3D axes).
"""
from __future__ import annotations

import dataclasses

import numpy as np

from loudpy.plot import options as opts
from loudpy.plot import style as lstyle

LINESTYLES = {"solid": "-", "dashed": "--", "dotted": ":", "dashdot": "-.", "none": "none"}


# ── the data as shown ────────────────────────────────────────────────────────
def prepare(d, style: dict | None = None):
    """The Data as the style shows it: show (a complex amplitude's modulus, real or imaginary part, phase [°],
    level [dB], SPL), mirror (about 0 along "angle"), normalize (to the maximum or the first value along the first
    axis: divided, or subtracted from a level)."""
    s = opts.check(style)
    show = opts.get(s, "show")
    cplx = np.iscomplexobj(np.asarray(d.values))
    if show == "auto":
        show = "spl" if d.quantity.name in ("p", "pressure") and cplx else ("abs" if cplx else "")
    if show == "abs":
        d = d.abs()
    elif show in ("real", "imag"):
        d = getattr(d, show)                                     # (properties of Data)
    elif show == "phase":
        d = d.phase(deg=True, unwrap=bool(opts.get(s, "unwrap")))
    elif show == "db":
        d = d.db()
    elif show == "spl":
        d = d.spl() if d.quantity.name in ("p", "pressure", "|p|") or "pressure" in d.quantity.label.lower() else d.db()
    if opts.get(s, "mirror") and "angle" in d.coords and np.asarray(d.coords["angle"], float).min() >= 0:
        d = d.mirror("angle")
    norm = opts.get(s, "normalize")
    if norm != "none":
        v = np.asarray(d.values)
        level = d.quantity.unit.startswith("dB")
        ref = np.nanmax(np.abs(v), axis=0, keepdims=True) if norm == "max" else np.take(v, [0], axis=0)
        if norm == "first" and "angle" in d.coords:                  # on the axis: the angle nearest 0
            ia = list(d.coords).index("angle")
            ref = np.take(v, [int(np.argmin(np.abs(np.asarray(d.coords["angle"], float))))], axis=ia)
        label = f"{d.quantity.label} (relative)"
        if level:
            d = d._new(v - ref, dataclasses.replace(d.quantity, unit="dB", label=label))
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                d = d._new(v / np.where(ref == 0, np.nan, ref), dataclasses.replace(d.quantity, unit="", label=label))
    return d


def _cmap(s, quantity_kind: str):
    name = opts.get(s, "cmap")
    if name == "auto":
        name = lstyle.CMAPS.get(quantity_kind, "viridis")
    if opts.get(s, "reverse"):
        name = name[:-2] if name.endswith("_r") else name + "_r"
    return name


def _colour_range(s, values):
    """(vmin, vmax) from the style: explicit, a dB range below the top, symmetric about 0, or automatic."""
    vmin, vmax = s.get("vmin"), s.get("vmax")
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if s.get("range_db") is not None and len(v):
        top = vmax if vmax is not None else float(np.nanmax(v))
        vmin, vmax = top - float(s["range_db"]), top
    if opts.get(s, "symmetric") and len(v):
        m = max(abs(vmin if vmin is not None else v.min()), abs(vmax if vmax is not None else v.max())) or 1.0
        vmin, vmax = -m, m
    return vmin, vmax


def _line_kwargs(s) -> dict:
    kw: dict = {"linestyle": LINESTYLES[opts.get(s, "line")]}
    if opts.get(s, "marker") != "none":
        kw["marker"] = opts.get(s, "marker")
    if s.get("linewidth") is not None:
        kw["linewidth"] = float(s["linewidth"])
    if s.get("color"):
        kw["color"] = s["color"]
    return kw


def _palette(ax, s):
    name = opts.get(s, "palette")
    if name == "loudpy":
        colours = lstyle.COLORS
    else:
        import matplotlib
        cm = matplotlib.colormaps[name]
        n = getattr(cm, "N", 8)
        colours = [cm(i) for i in range(min(n, 10))] if n <= 20 else [cm(x) for x in np.linspace(0, 0.9, 8)]
    ax.set_prop_cycle(color=colours)


def _finish(ax, s, polar=False):
    grid = opts.get(s, "grid")
    if grid == "none":
        ax.grid(False, which="both")
    elif grid == "both" and not polar:
        ax.minorticks_on()
        ax.grid(True, which="both", alpha=0.5)
    else:
        ax.grid(True, which="major")
    if not polar:
        if s.get("xmin") is not None or s.get("xmax") is not None:
            ax.set_xlim(s.get("xmin"), s.get("xmax"))
        if s.get("ymin") is not None or s.get("ymax") is not None:
            ax.set_ylim(s.get("ymin"), s.get("ymax"))
    legend = opts.get(s, "legend")
    handles = ax.get_legend_handles_labels()[0]
    if legend == "none" and ax.get_legend() is not None:
        ax.get_legend().remove()
    elif legend not in ("auto", "none") and handles:
        if legend == "outside":
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)
        else:
            ax.legend(loc=legend)
    if s.get("title"):
        ax.set_title(s["title"])


def _scale(s, name):
    v = opts.get(s, name)
    return None if v == "auto" else v == "log"


# ── the views ────────────────────────────────────────────────────────────────
def draw(resolved, ax=None, style: dict | None = None, **kwargs):
    """Draw a resolved spec in matplotlib with the style options; returns the axes (bode: the two axes)."""
    from loudpy import plot
    s = opts.check(style)
    view, data = resolved.view, resolved.data
    if view == "field" or hasattr(data, "mesh"):
        return _field(data, ax, s, **kwargs)
    if view == "balloon":
        return balloon(prepare(data, s), ax=ax, style=s, **kwargs)
    if view == "nyquist":
        return nyquist(data, ax=ax, style=s, **kwargs)
    if view == "bode":
        return bode(data, ax=ax, style=s, **kwargs)
    d = prepare(data, s)
    if view == "table":
        return table(d, ax=ax, digits=int(opts.get(s, "digits")), title=s.get("title") or resolved.title)
    if view == "polar":
        if ax is None:
            _, ax = lstyle.figure("square", subplot_kw=dict(projection="polar"))
        _palette(ax, s)
        rlim = None
        if s.get("ymin") is not None or s.get("ymax") is not None or s.get("range_db") is not None:
            v = np.asarray(d.values, float)
            top = float(s["ymax"]) if s.get("ymax") is not None else float(np.nanmax(v))
            bottom = float(s["ymin"]) if s.get("ymin") is not None else top - float(s.get("range_db") or 40.0)
            rlim = (bottom, top)
        plot.polar(d, ax=ax, rlim=rlim, zero=opts.get(s, "zero"), clockwise=bool(opts.get(s, "clockwise")),
                   **_line_kwargs(s), **kwargs)
        _finish(ax, s, polar=True)
        return ax
    if view == "map":
        vals = np.asarray(d.values, float)
        vmin, vmax = _colour_range(s, vals)
        fin = vals[np.isfinite(vals)]
        if vmin is None and vmax is None and len(fin):                # the range the bands and isolines span
            if d.quantity.unit == "dB":
                from loudpy.plot.curves import REL_RANGE
                vmin, vmax = REL_RANGE
            elif s.get("levels") or s.get("contours"):
                vmin, vmax = (float(x) for x in np.percentile(fin, [1, 99]))
        levels = int(opts.get(s, "levels") or 0)
        contours = int(opts.get(s, "contours") or 0)
        extra = {}
        if opts.get(s, "scale") == "log":
            from matplotlib.colors import LogNorm
            pos = vals[vals > 0]
            extra["norm"] = LogNorm(vmin=vmin or (pos.min() if len(pos) else 1e-12), vmax=vmax or vals.max())
            vmin = vmax = None
        lo = vmin if vmin is not None else (float(fin.min()) if len(fin) else 0.0)
        hi = vmax if vmax is not None else (float(fin.max()) if len(fin) else 1.0)
        lv = np.linspace(lo, hi, levels + 1) if levels and "norm" not in extra else None
        cl = np.linspace(lo, hi, contours + 2)[1:-1] if contours else None
        ax = plot.map(d, ax=ax, levels=lv, vmin=vmin, vmax=vmax, cmap=_cmap(s, d.quantity.kind),
                      logx=_scale(s, "xscale"), colorbar=bool(opts.get(s, "colorbar")), contours=cl, **extra,
                      **kwargs)
        _finish(ax, {**s, "grid": "none"})
        return ax
    if ax is None:
        _, ax = lstyle.figure("wide")
    _palette(ax, s)
    if d.ndim > 2:
        raise ValueError(f"a curve of {d.ndim}D data: fix the other dimensions first (at=…)")
    ax = plot.line(d, ax=ax, logx=_scale(s, "xscale"), logy=_scale(s, "yscale"), **_line_kwargs(s), **kwargs)
    _finish(ax, s)
    return ax


def _field(f, ax, s, **kwargs):
    from loudpy import plot
    show = opts.get(s, "show")
    show = None if show == "auto" else ("spl" if show == "db" else show)
    vals_kw = {}
    if s.get("vmin") is not None or s.get("vmax") is not None:
        vals_kw.update(vmin=s.get("vmin"), vmax=s.get("vmax"))
    if s.get("range_db") is not None:
        vals_kw["range_db"] = float(s["range_db"])
    if opts.get(s, "symmetric"):
        from loudpy.plot.fields import scalar_values
        v, *_ = scalar_values(f, show)
        m = float(np.nanmax(np.abs(v))) or 1.0
        vals_kw.update(vmin=-m, vmax=m)
    cm = None if opts.get(s, "cmap") == "auto" and not opts.get(s, "reverse") else _cmap(s, "plain")
    if opts.get(s, "cmap") == "auto" and opts.get(s, "reverse"):
        from loudpy.plot.fields import scalar_values
        cm = scalar_values(f, show)[2] + "_r"
    xlim = (s.get("xmin"), s.get("xmax")) if s.get("xmin") is not None or s.get("xmax") is not None else None
    ylim = (s.get("ymin"), s.get("ymax")) if s.get("ymin") is not None or s.get("ymax") is not None else None
    deform = bool(opts.get(s, "deform")) and getattr(f, "name", "") in ("u", "v", "a") and f.is_vector
    ax = plot.field(f, show=show, ax=ax, deformed=True if deform else None,
                    scale=s.get("deform_scale"), mirror=bool(opts.get(s, "mirror")), outline=bool(opts.get(s, "outline")),
                    mesh=bool(opts.get(s, "edges")), cmap=cm, xlim=xlim, ylim=ylim,
                    colorbar=bool(opts.get(s, "colorbar")), title=s.get("title") or None, **vals_kw, **kwargs)
    return ax


def table_rows(d, digits: int = 5) -> tuple[list[str], list[list[str]]]:
    """(header, rows) of Data as a table: one row per value of the first coordinate, one column per value of the
    second (or the values); complex values written a ± bj."""
    def fmt(x):
        if isinstance(x, (complex, np.complexfloating)):
            return f"{x.real:.{digits}g} {'+' if x.imag >= 0 else '−'} {abs(x.imag):.{digits}g}j"
        if isinstance(x, (float, np.floating)):
            return f"{x:.{digits}g}"
        return str(x)
    dims = list(d.coords)
    v = np.asarray(d.values)
    if v.ndim == 0:
        return [d.quantity.column()], [[fmt(v.item())]]
    first = d.coords[dims[0]]
    if v.ndim == 1:
        return [_coord_column(dims[0]), d.quantity.column()], [[fmt(a), fmt(b)] for a, b in zip(first, v)]
    if v.ndim == 2:
        second = d.coords[dims[1]]
        head = [_coord_column(dims[0])] + [f"{d.quantity.name} @ {fmt(c)}" for c in second]
        return head, [[fmt(a)] + [fmt(x) for x in row] for a, row in zip(first, v)]
    raise ValueError("a table of 1D or 2D data: fix the other dimensions first (at=…)")


def _coord_column(dim: str) -> str:
    """The column name of a coordinate: "f [Hz]"."""
    from loudpy.results.quantities import quantity
    return quantity(dim).column()


def table(d, ax=None, digits: int = 5, title: str = "", max_rows: int = 40):
    """Data drawn as a table (the first max_rows rows); returns the axes."""
    head, rows = table_rows(d, digits)
    if ax is None:
        _, ax = lstyle.figure("wide")
        ax.figure.set_layout_engine(None)                           # (a table has no axes to lay out)
    ax.axis("off")
    shown = rows[:max_rows] + ([["…"] * len(head)] if len(rows) > max_rows else [])
    t = ax.table(cellText=shown, colLabels=head, loc="upper center", cellLoc="right")
    t.auto_set_font_size(False)
    t.set_fontsize(7)
    t.scale(1.0, 1.15)
    if title:
        ax.set_title(title, fontsize=9)
    return ax


def bode(d, ax=None, style: dict | None = None, **kwargs):
    """The modulus (top; dB when the style's show is "db" or "spl") and the phase [°] (bottom) of complex Data over
    the frequency; returns (ax_modulus, ax_phase). Given one axes: the phase on a twin axis (dashed)."""
    from loudpy import plot
    s = opts.check(style)
    if not np.iscomplexobj(np.asarray(d.values)):
        raise ValueError("a bode plot needs complex amplitudes (a modulus and a phase)")
    show = opts.get(s, "show")
    mag = prepare(d, {**s, "show": show if show in ("db", "spl") else "abs"})
    ph = d.phase(deg=True, unwrap=bool(opts.get(s, "unwrap")))
    if ax is None:
        import matplotlib.pyplot as plt
        fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=lstyle.SIZES.get("wide", (6.4, 4.0)),
                                      constrained_layout=True)
        twin = False
    else:
        ax2, twin = ax.twinx(), True
    _palette(ax, s)
    _palette(ax2, s)
    kw = _line_kwargs(s)
    plot.line(mag, ax=ax, logx=_scale(s, "xscale"), logy=_scale(s, "yscale"), **kw, **kwargs)
    plot.line(ph, ax=ax2, logx=_scale(s, "xscale"), logy=False, **{**kw, **({"linestyle": "--"} if twin else {})},
              legend=False)
    if not twin:
        ax.set_xlabel("")
    _finish(ax, s)
    _finish(ax2, {**s, "ymin": None, "ymax": None, "title": "", "legend": "none"})
    return ax, ax2


def nyquist(d, ax=None, style: dict | None = None, **kwargs):
    """Imaginary against real part of complex Data along its first coordinate (2D: one curve per value of the
    second), equal axes; returns the axes."""
    s = opts.check(style)
    v = np.asarray(d.values)
    if not np.iscomplexobj(v):
        raise ValueError("a Nyquist plot needs complex amplitudes")
    if ax is None:
        _, ax = lstyle.figure("square")
    _palette(ax, s)
    kw = _line_kwargs(s)
    q = d.quantity
    cols = v[:, None] if v.ndim == 1 else v
    dims = list(d.coords)
    for j in range(cols.shape[1]):
        lab = d.label or None
        if v.ndim == 2:
            lab = f"{dims[1]} = {d.coords[dims[1]][j]}"
        ax.plot(cols[:, j].real, cols[:, j].imag, label=lab, **kw, **kwargs)
    unit = f" [{q.unit}]" if q.unit else ""
    ax.set_xlabel(f"Re {q.label}{unit}")
    ax.set_ylabel(f"Im {q.label}{unit}")
    ax.set_aspect("equal", adjustable="datalim")
    ax.axhline(0, color="0.6", lw=0.5)
    ax.axvline(0, color="0.6", lw=0.5)
    _finish(ax, s)
    return ax


def balloon(d, ax=None, style: dict | None = None, n_phi: int = 73, **kwargs):
    """A directivity balloon: levels over the angle θ from the axis (Data over "angle" [°], 0 … 180 or −θ … θ;
    one frequency) revolved around the axis — the radius is the level above the bottom of the range, the colour the
    level; returns the 3D axes."""
    s = opts.check(style)
    if list(d.coords) != ["angle"]:
        raise ValueError(f"a balloon takes data over the angle only (fix the others first: at=…), got {list(d.coords)}")
    th = np.asarray(d.coords["angle"], float)
    L = np.asarray(d.values, float)
    keep = th >= 0
    th, L = th[keep], L[keep]
    order = np.argsort(th)
    th, L = np.deg2rad(th[order]), L[order]
    top = float(s["vmax"]) if s.get("vmax") is not None else float(np.nanmax(L))
    rng = float(s["range_db"]) if s.get("range_db") is not None else 40.0
    bottom = float(s["vmin"]) if s.get("vmin") is not None else top - rng
    R = np.clip(L - bottom, 0.0, None)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    T, P = np.meshgrid(th, phi, indexing="ij")
    RR = np.repeat(R[:, None], n_phi, axis=1)
    X, Y, Z = RR * np.sin(T) * np.cos(P), RR * np.sin(T) * np.sin(P), RR * np.cos(T)
    import matplotlib
    import matplotlib.pyplot as plt
    if ax is None:
        fig = plt.figure(figsize=(5.2, 4.6))
        ax = fig.add_subplot(projection="3d")
    if getattr(ax, "name", "") != "3d":
        raise ValueError("a balloon is drawn on 3D axes (projection='3d')")
    cmap = matplotlib.colormaps[_cmap(s, d.quantity.kind if d.quantity.kind != "plain" else "spl")]
    norm = matplotlib.colors.Normalize(bottom, top)
    C = cmap(norm(np.repeat(L[:, None], n_phi, axis=1)))
    ax.plot_surface(X, Y, Z, facecolors=C, rstride=1, cstride=1, linewidth=0, antialiased=True, shade=False)
    m = float(R.max()) or 1.0
    ax.set_xlim(-m, m)
    ax.set_ylim(-m, m)
    ax.set_zlim(-m, m)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    if opts.get(s, "colorbar"):
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        ax.figure.colorbar(sm, ax=ax, shrink=0.6, pad=0.02, label=d.quantity.axis_label())
    ax.set_title(s.get("title") or f"{d.quantity.label}: balloon ({rng:g} dB range), axis vertical", fontsize=9)
    return ax


__all__ = ["prepare", "draw", "table", "table_rows", "bode", "nyquist", "balloon"]
