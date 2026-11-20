"""
Plots of Data (loudpy.results.Data): curves, colour maps and polar diagrams.

    line(d)       y over the first coordinate; 2D data: one curve per value of the second
    map(d)        2D data as a colour map over both coordinates (e.g. frequency × angle)
    polar(d)      data over "angle" on a polar diagram (2D: one curve per value of the other coordinate)
    data(d)       picks one of them (Data.plot)

Conventions: frequency on a log axis (20, 50, 100 … 1k ticks); complex amplitudes shown as their
modulus; moduli of physical amplitudes (displacement, pressure, impedance …) on a log axis, levels
(dB), phases and ratios on a linear one; labels "Name $symbol$ [unit]" from the quantity.
"""
from __future__ import annotations

import numpy as np

from loudpy.plot import style
from loudpy.results.quantities import quantity

MAX_LINES = 12
REL_RANGE = (-40.0, 6.0)                  # default colour / radial range of relative levels [dB]


def _new_axes(ax, size="wide", **kw):
    if ax is None:
        _, ax = style.figure(size, **kw)
    return ax


def _shown(d):
    """The real data drawn for d (modulus of complex amplitudes)."""
    return d.abs() if np.iscomplexobj(d.values) else d


def _auto_logy(d) -> bool:
    """Log axis for frequency-domain moduli spanning > 1.5 decades; linear for signals, levels, phases."""
    q = d.quantity
    if d.dims[0] != "f" or q.unit.startswith("dB") or q.kind in ("phase", "level") or q.unit in ("", "%", "°", "rad"):
        return False
    v = np.asarray(d.values, float)
    v = v[np.isfinite(v)]
    if len(v) < 2 or np.any(v < 0):                     # signed signals stay linear
        return False
    v = v[v > 0]
    return len(v) > 1 and np.log10(v.max()) - np.log10(v.min()) > 1.5


def _coord_label(dim, value) -> str:
    """Legend text of one value of a coordinate: "θ = 30°", "m = 2", the point label …"""
    q = quantity(dim)
    if dim in ("point", "component"):
        return {"r": "$r$", "z": "$z$", "theta": r"$\theta$"}.get(str(value), str(value))
    v = f"{value:.4g}" if np.isscalar(value) and np.asarray(value).dtype.kind in "fiu" else str(value)
    unit = "" if not q.unit else (q.unit if q.unit == "°" else f" {q.unit}")
    return f"${q.symbol}$ = {v}{unit}" if q.symbol else f"{v}{unit}"


def _x_axis(ax, dim, x, logx):
    if logx is None:
        logx = dim == "f" and np.all(x > 0) and x.max() / max(x.min(), 1e-30) > 8
    if logx and dim == "f":
        style.frequency_axis(ax, label=False)
    elif logx:
        ax.set_xscale("log")
    ax.set_xlabel(quantity(dim).axis_label() if dim not in ("r", "z") else style.LABELS[dim])


def line(d, ax=None, label=None, logx=None, logy=None, title=None, legend=None, **kwargs):
    """
    Curves of Data (or a list of Data on the same axes); returns the axes. 2D data draw one
    curve per value of their second coordinate (at most MAX_LINES; use .sel() first otherwise).
    label: legend of a 1D curve (default: the data label); kwargs go to matplotlib's plot.
    """
    if isinstance(d, (list, tuple)):
        for k in d:
            ax = line(k, ax=ax, logx=logx, logy=logy, legend=False, **kwargs)
        if any(k.label for k in d):
            ax.legend()
        if title:
            ax.set_title(title)
        return ax
    ax = _new_axes(ax)
    s = _shown(d)
    dims = list(s.coords)
    x = s.coords[dims[0]]
    if s.ndim == 1:
        ax.plot(x, s.values, label=label if label is not None else (d.label or None), **kwargs)
    elif s.ndim == 2:
        other = s.coords[dims[1]]
        if len(other) > MAX_LINES:
            raise ValueError(f"{len(other)} curves: select some first (.sel({dims[1]}=[…])) or plot a map")
        for j, v in enumerate(other):
            lab = _coord_label(dims[1], v)
            if d.label:
                lab = f"{d.label}, {lab}"
            ax.plot(x, s.values[:, j], label=lab, **kwargs)
    else:
        raise ValueError("line() takes 1D or 2D data")
    _x_axis(ax, dims[0], x, logx)
    if logy if logy is not None else _auto_logy(s):
        ax.set_yscale("log")
    ax.set_ylabel(s.quantity.axis_label())
    if legend if legend is not None else (s.ndim == 2 or bool(d.label or label)):
        ax.legend()
    if title:
        ax.set_title(title)
    return ax


def map(d, ax=None, levels=None, vmin=None, vmax=None, cmap=None, logx=None, colorbar=True, title=None,
        contours=None, **kwargs):
    """
    2D data as a colour map (first coordinate on x, second on y); returns the axes.
    levels: filled contour levels (contourf) instead of a smooth map; contours: line levels drawn
    on top (e.g. [-6, -3] for a normalised directivity); vmin / vmax / cmap: colour scale.
    """
    ax = _new_axes(ax)
    s = _shown(d)
    if s.ndim != 2:
        raise ValueError("map() takes 2D data")
    dims = list(s.coords)
    x, y = s.coords[dims[0]].astype(float), s.coords[dims[1]].astype(float)
    cmap = cmap or style.CMAPS.get(s.quantity.kind, "viridis")
    if s.quantity.kind == "level" and vmin is None and vmax is None:          # relative levels: −40 … +6 dB
        vmin, vmax = REL_RANGE
    Z = s.values.T
    if levels is not None:
        cs = ax.contourf(x, y, Z, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax, extend="both", **kwargs)
        extend = None
    else:
        cs = ax.pcolormesh(x, y, Z, cmap=cmap, vmin=vmin, vmax=vmax, shading="gouraud", rasterized=True, **kwargs)
        lo, hi = np.nanmin(Z), np.nanmax(Z)
        extend = {(True, True): "both", (True, False): "min", (False, True): "max"}.get(
            (vmin is not None and lo < vmin, vmax is not None and hi > vmax), "neither")
    if contours is not None:
        cl = ax.contour(x, y, Z, levels=contours, colors="k", linewidths=0.6, linestyles="solid")
        ax.clabel(cl, fmt="%g dB" if s.quantity.unit.startswith("dB") else "%g", fontsize=6.5)
    _x_axis(ax, dims[0], x, logx)
    ax.set_ylabel(quantity(dims[1]).axis_label())
    ax.grid(False, which="both")
    if colorbar:
        ax.figure.colorbar(cs, ax=ax, label=s.quantity.axis_label(), pad=0.015,
                           **({"extend": extend} if extend else {}))
    if title:
        ax.set_title(title)
    return ax


def polar(d, ax=None, rlim=None, title=None, zero="N", clockwise=True, **kwargs):
    """
    Data over "angle" [°] on a polar diagram (2D: one curve per value of the other coordinate, e.g.
    d.sel(f=[100, 1000, 5000])). rlim: radial limits (default: −40 … +6 dB for relative levels, 40 dB
    below the maximum for absolute levels); values below the inner limit are drawn at the centre.
    """
    if ax is None:
        _, ax = style.figure("square", subplot_kw=dict(projection="polar"))
    s = _shown(d)
    dims = list(s.coords)
    if "angle" not in dims:
        raise ValueError("polar() needs an 'angle' dimension")
    ia = dims.index("angle")
    ang = np.deg2rad(s.coords["angle"].astype(float))
    V = np.moveaxis(s.values, ia, 0)
    if rlim is None and s.quantity.unit.startswith("dB"):
        rlim = REL_RANGE if s.quantity.kind == "level" else (np.nanmax(V) - 40, np.nanmax(V) + 3)
    if rlim is not None:
        V = np.clip(V, rlim[0], None)                    # below the centre: drawn at the centre
    if V.ndim == 1:
        ax.plot(ang, V, label=d.label or None, **kwargs)
    else:
        other = [k for k in dims if k != "angle"][0]
        for j, v in enumerate(s.coords[other]):
            ax.plot(ang, V[:, j], label=_coord_label(other, v), **kwargs)
        ax.legend(loc="lower left", bbox_to_anchor=(-0.15, -0.15), fontsize=7)
    ax.set_theta_zero_location(zero)
    ax.set_theta_direction(-1 if clockwise else 1)
    if rlim is not None:
        ax.set_rlim(*rlim)
    ax.set_title(title or s.quantity.axis_label(), fontsize=9)
    return ax


def data(d, ax=None, kind=None, **kwargs):
    """Plot Data with the natural kind: polar / map / line (see Data.plot)."""
    if kind is None:
        if d.ndim == 2:
            n_other = d.shape[1]
            kind = "line" if n_other <= MAX_LINES and list(d.coords)[1] in ("point", "component", "order", "angle") \
                or n_other <= 6 else "map"
        else:
            kind = "line"
    if kind == "line":
        return line(d, ax=ax, **kwargs)
    if kind == "map":
        return map(d, ax=ax, **kwargs)
    if kind == "polar":
        return polar(d, ax=ax, **kwargs)
    raise ValueError(f"unknown kind {kind!r}: 'line', 'map' or 'polar'")
