"""
loudpy.plot — plots that take Results (lp.load or a study), not raw arrays.

    lp.plot.field(res, "p", f=1000, component="spl")     a field on its mesh at one entry
    lp.plot.mode(modes, 3)                              shape of mode 3
    lp.plot.spl(res, at=(0.0, 0.3))                     SPL at a point over frequency
    lp.plot.response(res, at=(0.0, 0.02), name="u")     displacement at a point over frequency
    lp.plot.history(trans, at=(0.0, 0.02), name="u")    a quantity at a point over time

component: "mag" (default: |value|, norm of the moduli for vectors), "r", "z", "theta" (modulus
of one displacement component), "real", "imag", "phase", "spl" (pressure in dB re 20 µPa RMS).
Every function draws on `ax` (a new figure without it) and returns the axes.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from loudpy.results.results import FieldView, spl as _spl

_COMPONENT = {"r": 0, "z": 1, "theta": 2}


def _scalar(values: np.ndarray, component: str) -> np.ndarray:
    v = np.asarray(values)
    if component == "spl":
        return _spl(v if v.ndim == 1 else np.linalg.norm(np.abs(v), axis=1))
    if component in _COMPONENT:
        if v.ndim == 1:
            raise ValueError(f"component {component!r} needs a vector field")
        return np.abs(v[:, _COMPONENT[component]])
    if component == "real":
        return (v if v.ndim == 1 else v[:, np.argmax(np.abs(v).max(axis=0))]).real
    if component == "imag":
        return (v if v.ndim == 1 else v[:, np.argmax(np.abs(v).max(axis=0))]).imag
    if component == "phase":
        if v.ndim == 1:
            return np.angle(v)
        dom = np.argmax(np.abs(v), axis=1)
        return np.angle(v[np.arange(len(v)), dom])
    if component == "mag":
        a = np.abs(v)
        return a if a.ndim == 1 else np.linalg.norm(a, axis=1)
    raise ValueError(f"unknown component {component!r}")


def _view(res, name, select) -> FieldView:
    if isinstance(res, FieldView):
        return res
    return res.field(name, **{k: v for k, v in select.items() if v is not None})


def field(res, name: str = "u", component: str = "mag", ax=None, cmap=None, title=None, **select):
    """A field at one entry: select with f= (frequency), t= (time) or index= (any results)."""
    from loudpy.Plotter.field_plots import plot_field
    view = _view(res, name, select)
    values = _scalar(view.values, component)
    default_cmap = "magma" if view.name == "p" else "viridis"
    unit = {"spl": "SPL [dB]", "phase": "phase [rad]"}.get(component, f"{component} {view.name}")
    ax = plot_field(view.mesh.coords, view.triangles(), values, cmap=cmap or default_cmap, ax=ax,
                    label=unit, title=title or f"{view.name} at {view.coordinate:.4g}",
                    xlabel="r [m]", ylabel="z [m]")
    return ax


def mode(modes, k: int, name: str = "u", component: str = "mag", ax=None, **kw):
    """Shape of mode k of a ModeResults."""
    view = modes.mode(k, name)
    zeta = view.extra.get("zeta")
    title = f"mode {k}: {view.coordinate:.1f} Hz" + (f", ζ = {zeta:.3g}" if zeta is not None and np.isfinite(zeta) else "")
    return field(view, name, component, ax=ax, title=title, **kw)


def _curve(x, y, ax, label, xlabel, ylabel, logx=True):
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.5))
    (ax.semilogx if logx else ax.plot)(x, y, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.3)
    if label:
        ax.legend()
    return ax


def spl(res, at, ax=None, label=None, rms: bool = True):
    """SPL at the node nearest `at`, over frequency."""
    return _curve(res.frequencies, res.spl(at, rms=rms), ax, label, "f [Hz]",
                  f"SPL [dB re 20 µPa{' RMS' if rms else ''}]")


def response(res, at, name: str = "u", component: str = "z", ax=None, label=None, db: bool = False):
    """|value| of a field (component) at the node nearest `at`, over frequency."""
    v = res.probe(at, name)
    if v.ndim == 1:
        y, ylabel = np.abs(v), f"|{name}|"
    elif component in _COMPONENT:
        y, ylabel = np.abs(v[:, _COMPONENT[component]]), f"|{name}_{component}|"
    else:
        y, ylabel = np.linalg.norm(np.abs(v), axis=1), f"|{name}|"
    if db:
        y, ylabel = 20 * np.log10(np.maximum(y, 1e-300)), ylabel + " [dB]"
    return _curve(res.frequencies, y, ax, label, "f [Hz]", ylabel)


def history(res, at, name: str = "u", component: str = "z", ax=None, label=None):
    """A field (component, real) at the node nearest `at`, over time."""
    v = res.probe(at, name)
    y = np.real(v[:, _COMPONENT[component]]) if v.ndim > 1 else np.real(v)
    return _curve(res.times, y, ax, label, "t [s]", f"{name}_{component}", logx=False)


def modes_grid(modes, indices=None, n: int | None = None, ncols: int = 4, deform_scale: float = 0.02,
               name: str = "u"):
    """Grid of mode shapes (|u| on the deformed mesh) of one mode set (e.g. modes.order(0))."""
    from loudpy.Plotter.field_plots import plot_modes_grid
    if len(modes.blocks) != 1:
        raise ValueError(f"pick one mode set first (sets: {modes.sets}), e.g. modes.order(0)")
    b = modes.blocks[0]
    mesh = b.mesh_of(name)
    kw = dict(mode_indices=list(indices)) if indices is not None else dict(n_plot=n or len(b))
    return plot_modes_grid(mesh.coords, mesh.triangles(), b.fields[name], b.values,
                           zetas=b.extra.get("zeta"), ncols=ncols, deform_scale=deform_scale, **kw)


def deformed_boundary(view: FieldView, tags, scale: float = 0.1, title: str | None = None, ax=None):
    """Rest and deformed positions of the nodes `tags` (e.g. res.boundary(name)) for a displacement view."""
    from loudpy.Plotter.field_plots import plot_interface_deformed
    coords, u = view.at_nodes(tags)
    return plot_interface_deformed(coords[:, 0], coords[:, 1], u[:, 0], u[:, 1], scale=scale,
                                   title=title or f"{view.name} at {view.coordinate:.4g}", ax=ax)


def coupled_modes_grid(modes, indices, labels=None, **kwargs):
    """
    Coupled structure + air modes of one mode set (e.g. modes.set("fsi")): pressure in the air and
    deformed structure side by side (keyword arguments: Plotter.plot_coupled_modes_grid).
    """
    from loudpy.Plotter.field_plots import plot_coupled_modes_grid
    if len(modes.blocks) != 1:
        raise ValueError(f"pick one mode set first (sets: {modes.sets})")
    b = modes.blocks[0]
    ma, mm = b.meshes["air"], b.meshes["solid"]
    ta, tm = ma.triangles(), mm.triangles()
    frames = [dict(freq=float(b.values[k]), zeta=float(b.extra.get("zeta", [np.nan] * len(b))[k]), index=k,
                   coords_a=ma.coords, tris_a=ta, p=b.fields["p"][k],
                   coords_m=mm.coords, tris_m=tm, u=b.fields["u"][k],
                   meta={n: v[k] for n, v in b.extra.items()})
              for k in range(len(b))]
    return plot_coupled_modes_grid(frames, mode_indices=list(indices), labels=labels, **kwargs)


__all__ = ["field", "mode", "spl", "response", "history", "modes_grid", "deformed_boundary",
           "coupled_modes_grid"]
