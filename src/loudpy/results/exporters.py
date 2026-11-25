"""
Exports, by the file's suffix: one lp.export(obj, path) for everything loudpy writes for other programs — the
interface calls the same function. A new format is one entry (register).

    lp.export(res.spl(at=pt), "spl.csv")           Data: .csv .txt .dat · .nc · .mat · .npz
    lp.export([d1, d2], "curves.csv")              several curves over the same axis: .csv · .mat · .npz (a column each)
    lp.export(rows, "modes.csv")                   a table (a list of dicts): .csv
    lp.export(res.field("p", f=1000), "p.vtu")     Field: .vtu .vtk · .html · .csv .txt · .mat · .npz · .gif .mp4
    lp.export(res, "all.mat")                      Results: .mat · .npz · .pvd · .vtkhdf · .h5 (loudpy's own)
    lp.export(view, "scene.glb")                   View3D: .png .jpg .tif .pdf .svg .eps · .html · .gltf .glb · .vtm
    lp.export(ax.figure, "fig.pdf")                a matplotlib figure (or axes): .pdf .png .svg .eps
    lp.export(trans, "u.mp4")                      TimeResults: .gif .mp4 (lp.plot.animate)
    formats(obj)                                   the suffixes offered for an object

Every writer goes through a temporary file renamed at the end (safe on synced folders); folders are made.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Exporter:
    """One format of one kind of object: its suffixes, which objects it takes, how it writes, what it is."""
    suffixes: tuple
    accepts: Callable
    write: Callable
    what: str


EXPORTERS: list[Exporter] = []


def register(suffixes, accepts: Callable, what: str):
    """Decorator: a writer write(obj, path, **kwargs) for the objects `accepts(obj)` takes, under these suffixes."""
    def deco(fn):
        EXPORTERS.append(Exporter(tuple(s.lower() for s in suffixes), accepts, fn, what))
        return fn
    return deco


def exporters_for(obj) -> list[Exporter]:
    """The exporters that take this object."""
    out = []
    for e in EXPORTERS:
        try:
            if e.accepts(obj):
                out.append(e)
        except Exception:                                            # noqa: BLE001  (an object it cannot inspect)
            continue
    return out


def formats(obj) -> list[str]:
    """The suffixes offered for an object."""
    return [s for e in exporters_for(obj) for s in e.suffixes]


def export(obj, path, **kwargs):
    """Write an object to a file by its suffix (module docstring); returns the path (the paths, for a series)."""
    path = Path(path)
    suffix = path.suffix.lower()
    for e in exporters_for(obj):
        if suffix in e.suffixes:
            path.parent.mkdir(parents=True, exist_ok=True)
            return e.write(obj, path, **kwargs)
    offered = formats(obj)
    raise ValueError(f"cannot export a {type(obj).__name__} as {suffix or 'a file without suffix'!r}; offered: "
                     f"{', '.join(offered) or 'none'}")


# ── which objects ──────────────────────────────────────────────────────────────────────────────────────────────────
def _is(name):
    def check(obj) -> bool:
        return any(c.__name__ == name for c in type(obj).__mro__)
    return check


_data, _field, _results, _sweep, _view = (_is("Data"), _is("Field"), _is("Results"), _is("SweepResults"),
                                          _is("View3D"))


def _curves(obj) -> bool:
    return isinstance(obj, (list, tuple)) and bool(obj) and all(_data(d) and d.ndim == 1 for d in obj)


def _table(obj) -> bool:
    return isinstance(obj, list) and all(isinstance(r, dict) for r in obj)


def _figure(obj) -> bool:
    return any(c.__name__ in ("Figure", "Axes") and c.__module__.startswith("matplotlib") for c in type(obj).__mro__)


def _animated(obj) -> bool:
    return (_field(obj) and np.iscomplexobj(np.asarray(obj.values))) or getattr(obj, "kind", "") == "time"


# ── Data ───────────────────────────────────────────────────────────────────────────────────────────────────────────
register((".csv", ".txt", ".dat"), _data, "text table")(lambda d, path: d.to_csv(path))
register((".nc", ".cdf", ".netcdf"), _data, "netCDF")(lambda d, path: d.to_netcdf(path))
register((".mat",), _data, "MATLAB")(lambda d, path: d.to_mat(path))
register((".npz",), _data, "NumPy")(lambda d, path: d.to_npz(path))


@register((".csv", ".mat", ".npz"), _curves, "curves over one axis (a column each)")
def write_curves(datas, path):
    """Several curves over the same axis: one column each (complex: real and imaginary parts)."""
    from loudpy.results.data import write_atomic
    from loudpy.results.netcdf import slug
    dim = datas[0].dims[0]
    cols = {dim: np.asarray(datas[0].coords[dim], float)}
    for d in datas:
        v = np.asarray(d.values)
        name = slug(d.label or d.quantity.name) or "curve"
        if np.iscomplexobj(v):
            cols[f"{name}_re"], cols[f"{name}_im"] = v.real, v.imag
        else:
            cols[name] = v
    if path.suffix.lower() == ".csv":
        def write(tmp):
            with open(tmp, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(list(cols))
                w.writerows(np.column_stack(list(cols.values())).tolist())
    elif path.suffix.lower() == ".mat":
        from scipy.io import savemat

        def write(tmp):
            savemat(tmp, cols)
    else:
        def write(tmp):
            np.savez(tmp, **cols)  # type: ignore[arg-type]
    return write_atomic(path, write)


@register((".csv",), _table, "table")
def write_table(rows, path):
    """A table (a list of dicts, the keys of the first row as columns) as CSV."""
    from loudpy.results.data import write_atomic

    def write(tmp):
        with open(tmp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["empty"])
            w.writeheader()
            w.writerows(rows)
    return write_atomic(path, write)


# ── Field ──────────────────────────────────────────────────────────────────────────────────────────────────────────
@register((".vtu", ".vtk"), _field, "ParaView (mesh + values)")
def write_field_vtu(f, path):
    """A field on its mesh for ParaView."""
    from loudpy.results.data import write_atomic
    return write_atomic(path, lambda tmp: f.to_pyvista().cast_to_unstructured_grid().save(tmp))


@register((".html", ".htm"), _field, "interactive viewer")
def write_field_html(f, path, show=None):
    """A self-contained interactive viewer (lp.plot.html)."""
    from loudpy.plot.html import html
    return html(f, path) if show is None else html(f, path, show=show)


@register((".csv", ".txt"), _field, "node table")
def write_field_table(f, path):
    """One row per node: its coordinates and values (re / im for complex)."""
    from loudpy.results.data import write_atomic
    cols = f.table()
    head = "\n".join(f"{k}: {v}" for k, v in f.header().items())
    return write_atomic(path, lambda tmp: np.savetxt(tmp, np.column_stack(list(cols.values())), delimiter=",",
                                                     fmt="%.10g", header=head + "\n" + ",".join(cols),
                                                     comments="# "))


@register((".mat",), _field, "MATLAB (coordinates, triangles, values)")
def write_field_mat(f, path):
    """Coordinates, triangles (1-based), node tags and values for MATLAB / Octave."""
    from scipy.io import savemat
    from loudpy.results.data import mat_name, write_atomic
    tri = f.mesh.triangles(refine=True) if f.mesh.dim == 2 else np.zeros((0, 3), int)
    d = {mat_name(k): v for k, v in f.table().items()}
    d.update(values=f.values, triangles=tri + 1, node_tags=f.mesh.node_tags, info=f.header())
    return write_atomic(path, lambda tmp: savemat(tmp, d))


@register((".npz",), _field, "NumPy")
def write_field_npz(f, path):
    """Values, coordinates, triangles, node tags and the header (JSON) for NumPy."""
    import json
    from loudpy.results.data import write_atomic
    tri = f.mesh.triangles(refine=True) if f.mesh.dim == 2 else np.zeros((0, 3), int)
    return write_atomic(path, lambda tmp: np.savez(tmp, values=f.values, coords=f.mesh.coords, triangles=tri,
                                                    node_tags=f.mesh.node_tags,
                                                    header=np.array(json.dumps(f.header()))))


@register((".gif", ".mp4"), _animated, "animation")
def write_animation(obj, path, **kwargs):
    """A harmonic field over a period, or the steps of a transient (lp.plot.animate)."""
    from loudpy.plot.animate import animate
    if _field(obj):
        kwargs.setdefault("name", obj.name)
    return animate(obj, path, **kwargs)


# ── Results ────────────────────────────────────────────────────────────────────────────────────────────────────────
@register((".mat", ".npz"), lambda r: _results(r) and not _sweep(r), "every block (arrays)")
def write_results(res, path, fields=None):
    """Every block of a result (meshes, fields, per-entry quantities)."""
    from loudpy.results.export import export_results
    return export_results(res, path, fields)


@register((".pvd",), lambda r: _results(r) and not _sweep(r), "ParaView series (.vtu per entry)")
def write_series(res, path, fields=None):
    """A ParaView series: one .vtu per entry and mesh, the frequency / time / mode as the time step."""
    from loudpy.results.export import export_series
    return export_series(res, path, fields)


@register((".vtkhdf", ".hdf"), lambda r: _results(r) and not _sweep(r) and getattr(r, "kind", "") != "modes",
          "VTKHDF series (ParaView)")
def write_vtkhdf(res, path, fields=None):
    """A VTKHDF time / frequency series (one file per block and mesh when several)."""
    from loudpy.results.export import export_vtkhdf
    return export_vtkhdf(res, path, fields)


@register((".h5",), lambda r: _results(r) or _sweep(r), "loudpy's own result file")
def write_h5(res, path):
    """loudpy's result file (format 3: lp.load reads it back)."""
    return res.save(path)


# ── views and figures ──────────────────────────────────────────────────────────────────────────────────────────────
@register((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".pdf", ".svg", ".eps", ".html", ".gltf", ".glb", ".vtm",
           ".vtmb"), _view, "3D view (figure, viewer, 3D model, datasets)")
def write_view(view, path, **kwargs):
    """A 3D view (View3D.save)."""
    return view.save(path, **kwargs)


@register((".pdf", ".png", ".svg", ".eps"), _figure, "figure")
def write_figure(fig, path, **kwargs):
    """A matplotlib figure (or the figure of axes), loudpy's way (lp.plot.save: folders made, atomic)."""
    from loudpy.plot import style
    return style.save(getattr(fig, "figure", fig), path, **kwargs)


__all__ = ["Exporter", "EXPORTERS", "register", "export", "formats", "exporters_for", "write_curves", "write_table"]
