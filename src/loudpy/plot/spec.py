"""
A plot described as plain data (JSON): what to plot (a quantity, where), over which axes, at which values of the
other dimensions, shown how — resolved on results to labelled data (lp.Data) or a field (lp.Field) by slicing named
dimensions (the xarray idea). Every dimension that is not plotted becomes a *control* (a slider over numbers — a
frequency, a time, a mode, a sweep run, the phase of a harmonic field — or a choice among names), so a program can
generate its widgets, step through them or play them as an animation.

    spec = lp.plot.PlotSpec(quantity="spl", where={"kind": "arc", "radius": 0.5, "to": 180}, axes=["angle"],
                            view="polar", at={"f": 1000.0})
    out = spec.resolve(res)          Resolved: out.data (Data over angle), out.view "polar",
                                     out.controls [Control f: 200 values, index at 1000 Hz]
    out.controls[0].index = 10;  spec.at["f"] = out.controls[0].values[10]      (or spec.step(res, "f", +1))
    out.plot()                       the same in matplotlib (loudpy's style): a publication figure
    spec.to_dict()                   plain data, saved with a simulation; PlotSpec.from_dict(…) back

    PlotSpec(quantity="field", field="u", view="field", style={"show": "real"})     a harmonic field at one phase:
                                     controls entry (the solved axis) and phase (0…360°: play it to animate)

QUANTITIES lists what can be plotted (each says when results offer it); WHERE the places a quantity is evaluated
at (a point, a line, an arc, a named boundary or region: {"kind": "on", "name": "coil1", "reduce": "mean"}); VIEWS
the ways of showing it (curve, map, polar, table, field, view3d, bode, nyquist, balloon); quantities(res) what these
results offer. spec.style: how it is shown — the options of loudpy.plot.options (show, normalize, axis scales and
limits, line style, marker, colours, colormap and range, levels, contours, deformed shape …), checked here and
applied by Resolved.plot (loudpy.plot.draw); Resolved.table() the rows of a table.
"""
from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from loudpy.errors import ResultsError

VIEWS = ("curve", "map", "polar", "table", "field", "view3d", "bode", "nyquist", "balloon")
PHASES = tuple(float(x) for x in range(0, 360, 15))                  # the phase control's values [°]


@dataclass(frozen=True)
class Quantity:
    """A plottable quantity: its title, whether results offer it, and how it is evaluated (res, where, spec) →
    lp.Data | lp.Field; `where`: the places it takes (none: its own dimensions, e.g. an impedance over f)."""
    title: str
    offered: Callable
    evaluate: Callable
    where: tuple = ("point", "line", "arc", "on")
    components: tuple = ()


def _names(res) -> list:
    return list(getattr(res, "names", ()) or ())


def _forced(res) -> bool:
    return getattr(res, "kind", "") in ("frequency", "time")


def _component(spec):
    return spec.component or None


QUANTITIES: dict[str, Quantity] = {
    "spl": Quantity("SPL", lambda r: getattr(r, "kind", "") == "frequency" and "p" in _names(r),
                    lambda r, at, s: r.spl(at=at)),
    "pressure": Quantity("Pressure", lambda r: _forced(r) and "p" in _names(r), lambda r, at, s: r.pressure(at=at)),
    "displacement": Quantity("Displacement", lambda r: "u" in _names(r),                 # modes too: their shapes
                             lambda r, at, s: r.displacement(at=at, component=_component(s)), components=("z", "r")),
    "velocity": Quantity("Velocity", lambda r: _forced(r) and "u" in _names(r),
                         lambda r, at, s: r.velocity(at=at, component=_component(s)), components=("z", "r")),
    "acceleration": Quantity("Acceleration", lambda r: _forced(r) and "u" in _names(r),
                             lambda r, at, s: r.acceleration(at=at, component=_component(s)), components=("z", "r")),
    "impedance": Quantity("Electrical impedance", lambda r: getattr(r, "kind", "") == "frequency" and
                          "I_coil" in getattr(r.blocks[0], "extra", {}), lambda r, at, s: r.impedance(), where=()),
    "field": Quantity("A field at one entry", lambda r: bool(_names(r)), lambda r, at, s: _field(r, s), where=()),
}


def quantities(res) -> list[str]:
    """The quantities these results offer (one run of a sweep)."""
    out = []
    for name, q in QUANTITIES.items():
        try:
            if q.offered(res):
                out.append(name)
        except Exception:                                            # noqa: BLE001  (results without that part)
            continue
    return out


def fields(res) -> list[str]:
    """The FE fields these results offer to quantity "field": the stored ones, and the velocity "v" and acceleration
    "a" of a stored displacement of a harmonic result (jω u, −ω² u: Results.field)."""
    out = _names(res)
    if "u" in out and getattr(res, "kind", "") == "frequency" and not getattr(res, "real", False):
        out += [n for n in ("v", "a") if n not in out]
    return out


def where(spec_where: dict | None, dim: int):
    """The selection of a `where` (plain data) for results of dimension `dim` (2: r, z; 3: x, y, z):
    {"kind": "point", "point": [r, z]}, {"kind": "line", "start": […], "end": […], "n": 100},
    {"kind": "arc", "radius": R, "from": 0, "to": 90, "step": 2, "center": […]}, {"kind": "on", "name": "coil1",
    "reduce": None | "mean" | "integral" | "max"} (a named boundary or region: lp.On); None: no place."""
    from loudpy.results.selection import Arc, Line, On, Point
    w = dict(spec_where or {})
    kind = w.get("kind")
    if kind is None:
        return None
    if kind == "point":
        return Point(*[float(x) for x in w["point"]])
    if kind == "line":
        return Line(np.asarray(w["start"], float), np.asarray(w["end"], float), int(w.get("n", 100)))
    if kind == "on":
        return On(w["name"], w.get("reduce") or None)
    if kind == "arc":
        lo, hi, step = float(w.get("from", 0.0)), float(w.get("to", 90.0)), float(w.get("step", 2.0))
        center = w.get("center") or [0.0] * dim
        return Arc(float(w["radius"]), np.arange(lo, hi + 1e-9, step), np.asarray(center, float))
    raise ResultsError(f"where: kind point, line, arc or on, got {kind!r}")


def _axis(res) -> tuple[str, str, np.ndarray]:
    """(dimension, unit, values) of the results' solved axis: f [Hz] (frequencies, modes), t [s]."""
    kind = getattr(res, "kind", "")
    return ("mode" if kind == "modes" else "t" if kind == "time" else "f",
            "s" if kind == "time" else "Hz", np.asarray(res.values, float))


def _field(res, spec):
    """The FE field of the spec at the entry it names (index or nearest value)."""
    dim, _, values = _axis(res)
    k = _index(values, spec.at.get(dim if dim != "mode" else "mode"))
    if getattr(res, "kind", "") == "modes":
        return res.mode(k)
    return res.field(spec.field or ("p" if "p" in _names(res) else "u"), index=k)


def _index(values, at) -> int:
    """The position of `at` among values: {"index": k}, a value (the nearest), None (the first)."""
    if at is None or len(values) == 0:
        return 0
    if isinstance(at, dict):
        return int(np.clip(int(at.get("index", 0)), 0, len(values) - 1))
    vals = np.asarray(values)
    if vals.dtype.kind in "fiu":
        return int(np.argmin(np.abs(vals.astype(float) - float(at))))
    hit = np.flatnonzero(vals == at)
    return int(hit[0]) if len(hit) else 0


@dataclass
class Control:
    """A dimension not plotted: its name, label, unit, values (numbers or names) and the index shown; numeric
    controls are sliders (step, play), named ones choices."""
    dim: str
    label: str
    values: list
    index: int = 0
    unit: str = ""

    @property
    def numeric(self) -> bool:
        """Whether its values are numbers (a slider)."""
        return all(isinstance(v, (int, float, np.integer, np.floating)) for v in self.values)

    @property
    def value(self):
        """The value shown."""
        return self.values[self.index] if self.values else None

    def text(self, i: int | None = None) -> str:
        """A value as shown: "1000 Hz", "run 2", "cone"."""
        v = self.values[self.index if i is None else i]
        return (f"{v:.4g} {self.unit}".strip() if isinstance(v, (float, np.floating)) else
                f"{v} {self.unit}".strip())

    def to_dict(self) -> dict:
        """As plain data (the interface's widget)."""
        return {"dim": self.dim, "label": self.label, "unit": self.unit, "index": int(self.index),
                "values": [float(v) if isinstance(v, (float, np.floating)) else
                           int(v) if isinstance(v, (int, np.integer)) else str(v) for v in self.values],
                "texts": [self.text(i) for i in range(len(self.values))], "numeric": self.numeric}


@dataclass
class Resolved:
    """A spec resolved on results: the data (lp.Data, a list of them, or an lp.Field), the view, the controls of the
    dimensions not plotted, a title."""
    data: Any
    view: str
    controls: list
    title: str = ""
    style: dict = dataclasses.field(default_factory=dict)

    def control(self, dim: str) -> Control | None:
        """The control of a dimension (None: it has none)."""
        return next((c for c in self.controls if c.dim == dim), None)

    def plot(self, ax=None, **kwargs):
        """The same plot in matplotlib, loudpy's style, with the spec's style options (loudpy.plot.draw); returns
        the axes (bode: the two axes)."""
        from loudpy.plot.draw import draw
        return draw(self, ax=ax, style=self.style, **kwargs)

    def shown(self):
        """The data as the style shows it (show, normalize, mirror: loudpy.plot.draw.prepare); a field as it is."""
        from loudpy.plot.draw import prepare
        return self.data if hasattr(self.data, "mesh") else prepare(self.data, self.style)

    def table(self) -> tuple[list, list]:
        """(header, rows) of the data as shown, as text (the table view)."""
        from loudpy.plot.draw import table_rows
        from loudpy.plot.options import get
        return table_rows(self.shown(), int(get(self.style, "digits")))


@dataclass
class PlotSpec:
    """
    What to plot, as plain data (module docstring): quantity (QUANTITIES), where (a point, a line, an arc), component
    ("z" / "r" of a vector), field (the FE field of quantity "field"), axes (the dimensions plotted; default: the
    view's), at ({dimension: a value, or {"index": k}} for the others; "run" for a sweep, "phase" [°]), view
    (VIEWS), style (show: abs / real / spl / phase, scale, cut, cmap, range_db …: what the view uses), preset and
    params (a plot made by a program's own recipe: kept here, not resolved here), name.
    """
    quantity: str = "spl"
    where: dict = dataclasses.field(default_factory=dict)
    component: str = ""
    field: str = ""
    axes: list = dataclasses.field(default_factory=list)
    at: dict = dataclasses.field(default_factory=dict)
    view: str = ""
    style: dict = dataclasses.field(default_factory=dict)
    preset: str = ""
    params: dict = dataclasses.field(default_factory=dict)
    name: str = ""

    def to_dict(self) -> dict:
        """As plain data (JSON): only what differs from an empty spec, and the quantity."""
        out: dict = {"quantity": self.quantity}
        for k in ("where", "component", "field", "axes", "at", "view", "style", "preset", "params", "name"):
            v = getattr(self, k)
            if v not in ("", [], {}, None):
                out[k] = copy.deepcopy(v)
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "PlotSpec":
        """A spec from plain data (unknown keys refused)."""
        known = {f for f in cls.__dataclass_fields__}
        bad = set(d) - known
        if bad:
            raise ResultsError(f"PlotSpec: unknown keys {sorted(bad)} (keys: {sorted(known)})")
        return cls(**copy.deepcopy(d))

    # ── resolving ──────────────────────────────────────────────────────────
    def resolve(self, res) -> Resolved:
        """The data this spec shows on `res` (a sweep: the run at["run"]) and the controls of every dimension not
        plotted."""
        from loudpy.plot.options import check
        try:
            style = check(self.style, self.view)
        except ValueError as e:
            raise ResultsError(str(e)) from None
        controls: list = []
        if getattr(res, "is_sweep", False):
            runs = [", ".join(f"{k} = {v:g}" for k, v in r.items()) for r in res.runs]
            k = _index(list(range(len(runs))), self.at.get("run", {"index": 0}))
            controls.append(Control("run", "run", list(range(len(runs))), k))
            if not len(res):
                raise ResultsError("the sweep has no run")
            res = res.results[k]
        if self.quantity not in QUANTITIES:
            raise ResultsError(f"no quantity {self.quantity!r}; quantities: {list(QUANTITIES)}")
        q = QUANTITIES[self.quantity]
        if self.quantity not in quantities(res):
            raise ResultsError(f"these {getattr(res, 'kind', '')} results do not offer {self.quantity!r} "
                               f"(they offer {quantities(res)})")
        if self.quantity == "field":
            out = self._field(res, controls)
            out.style = style
            return out
        dim = _dim(res)
        sel = where(self.where, dim) if q.where else None
        if q.where and sel is None:
            raise ResultsError(f"{self.quantity}: where is it evaluated? (where: {list(q.where)})")
        d = q.evaluate(res, sel, self)
        view = self.view or ("polar" if "angle" in d.dims and len(d.dims) == 2 and self.axes == ["angle"] else "")
        axes = list(self.axes) or self._default_axes(d.dims, view)
        bad = [a for a in axes if a not in d.dims]
        if bad:
            raise ResultsError(f"axes {bad} are not dimensions of {self.quantity} here ({list(d.dims)})")
        for name in [n for n in d.dims if n not in axes]:
            values = list(np.asarray(d.coords[name]).tolist())
            k = _index(values, self.at.get(name))
            unit = {"f": "Hz", "t": "s", "s": "m", "angle": "°"}.get(name, "")
            controls.append(Control(name, {"f": "frequency", "t": "time", "s": "distance", "angle": "angle",
                                           "node": "node"}.get(name, name), values, k, unit))
            d = d.isel(**{name: k})
        view = view or {0: "table", 1: "polar" if axes == ["angle"] and self.view == "polar" else "curve",
                        2: "map"}.get(len(axes), "table")
        if view not in VIEWS:
            raise ResultsError(f"view: one of {VIEWS}, got {view!r}")
        title = f"{q.title}" + (f" at {sel.describe()}" if sel is not None else "")
        return Resolved(d, view, controls, title, style)

    @staticmethod
    def _default_axes(dims, view) -> list:
        if view in ("polar", "balloon") and "angle" in dims:
            return ["angle"]
        if view == "map" and len(dims) >= 2:
            return list(dims[:2])
        return list(dims[:1])

    def _field(self, res, controls) -> Resolved:
        dim, unit, values = _axis(res)
        k = _index(list(values), self.at.get(dim))
        controls.append(Control(dim, {"f": "frequency", "t": "time", "mode": "mode"}[dim], list(values), k, unit))
        spec = copy.copy(self)
        spec.at = {**self.at, dim: {"index": k}}
        f = _field(res, spec)
        if self.component and f.is_vector:
            f = f.component(self.component)
        show = self.style.get("show", "")
        if show == "real" and np.iscomplexobj(f.values):              # one instant: the phase is a control
            j = _index(list(PHASES), self.at.get("phase", 0.0))
            controls.append(Control("phase", "phase", list(PHASES), j, "°"))
            amplitude = float(np.abs(np.asarray(f.values)).max())      # (one scale for every phase: a deformed
            f = f.at_phase(np.deg2rad(PHASES[j]))                      # shape swings with the motion)
            f.meta["amplitude"] = amplitude
        what = f"{f.quantity.name} at {values[k]:.4g} {unit}" if dim != "mode" else f"mode {k} ({values[k]:.4g} Hz)"
        return Resolved(f, self.view or "field", controls, what)

    def step(self, res, dim: str, by: int = 1, wrap: bool = True) -> "PlotSpec":
        """The spec with the control `dim` moved by `by` positions (wrapping round, as a player does)."""
        out = self.resolve(res)
        c = out.control(dim)
        if c is None:
            raise ResultsError(f"no control {dim!r} (controls: {[x.dim for x in out.controls]})")
        n = len(c.values)
        i = (c.index + by) % n if wrap else int(np.clip(c.index + by, 0, n - 1))
        new = copy.deepcopy(self)
        new.at[dim] = {"index": i}
        return new

    def animate(self, res, dim: str, path, fps: float = 6.0, **kwargs):
        """Play the control `dim` (every value in turn: the frequencies, the times, the runs, the phase of a
        harmonic field …) into an animation: .gif (Pillow) or .mp4 (ffmpeg from imageio-ffmpeg); each frame is the
        spec resolved at that value and drawn by Resolved.plot (kwargs: passed to it). Returns the path."""
        import matplotlib.pyplot as plt
        from matplotlib import animation
        from loudpy.plot.animate import writer as _movie_writer
        from loudpy.plot import style
        first = self.resolve(res)
        c = first.control(dim)
        if c is None:
            raise ResultsError(f"no control {dim!r} to animate (controls: {[x.dim for x in first.controls]})")
        polar = first.view == "polar"
        with style.context():
            fig = plt.figure(figsize=(6.4, 4.8))
            ax = fig.add_subplot(projection="polar" if polar else None)

            def draw(i):
                fig.clear()
                axes = fig.add_subplot(projection="polar" if polar else None)
                spec = copy.deepcopy(self)
                spec.at[dim] = {"index": i}
                out = spec.resolve(res)
                out.plot(ax=axes, **kwargs)
                axes.set_title(f"{c.label} = {c.text(i)}", fontsize=10)
                return []
            del ax
            anim = animation.FuncAnimation(fig, draw, frames=len(c.values), blit=False)
            anim.save(str(path), writer=_movie_writer(path, fps))
            plt.close(fig)
        return path


def _dim(res) -> int:
    """The coordinate dimension of the results' meshes (2: r, z; 3: x, y, z)."""
    try:
        return int(np.asarray(next(iter(res.blocks[0].meshes.values())).coords).shape[1])
    except Exception:                                                # noqa: BLE001  (no mesh: a 3D scene's air)
        return 3


__all__ = ["PlotSpec", "Resolved", "Control", "Quantity", "QUANTITIES", "VIEWS", "PHASES", "quantities", "where"]
