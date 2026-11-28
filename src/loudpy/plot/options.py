"""
How a plot is shown: the style options of lp.plot.PlotSpec (spec.style), declared once — their kind, choices,
default, the views they apply to and a hint — so a program builds its form from them and a spec is checked
against them (unknown or wrong options refused).

    options("curve")             the options of a view, in order (Option: name, kind, choices, default, hint)
    check({"yscale": "log"}, "curve")      the style cleaned and checked (ValueError on a bad option)
    STYLE_OPTIONS                every option

What they cover (COMSOL plot settings, Ansys result display, ParaView display properties):
  data      show (modulus, real, imaginary, phase, dB, SPL), normalize (to the maximum or the first value),
            unwrap (phase), mirror (both sides of the axis)
  axes      xscale / yscale (linear, log), limits, grid, title, legend
  lines     line style (solid, dashed, dotted, dash-dot, none: points only), marker, width, colour, palette
  colours   colormap (+ reversed), colour range (vmin / vmax, symmetric about 0, a dB range), colour scale (log),
            discrete bands (levels), isolines (contours), colour bar
  fields    deformed shape (+ scale), element edges, region outlines
  polar     zero direction, clockwise
  table     digits
"""
from __future__ import annotations

from dataclasses import dataclass

CURVES = ("curve", "bode", "nyquist")
LINES = ("curve", "polar", "bode", "nyquist")
COLOURED = ("map", "field", "balloon")
ALL = ("curve", "map", "polar", "table", "field", "view3d", "bode", "nyquist", "balloon")

CMAPS = ("auto", "viridis", "magma", "inferno", "plasma", "cividis", "turbo", "jet", "rainbow", "coolwarm",
         "RdBu_r", "seismic", "twilight", "Greys", "gray", "hot", "Spectral")
PALETTES = ("loudpy", "tab10", "Dark2", "Set1", "gray", "viridis")


@dataclass(frozen=True)
class Option:
    """One style option: name, kind ("choice" | "number" | "bool" | "text"), choices, default (None: automatic),
    the views it applies to, a hint."""
    name: str
    kind: str
    default: object = None
    choices: tuple = ()
    views: tuple = ALL
    hint: str = ""

    def applies(self, view: str) -> bool:
        """Whether the option means something for this view."""
        return view in self.views


STYLE_OPTIONS: tuple[Option, ...] = (
    # data
    Option("show", "choice", "auto", ("auto", "abs", "real", "imag", "phase", "db", "spl", "instant", "r", "z",
                                      "theta"),
           ("curve", "map", "polar", "table", "field", "balloon"),
           "what of a complex amplitude: its modulus, real or imaginary part, phase [°], level 20 log10|x| [dB], "
           "or the SPL [dB re 20 µPa] of a pressure (auto: SPL for a pressure, the modulus otherwise); fields "
           "also: instant (the real field at the phase of its peak) and a component r, z, theta"),
    Option("normalize", "choice", "none", ("none", "max", "first"), ("curve", "map", "polar", "table", "balloon"),
           "relative to the maximum, or to the first value along the plotted axis (e.g. on the axis of a "
           "directivity): divided by it, or subtracted from a level in dB"),
    Option("unwrap", "bool", True, (), ("curve", "map", "table", "bode"), "phase without its 360° jumps"),
    Option("mirror", "bool", False, (), ("polar", "map", "field", "balloon"),
           "both sides of the axis (axisymmetric: −θ … θ, −r … r)"),
    # axes
    Option("xscale", "choice", "auto", ("auto", "linear", "log"), ("curve", "map", "bode"),
           "auto: a log frequency axis when it spans more than a factor 8"),
    Option("yscale", "choice", "auto", ("auto", "linear", "log"), ("curve", "bode"),
           "auto: log for moduli spanning more than 1.5 decades, linear for levels, phases and signals"),
    Option("xmin", "number", None, (), ("curve", "map", "bode", "nyquist", "field")),
    Option("xmax", "number", None, (), ("curve", "map", "bode", "nyquist", "field")),
    Option("ymin", "number", None, (), ("curve", "map", "bode", "nyquist", "field"), "polar: the inner radius"),
    Option("ymax", "number", None, (), ("curve", "map", "bode", "nyquist", "field"), "polar: the outer radius"),
    Option("grid", "choice", "major", ("major", "both", "none"), ("curve", "polar", "bode", "nyquist")),
    Option("title", "text", "", (), ALL, "empty: automatic"),
    Option("legend", "choice", "auto", ("auto", "best", "upper right", "upper left", "lower right", "lower left",
                                        "outside", "none"), LINES),
    # lines
    Option("line", "choice", "solid", ("solid", "dashed", "dotted", "dashdot", "none"), LINES,
           "none: the points only (give a marker)"),
    Option("marker", "choice", "none", ("none", "o", "s", "^", "v", "D", "x", "+", "."), LINES),
    Option("linewidth", "number", None, (), LINES, "points (empty: the style's 1.2)"),
    Option("color", "text", "", (), LINES, "one colour for every curve: a name or #rrggbb (empty: the palette)"),
    Option("palette", "choice", "loudpy", PALETTES, LINES, "the colours of successive curves (loudpy: Okabe–Ito, "
                                                           "readable by colour-blind readers)"),
    # colours
    Option("cmap", "choice", "auto", CMAPS, COLOURED + ("view3d",),
           "auto: by quantity (SPL magma, signed values RdBu_r, displacement viridis, phase twilight); jet / "
           "rainbow: COMSOL's look"),
    Option("reverse", "bool", False, (), COLOURED + ("view3d",), "the colormap reversed"),
    Option("vmin", "number", None, (), COLOURED + ("view3d",), "bottom of the colour range (empty: automatic)"),
    Option("vmax", "number", None, (), COLOURED + ("view3d",), "top of the colour range (empty: automatic)"),
    Option("symmetric", "bool", False, (), COLOURED + ("view3d",), "a colour range symmetric about 0 (signed values)"),
    Option("range_db", "number", None, (), ("map", "polar", "field", "balloon", "view3d"),
           "levels in dB: the range shown below the top (polar: the inner radius; balloon: the radius at 0)"),
    Option("scale", "choice", "linear", ("linear", "log"), ("map", "field"), "colour scale (log: positive values)"),
    Option("levels", "number", 0, (), ("map", "field"), "discrete colour bands (0: smooth)"),
    Option("contours", "number", 0, (), ("map", "field"), "isolines drawn on top (0: none)"),
    Option("colorbar", "bool", True, (), COLOURED + ("view3d",)),
    # fields
    Option("deform", "bool", False, (), ("field", "view3d"), "displacement fields: on the deformed shape"),
    Option("deform_scale", "number", None, (), ("field", "view3d"),
           "peak deformation as a fraction of the size (empty: 0.05)"),
    Option("edges", "bool", False, (), ("field",), "the element edges (ParaView's surface with edges)"),
    Option("outline", "bool", True, (), ("field",), "the outlines of the regions"),
    # polar
    Option("zero", "choice", "N", ("N", "E", "S", "W"), ("polar",), "where 0° is drawn"),
    Option("clockwise", "bool", True, (), ("polar",)),
    # table
    Option("digits", "number", 5, (), ("table",), "significant digits"),
)

_BY_NAME = {o.name: o for o in STYLE_OPTIONS}


def options(view: str) -> list[Option]:
    """The style options of a view, in order."""
    return [o for o in STYLE_OPTIONS if o.applies(view)]


def option(name: str) -> Option:
    """One option by name."""
    if name not in _BY_NAME:
        raise ValueError(f"no style option {name!r} (options: {sorted(_BY_NAME)})")
    return _BY_NAME[name]


def check(style: dict | None, view: str = "") -> dict:
    """The style checked: every key a known option of the view (other views' options kept, they are harmless when
    the view changes back), values of the option's kind and choices; empty values dropped."""
    out: dict = {}
    for k, v in (style or {}).items():
        o = option(k)
        if v is None or v == "":
            continue
        if o.kind == "choice":
            if v not in o.choices:
                raise ValueError(f"style {k}: one of {list(o.choices)}, got {v!r}")
        elif o.kind == "number":
            try:
                v = float(v)
            except (TypeError, ValueError):
                raise ValueError(f"style {k}: a number, got {v!r}") from None
        elif o.kind == "bool":
            if not isinstance(v, bool):
                raise ValueError(f"style {k}: true or false, got {v!r}")
        else:
            v = str(v)
        out[k] = v
    return out


def get(style: dict, name: str):
    """The value of an option in a style (its default when not set)."""
    v = style.get(name)
    return option(name).default if v is None else v


__all__ = ["Option", "STYLE_OPTIONS", "options", "option", "check", "get", "CMAPS", "PALETTES"]
