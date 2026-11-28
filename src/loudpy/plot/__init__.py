"""
loudpy.plot — scientific figures of loudpy results, one style for all of them.

One way to plot: every result object has .plot() — a figure in the loudpy style (lp.Data: curves, maps, polar;
lp.Field: a map on the mesh; a 3D scene result: cut-away views) — and the 3D ones .view() — a View3D to save as
an image, an interactive HTML page or a 3D model, or to turn in a window. The functions here compose figures
(several curves on one axes, grids of modes) and draw the composite plots:

    res.spl(at=(0, 0.3)).plot()                          a curve (frequency on a log axis)
    res.spl(at=lp.Arc(0.3, range(0, 91, 2))).plot()      a frequency × angle map
    res.field("p", f=1000).plot()                        SPL on the r–z section
    res.field("u", f=1000).plot(deformed=True)           |u| on the deformed structure

    fig, axes = lp.plot.figure("wide", 1, 2)              standard sizes, constrained layout
    lp.plot.line([d1, d2], ax=axes[0])                   several Data on one axes
    lp.plot.map(d), lp.plot.polar(d), lp.plot.field(f)   explicit kinds
    lp.plot.directivity(res, f=[500, 2000], radius=0.3)  polar SPL (2D or 3D results)
    lp.plot.bode(z), lp.plot.nyquist(z)                  modulus + phase over f; Im against Re (an impedance)
    lp.plot.balloon(d.sel(f=1000))                       a directivity over the angle revolved around the axis (3D)
    lp.plot.table(d)                                     Data drawn as a table
    lp.plot.modes(modes.order(0))                        grid of mode shapes
    lp.plot.coupled_modes(modes.set("fsi"), [0, 3, 7])   coupled modes: air pressure + structure
    lp.plot.mode_chart(modes, y="zeta" | "order")        modes as points: f against ζ or the order m
    lp.plot.boundary(field, res.boundary(name))          deformed boundary profile
    lp.plot.selection(model, selector)                   what a selector selects (2D model, 3D scene surfaces)
    lp.plot.html(field, "x.html")                        interactive viewer (one self-contained HTML file)
    field.view(cut=…)                                    a 3D view (View3D) of a field (3D, or a 2D displacement revolved)
    scene_res.plot(f)                                    cut-away views of a 3D scene result (SPL + drivers), default
                                                         cameras, cut through the driver axis, dashed outline
    scene_res.view(f, camera)                            one of them as a View3D (cut=("y", 0.2), surfaces=, points=,
                                                         arcs=)
    scene_res.wall_view(f), scene_res.balloon_view(f)    the elastic walls deformed; the directivity balloon
    view.save("x.png" | ".pdf" | ".html" | ".glb" | ".vtm"), view.interactive()   save / turn a 3D view
    lp.plot.View3D(…)                                    your own 3D scene (pyvista) in matplotlib axes
    lp.plot.default_cameras(bounds, drivers)             the default cameras of a scene
    lp.plot.animate(trans, "u.mp4")                      animations
    lp.plot.PlotSpec(quantity=…, where=…, axes=…, at=…)  a plot as plain data: .resolve(res) → the data and a control
                                                         per dimension not plotted (slider / player), .to_dict();
                                                         style={…}: how it is shown (lp.plot.options: log / linear
                                                         axes, line style, colours, colormap, range, contours …)
    lp.plot.save(fig, "Results/Figures/x.pdf")           folders created, safe on synced folders

Every lp.plot function draws in the loudpy style (loudpy.plot.style: serif fonts, Okabe–Ito colours,
one colormap per quantity, labels "Name $symbol$ [unit]") without changing your global matplotlib
settings; lp.plot.style.use() makes it the global style (for figures you draw yourself too), and
lp.plot.style.reset() restores matplotlib's defaults.
"""
from __future__ import annotations

import numpy as np

try:
    import matplotlib  # noqa: F401
except ImportError as e:                                  # an optional dependency
    raise ImportError("lp.plot needs matplotlib: pip install 'loudpy[plot]'") from e

from loudpy.plot import style
from loudpy.plot.animate import animate
from loudpy.plot.curves import data, line, map, polar
from loudpy.plot.fields import boundary, coupled_modes, field, mode_chart, modes
from loudpy.plot.scene3d import Camera, default_cameras
from loudpy.plot.selection import selection
from loudpy.plot.html import html
from loudpy.plot.view3d import View3D
from loudpy.plot.spec import PlotSpec
from loudpy.plot import options
from loudpy.plot.draw import balloon, bode, nyquist, table

figure = style.figure
save = style.save
pdf_pages = style.pdf_pages
label_panels = style.label_panels


@style.styled
def directivity(res, f, radius: float | None = None, angles=None, center=None, axis="z", toward="x",
                normalize: bool = True, mirror: bool | None = None, ax=None, **kwargs):
    """
    Polar SPL at the frequencies f on an arc of `radius` around `center`, angles [°] from `axis`. 2D results
    (res.directivity): centre on the baffle and radius read from the mesh by default, mirrored to −θ … θ (both sides
    of the axis; mirror=False: 0 … θ only); 3D scenes: radius and centre needed, in the plane of `axis` and `toward`,
    default −180 … 180°. normalize: dB re the 0° direction. Returns the polar axes.
    """
    if type(res).__name__ == "SceneResults":
        d = res.directivity(radius, angles, center, mirror=bool(mirror), axis=axis, toward=toward)
    else:
        d = res.directivity(radius, angles, center, mirror=True if mirror is None else mirror)
    d = d.sel(f=list(np.atleast_1d(f)))
    if normalize:
        d = d.normalize(angle=0)
    return polar(d, ax=ax, **kwargs)


__all__ = ["selection", "html", "style", "figure", "save", "pdf_pages", "label_panels", "data", "line", "map",
           "polar", "field", "boundary", "modes", "coupled_modes", "mode_chart", "directivity", "View3D",
           "Camera", "default_cameras", "animate", "PlotSpec", "options", "bode", "nyquist", "balloon", "table"]
