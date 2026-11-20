"""
loudpy.plot — scientific figures of loudpy results, one style for all of them.

Most plots come from the evaluated results themselves (Data.plot, Field.plot); the functions
here draw them, compose figures and add a few composite plots:

    res.spl(at=(0, 0.3)).plot()                          a curve (frequency on a log axis)
    res.spl(at=lp.Arc(0.3, range(0, 91, 2))).plot()      a frequency × angle map
    res.field("p", f=1000).plot()                        SPL on the r–z section
    res.field("u", f=1000).plot(deformed=True)           |u| on the deformed structure

    fig, axes = lp.plot.figure("wide", 1, 2)              standard sizes, constrained layout
    lp.plot.line([d1, d2], ax=axes[0])                   several Data on one axes
    lp.plot.map(d), lp.plot.polar(d), lp.plot.field(f)   explicit kinds
    lp.plot.directivity(res, f=[500, 2000], radius=0.3)  polar SPL (2D or 3D results)
    lp.plot.modes(modes.order(0))                        grid of mode shapes
    lp.plot.coupled_modes(modes.set("fsi"), [0, 3, 7])   coupled modes: air pressure + structure
    lp.plot.mode_chart(modes, y="zeta" | "order")        modes as points: f against ζ or the order m
    lp.plot.boundary(field, res.boundary(name))          deformed boundary profile
    lp.plot.View3D(…)                                    3D scenes (pyvista) in matplotlib axes
    lp.plot.scene_view(res, f, cameras, cut=…)           cut-away views of a 3D scene result (SPL + drivers)
    lp.plot.animate(trans, "u.mp4")                      animations
    lp.plot.save(fig, "Results/Figures/x.pdf")           folders created, safe on synced folders

Importing loudpy.plot applies the loudpy style (loudpy.plot.style: serif fonts, Okabe–Ito colours,
one colormap per quantity, labels "Name $symbol$ [unit]"); lp.plot.style.reset() undoes it.
"""
from __future__ import annotations

import numpy as np

from loudpy.plot import style
from loudpy.plot.animate import animate
from loudpy.plot.curves import data, line, map, polar
from loudpy.plot.fields import boundary, coupled_modes, field, mode_chart, modes
from loudpy.plot.scene3d import Camera, scene_view
from loudpy.plot.view3d import View3D

style.use()

figure = style.figure
save = style.save
pdf_pages = style.pdf_pages
label_panels = style.label_panels


def directivity(res, f, radius: float, angles=None, center=None, axis="z", toward="x", normalize: bool = True,
                ax=None, **kwargs):
    """
    Polar SPL at the frequencies f on an arc of `radius` around `center`, angles [°] from `axis`
    (2D results: from the z axis toward +r, default 0…180°; 3D scenes: in the plane of `axis` and
    `toward`, default −180…180°); normalize: dB re the 0° direction. Returns the polar axes.
    """
    from loudpy.results.selection import Arc
    dim = 3 if hasattr(res, "far_field") else 2                    # scene results are 3D
    if angles is None:
        angles = np.arange(0, 181, 2) if dim == 2 else np.arange(-180, 181, 2)
    d = res.spl(at=Arc(radius, angles, center, axis, toward)).sel(f=list(np.atleast_1d(f)))
    if normalize:
        d = d.normalize(angle=0)
    return polar(d, ax=ax, **kwargs)


__all__ = ["style", "figure", "save", "pdf_pages", "label_panels", "data", "line", "map", "polar", "field",
           "boundary", "modes", "coupled_modes", "mode_chart", "directivity", "View3D", "Camera", "scene_view", "animate"]
