Plotting
========

Figures come from the results themselves: every :class:`Data <loudpy.Data>` and
:class:`Field <loudpy.Field>` has a ``plot()`` method, and :mod:`lp.plot <loudpy.plot>` composes
them. The structure follows COMSOL's results tree — *where* (a selection: point, arc, line, the
mesh), *what* (a quantity with its unit), *how* (a plot kind) — with the brevity of xarray / pandas.

.. code-block:: python

   res.spl(at=(0, 0.3)).plot()                          # curve: SPL over frequency (log axis)
   res.spl(at=lp.Arc(0.3, [0, 30, 60])).plot()          # one curve per angle, legend "θ = 30°"
   res.spl(at=lp.Arc(0.3, range(0, 91, 2))).plot()      # frequency × angle colour map
   d.normalize(angle=0).plot(kind="map", contours=[-6, -3])   # re on axis, iso-lines
   d.sel(f=[500, 2000, 8000]).plot(kind="polar")        # polar diagram
   res.field("p", f=1000).plot()                        # SPL on the r–z section
   res.field("u", f=1000).plot(deformed=True)           # |u| on the deformed structure
   res.field("p", f=1000).plot(mirror=True)             # the full section −r ... r
   modes.mode(3).plot("z")                              # one component
   lp.plot.modes(modes.order(0), n=12)                  # grid of mode shapes
   lp.plot.mode_chart(modes, y="order")                 # f against the azimuthal order (colour ζ)
   lp.plot.coupled_modes(modes.set("fsi"), [0, 3, 7])   # pressure in the air + deformed structure
   lp.plot.directivity(res, f=[1000, 4000])             # polar SPL re on axis; 2D: around the baffle, ±θ
   lp.plot.boundary(field, res.boundary("wet_front"))   # deformed boundary profile
   lp.plot.animate(trans, "u.mp4")                      # animation (transient, or a harmonic field)

Every function draws on ``ax=`` when given (a new figure otherwise) and returns the axes; labels
come from the quantity (``"Sound pressure level $L_p$ [dB re 20 µPa]"``, ``"Frequency $f$ [Hz]"``),
so several Data on one axes stay consistent:

.. code-block:: python

   fig, ax = lp.plot.figure("tall", 2, 1, sharex=True)  # standard sizes, constrained layout
   res.spl(at=lp.Arc(0.3, [0, 30, 60])).plot(ax=ax[0])
   u = res.displacement(at=(0, 0), component="z")
   lp.plot.line([u.rename(label="u"), u.velocity().rename(label="v")], ax=ax[1])
   lp.plot.save(fig, "Results/Figures/response.pdf")

Plot conventions
----------------

* frequency on a log axis with 20, 50, 100 ... 1k ticks; complex amplitudes shown as moduli
* moduli of physical amplitudes (displacement, pressure, impedance...) over frequency on a log axis
  when they span more than 1.5 decades; levels (dB), phases, ratios and signals on a linear one
* relative levels (``normalize``) on a −40 ... +6 dB range
* fields: smooth shading on the P2 triangles split through their mid-side nodes, region outlines,
  equal aspect, colour bar with the quantity; SPL maps on the 50 dB below their top (``range_db``)
* signed fields (``"real"``, ``"imag"``, ``"instant"``) on a symmetric diverging map
* deformed shapes: the peak displacement 5 % of the mesh size (``scale``), the rest shape dashed

3D views
--------

3D figures are pyvista scenes rendered into matplotlib axes, so they share the fonts, colormaps
and colour bars of the 2D figures.

.. code-block:: python

   v = lp.plot.View3D(size=(900, 700))
   v.add(modes.mode(5).revolve(), scalars="|u|", clim=(0, 1))     # a mode revolved in 3D
   v.view("xz", azimuth=35, elevation=25).show()
   res.field("p", f=500).plot(cut=((-1, 0, 0), (0.1, 0, 0)))      # a 3D field, cut open
   scene_res.plot(470)                                             # a 3D scene result: six default views

A scene result (lp.Scene with an FE air domain) plots itself: ``scene_res.plot(f)`` shows the SPL on the
ground, the walls and a cut plane on one colour scale, iso-SPL lines, the dashed outline of the removed part
and the deformed drivers at one instant and one scale. By default it uses six cameras around the cabinet
(:func:`lp.plot.default_cameras <loudpy.plot.default_cameras>`: front left, front right, side, from above, rear,
close-up on the driver) and the vertical plane through the first driver's axis as the cut; choose with
``cameras=["from above", …]`` (default cameras by name) or ``{name: lp.plot.Camera(…)}``, and ``cut=None``
(the whole scene) or ``cut=(normal, origin)``. ``scene_res.view(f, camera)`` is one of them as a View3D; it
also takes ``cut=("y", 0.2)`` (the plane across x, y or z at a value: the half x < value, y > value or z > value
removed, "-y" the other half, the camera on the removed side), ``surfaces=`` (which of
``scene_res.air_surfaces`` carry the SPL: the ground, the walls, the openings by default) and ``points=``,
``arcs=`` (observation points and ``lp.Arc`` arcs drawn as spheres and tubes, the camera backing off to show
them). With elastic walls, ``scene_res.wall_view(f, panels=[…])`` draws the walls deformed and coloured by their
normal displacement (``loudpy.plot.scene3d.wall_mode_view(modes, k)``: a wall mode), and
``scene_res.balloon_view(f)`` the directivity balloon on the half sphere above the ground. The
building blocks (``view3d.tet_grid``, ``tri_surface``, ``clip``, ``isolines``, ``edges``, ``dashed``) are in
:mod:`loudpy.plot.view3d`.

Saving and turning 3D views
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Any 3D view is a :class:`lp.plot.View3D <loudpy.plot.View3D>`. ``.plot()`` draws a result into a figure;
``.view()`` gives the same 3D view as an object to save in the format the reader needs, or to turn in a window:
``field.view(cut=…)`` (a 3D field, or a 2D displacement revolved) and ``scene_res.view(f, camera)``; or build
one by hand.

.. code-block:: python

   v = res.field("u", f=1000).view(sweep=1.5 * np.pi)       # 2D result: the structure revolved, |u|
   v = scene_res.view(500)                                   # a scene result, cut open (default camera)

   v.save("view.png")      # .png .jpg .tif .pdf .svg: the figure (image at 2 × size, colour bars, title), 300 dpi
   v.save("view.html")     # interactive viewer in one self-contained file (any browser, no server, no package)
   v.save("view.glb")      # glTF 3D model with its colours (.gltf / .glb): Blender, PowerPoint, Windows 3D viewer,
                           # web viewers
   v.save("view.vtm")      # the datasets with their values (ParaView: File > Open)
   v.interactive()         # a window to turn, zoom and pan (in Jupyter: inline with pyvista's notebook backend)
   v.plotter()             # the pyvista Plotter itself, to go further

The HTML file shows every surface of the view with its own colours (SPL and the drivers' ``|u|`` side by side) and the
first colour bar; lines (iso-lines, edges) are left out. Examples: ``06_response_voltage/plot.py`` (revolved
structure, HTML), ``02_azimuthal_modes/plot_3d.py`` and ``11_box_radiation/plot_3d_views.py`` (HTML, glTF and
``--interactive``).

A plot as plain data
--------------------

:class:`lp.plot.PlotSpec <loudpy.plot.PlotSpec>` describes a plot as plain data (JSON): a quantity, where it is
evaluated, the dimensions plotted, the values of the others and how it is shown. Resolved on results it slices the
named dimensions of the labelled data (as xarray does) and turns every dimension that is not plotted into a
*control* — a slider over the frequencies, the times, the modes, the runs of a sweep or the phase of a harmonic field;
a choice among names — that a program can step, play or show as a widget (the interface generates its sliders from
them):

.. code-block:: python

   spec = lp.plot.PlotSpec(quantity="spl", where={"kind": "arc", "radius": 0.5, "to": 180, "center": [0, 0.03]},
                           axes=["angle"], view="polar", at={"f": 1000.0})
   out = spec.resolve(res)             # out.data: Data over angle; out.controls: [Control f (index at 1000 Hz)]
   out.plot()                          # the same in matplotlib, loudpy's style
   spec.step(res, "f", +1)             # the next frequency (a player wraps round)
   spec.animate(res, "f", "polar.gif") # every frequency in turn (.gif, .mp4)
   lp.plot.PlotSpec(quantity="field", field="u", style={"show": "real"}).animate(res, "phase", "u.mp4")
   spec.to_dict()                      # kept with a simulation: lp.Document(...).plots

``lp.plot.spec.quantities(res)`` lists what results offer (SPL, pressure, displacement, velocity, acceleration,
impedance, a field at one entry); ``where`` is a point, a line (dimension ``s``), an arc (dimension ``angle``)
or a named boundary or region (``{"kind": "on", "name": "coil1", "reduce": "mean"}``: :doc:`results`).

How a plot is shown
-------------------

``spec.style`` holds how a plot is shown: the options of :mod:`loudpy.plot.options`, declared once (kind, choices,
default, the views they apply to, a hint) so a program builds its form from them; a spec with an unknown or wrong option
is refused. They cover what COMSOL's plot settings, Ansys' result display and ParaView's display properties offer:

.. list-table::
   :widths: 18 82

   * - data
     - ``show`` (modulus, real or imaginary part, phase, level dB, SPL; fields: instant, a component), ``normalize``
       (to the maximum, or to the first value — the axis of a directivity), ``unwrap``, ``mirror`` (both sides)
   * - axes
     - ``xscale`` / ``yscale`` (auto, linear, log), ``xmin`` … ``ymax``, ``grid`` (major, both, none), ``title``,
       ``legend`` (positions, outside, none)
   * - lines
     - ``line`` (solid, dashed, dotted, dash-dot, none: points), ``marker``, ``linewidth``, ``color``, ``palette``
       (loudpy's Okabe–Ito, tab10, Dark2, Set1, gray, viridis)
   * - colours
     - ``cmap`` (by quantity, or viridis, magma, inferno, plasma, cividis, turbo, jet, rainbow, coolwarm, RdBu_r …),
       ``reverse``, ``vmin`` / ``vmax``, ``symmetric``, ``range_db``, ``scale`` (log colours), ``levels`` (bands),
       ``contours`` (isolines), ``colorbar``
   * - fields
     - ``deform`` (+ ``deform_scale``), ``edges`` (the mesh), ``outline``
   * - polar, table
     - ``zero``, ``clockwise``; ``digits``

The views: ``curve``, ``map``, ``polar``, ``table``, ``field``, ``view3d``, and ``bode`` (modulus and phase over f),
``nyquist`` (imaginary against real part: an impedance), ``balloon`` (a directivity revolved around the axis, 3D):

.. code-block:: python

   arc = {"kind": "arc", "radius": 0.3, "to": 90, "center": [0, z_b]}
   lp.plot.PlotSpec(quantity="spl", where=arc, view="map",
                    style={"normalize": "first", "levels": 12, "contours": 3, "cmap": "jet", "mirror": True})
   lp.plot.PlotSpec(quantity="spl", where=arc, view="balloon", at={"f": 3000}, style={"range_db": 30})
   lp.plot.PlotSpec(quantity="impedance", view="bode", style={"yscale": "linear", "grid": "both"})
   lp.plot.PlotSpec(quantity="displacement", component="z", view="table",
                    where={"kind": "on", "name": "coil1", "reduce": "mean"})       # out.table(): header, rows
   lp.plot.PlotSpec(quantity="displacement", component="z", where={"kind": "on", "name": "fluidstructure1"},
                    axes=["s"], at={"mode": 3}, style={"show": "real", "line": "dashed", "marker": "."})

``Resolved.plot()`` draws with the style (:mod:`loudpy.plot.draw`), ``Resolved.shown()`` gives the data as the style
shows it and ``Resolved.table()`` the rows of a table; ``lp.plot.bode``, ``nyquist``, ``balloon`` and ``table`` draw
Data directly.

Selections
----------

``lp.plot.selection(model, selector)`` shows what a selector selects on the mesh, as the solver sees it: on a
meshed 2D model the selected domain groups are filled and the selected edges drawn thick (``kind="domain"`` or
``"boundary"`` for one of them); on a meshed ``lp.Scene`` the selector names surface groups of its meshes
(``"walls"``, ``"interface_driver"``, ``"dtn"``, ``"ground"``…; ``mesh="cavity"`` or the index of a band mesh); a
3D ``lp.Mesh`` works the same way. The 3D views use matplotlib's 3D axes (no OpenGL needed).

.. code-block:: python

    lp.plot.selection(model, lp.Interface("coil", "former"))
    lp.plot.selection(scene, ["walls", "interface_driver"])

Interactive viewer
------------------

``lp.plot.html(field, "x.html")`` (or ``field.export("x.html")``) writes one self-contained HTML file — no
server, no internet connection, no extra package: the surface of a 3D field (outer faces of the tetrahedra, shell
and wall groups) or the r–z section of a 2D one (``mirror=True`` for −r … r), coloured with the quantity's
colormap. Drag to rotate, wheel to zoom, shift-drag to pan, double-click to reset. A list of fields on one mesh
gives frames with a slider and one colour range:

.. code-block:: python

    lp.plot.html([res.field("p", f=f) for f in (200, 500, 1000)], "Results/Figures/p.html")

The loudpy style
----------------

:mod:`loudpy.plot.style` is the single place where figures are styled. Every ``lp.plot`` function
draws in that style without touching your global matplotlib settings; scripts that also draw with
matplotlib directly (titles, extra lines, their own figures) call ``lp.plot.style.use()`` once, as all
the examples do.

* serif text and mathematics (STIX), 8.5–10 pt, thin axes, ticks inside on four sides, minor ticks
* the Okabe–Ito colour cycle (colour-blind safe)
* one colormap per quantity — ``style.CMAPS``: ``"spl"`` magma, ``"displacement"`` viridis,
  ``"phase"`` twilight (cyclic), ``"level"`` viridis; ``style.DIVERGING`` (RdBu_r) for signed fields
* labels of every quantity — ``style.LABELS["spl"]``, ``["u_z"]``, ``["f"]``, ... (from
  :mod:`loudpy.results.quantities`)
* standard sizes — ``lp.plot.figure("single" | "wide" | "square" | "tall" | "field")``
* ``style.frequency_axis(ax)``: log axis with readable ticks; ``style.tidy(fig)`` brings a figure
  drawn with plain matplotlib to the same look
* ``lp.plot.save(fig, path)`` creates the folder and writes next to the target before renaming it
  (safe on synced folders); ``lp.plot.pdf_pages(path)`` for multi-page PDFs (each page tidied and
  closed); fonts embedded as TrueType (editable text)

Where figures go
----------------

Every example writes its result files to ``Results/Files/`` and its figures to
``Results/Figures/`` next to the scripts; 3D sweeps write one PDF per frequency band in a
sub-folder (``Results/Figures/exterior_3d/`` ...).
