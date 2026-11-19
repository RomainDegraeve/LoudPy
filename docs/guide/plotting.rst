Plotting
========

:mod:`lp.plot <loudpy.plot>` draws results, not raw arrays, in one consistent scientific style.

.. code-block:: python

   lp.plot.field(res, "p", f=1000, component="spl")    # a field on its mesh
   lp.plot.mode(modes, 3)                              # mode shape (|u|)
   lp.plot.modes_grid(modes.order(0), n=12)            # grid of mode shapes
   lp.plot.spl(res, at=(0.0, 0.5))                     # SPL over frequency
   lp.plot.response(res, at=(0.0, 0.02), name="u")     # |u_z| over frequency
   lp.plot.history(trans, at=(0.0, 0.02), name="u")    # u_z over time
   lp.plot.directivity(scene_res, f=[100, 1000], center=(0.1, 0.2, 0.03))   # polar curves (3D)

Every function draws on ``ax=`` when given (a new figure otherwise) and returns the axes, so
plots compose with ordinary matplotlib code.

The loudpy style
----------------

:mod:`loudpy.plot.style` is the single place where figures are styled. Importing ``lp.plot``
applies it; scripts that use matplotlib directly call ``lp.plot.style.use()`` once (all the
examples do).

* serif text and mathematics (STIX), 8.5–10 pt, thin axes, ticks inside on four sides, minor ticks
* the Okabe–Ito colour cycle (colour-blind safe)
* one colormap per quantity — ``style.CMAPS``: ``"spl"`` magma, ``"pressure"`` (signed) RdBu_r,
  ``"displacement"`` viridis, ``"phase"`` twilight (cyclic), ``"level"`` viridis
* standard labels with units — ``style.LABELS["f"]``, ``["spl"]``, ``["uz"]``, …
* standard sizes — ``style.figure("single" | "wide" | "square" | "tall" | "field")``
* frequency axes — ``style.frequency_axis(ax)``: log scale with 20, 50, 100, 200, 500, 1k … ticks
* SPL is RMS, in dB re 20 µPa, everywhere
* tidying — ``style.tidy(fig)`` brings a figure drawn with plain matplotlib to the same look before
  saving (Hz ticks on log frequency axes, standard labels, text scaled on figures wider than a
  page); ``style.save`` calls it
* saving — ``style.save(fig, path)`` creates the folder and writes next to the target before
  renaming it (safe on synced folders); ``style.pdf_pages(path)`` for multi-page PDFs; fonts are
  embedded as TrueType (editable text in vector editors)

.. code-block:: python

   import loudpy as lp
   lp.plot.style.use()
   fig, ax = lp.plot.style.figure("wide")
   ax.plot(res.frequencies, res.spl(at=(0, 1.0)), label="on axis")
   lp.plot.style.frequency_axis(ax)
   ax.set_ylabel(lp.plot.style.LABELS["spl_n"])
   ax.legend()
   lp.plot.style.save(fig, "Results/Figures/spl.pdf")

``lp.plot.style.context()`` applies the style inside a ``with`` block only, and
``lp.plot.style.reset()`` restores the matplotlib defaults.

Where figures go
----------------

Every example writes its result files to ``Results/Files/`` and its figures to
``Results/Figures/`` next to the scripts; 3D sweeps write one PDF per frequency band in a
sub-folder (``Results/Figures/exterior_3d/`` …).
