loudpy
======

**loudpy** is a Python library for the finite element simulation of loudspeakers: axisymmetric
structures (cone, surround, spider, coil…), the air around them, their electro-mechanical drive,
and 3D enclosures with one or several drivers.

.. code-block:: python

   import numpy as np
   import loudpy as lp

   model = lp.Model("examples/Geometries/Loudspeaker/HPNEW-Sketch.geo")
   model.add(lp.Solid("cone", "Paper"), lp.Solid("coil", "Copper"), lp.Solid("surround", "Rubber"),
             lp.Solid("spider", "PhenolicCloth"), lp.Solid("former", "Kapton"), lp.Solid("glue", "SolidGlue"),
             lp.Solid("dustcap", "Polypropylene"), lp.Air("air"), lp.PML("PML", thickness=0.1, tuned_to=1000.0),
             lp.Clamped("rims", on=lp.Group("surround_rim", "spider_rim")), lp.Clamped("axis", dofs="r"), lp.Coil("coil_drive"),
             lp.FluidStructure("wet_front"), lp.FluidStructure("wet_rear"))
   model.mesh("mesh.msh")

   res = lp.FrequencyResponse(model, drive=lp.Force(1.0)).solve(np.geomspace(20, 20000, 200))
   spl = res.spl(at=lp.Arc(0.5, range(0, 91, 5)))   # SPL over frequency × angle, with its units
   spl.plot()                                       # scientific figure, labels from the quantity
   spl.export("spl.csv")                            # for Excel, MATLAB, Origin … (or .mat, .npz)
   res.field("p", f=1000).plot()                    # the pressure on the r–z section

One vocabulary everywhere: a **model** (or a 3D **scene**) goes into a **study**
(``Modes``, ``FrequencyResponse``, ``Transient``); the study returns **results**, which are
saved and loaded back with :func:`lp.load <loudpy.load>`. Evaluating results gives labelled data
(:class:`Data <loudpy.Data>`, :class:`Field <loudpy.Field>`) that plot themselves and convert to
NumPy, pandas and files.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   getting_started
   guide/models
   guide/studies
   guide/results
   guide/postprocessing
   guide/plotting
   guide/scenes
   guide/circuits
   guide/shells
   guide/examples

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/index
   guide/building_blocks
   guide/file_format
   guide/architecture
   guide/theory
   guide/elements
   guide/verification
   guide/verification_report
   guide/conventions

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
