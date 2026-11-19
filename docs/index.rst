loudpy
======

**loudpy** is a Python library for the finite element simulation of loudspeakers: axisymmetric
structures (cone, surround, spider, coil…), the air around them, their electro-mechanical drive,
and 3D enclosures with one or several drivers.

.. code-block:: python

   import numpy as np
   import loudpy as lp

   model = lp.Model("HPNEW-Sketch.geo")
   model.add(lp.Solid("membranne", "Paper"), lp.Solid("surround", "Rubber"),
             lp.Air("subacou"), lp.PML("PML"),
             lp.Clamped("interface_constrained"), lp.Coil("interface_forced"),
             lp.FluidStructure("interface_acou_meca_front"), lp.FluidStructure("interface_acou_meca_rear"))
   model.mesh("mesh.msh")

   res = lp.FrequencyResponse(model, drive=lp.Force(1.0)).solve(np.geomspace(20, 20000, 200))
   lp.plot.spl(res, at=(0.0, 0.5))

One vocabulary everywhere: a **model** (or a 3D **scene**) goes into a **study**
(``Modes``, ``FrequencyResponse``, ``Transient``); the study returns **results**, which are
saved, loaded back with :func:`lp.load <loudpy.load>` and plotted with :mod:`lp.plot <loudpy.plot>`.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   getting_started
   guide/models
   guide/studies
   guide/results
   guide/plotting
   guide/scenes
   guide/examples

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/index
   guide/file_format
   guide/architecture
   guide/conventions

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
