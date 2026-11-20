Getting started
===============

Installation
------------

.. code-block:: bash

   pip install "loudpy[all]"          # release, with plotting and 3D views
   pip install -e ".[all]"            # from a clone of the repository (development)

loudpy needs Python ≥ 3.13, NumPy, SciPy, Numba, h5py and gmsh (meshing). The extras add plotting
(``loudpy[plot]``: matplotlib, pandas) and 3D views (``loudpy[3d]``: PyVista, imageio);
``loudpy[all]`` installs both.

.. tip::

   On macOS with the repository in an iCloud-synced folder, set ``NUMBA_CACHE_DIR`` to a local
   folder (``export NUMBA_CACHE_DIR=/tmp/numba_cache``): Numba's cache next to the sources can
   stall on synced folders.

A first model: the modes of a driver
------------------------------------

The geometry is a gmsh ``.geo`` file of the meridian half-plane (``r ≥ 0``, ``z`` along the
axis) whose physical groups follow the :doc:`naming convention <guide/conventions>`
(``submeca_membranne``, ``interface_constrained_rz_1``, ``subacou``…). Regions and boundaries
pick groups by name:

.. code-block:: python

   import loudpy as lp

   lp.set_verbosity("info")                        # progress messages (loudpy is silent by default)

   model = lp.Model("LoudPy_exemples/Geometries/Loudspeaker/HPNEW-Sketch.geo")
   model.add(
       lp.Solid("membranne", "Paper", mesh_size=0.0005),
       lp.Solid("coil", "Copper"),
       lp.Solid("surround", "Rubber"),
       lp.Solid("spider", "PhenolicCloth"),
       lp.Solid("former", "Kapton"),
       lp.Solid("glue", "SolidGlue"),
       lp.Solid("dustcap", "Polypropylene"),
       lp.Clamped("interface_constrained_rz"), lp.Clamped("interface_constrained_r", dofs="r"),
       lp.Coil("interface_forced"),
   )
   model.mesh("Results/Files/mesh.msh")            # reused while the inputs are unchanged

   study = lp.Modes(model)
   modes = study.solve(f_max=20000)
   study.save("Results/Files/modes.h5")

   modes = lp.load("Results/Files/modes.h5")      # later, in another script
   print(modes.table().head())                    # f, ζ of the first modes (pandas)
   modes.mode(0).plot(deformed=True)              # the first mode shape
   lp.plot.modes(modes, n=12)                     # a grid of the first twelve

The same pattern holds for every study: build the model, ``solve``, ``save``; then ``lp.load``
and plot in a separate script. The examples follow it with a ``Running_*.py`` (computation, writes
``Results/Files``) and a ``Loading_*.py`` / ``Plot_*.py`` (figures in ``Results/Figures``).

Where to go next
----------------

* :doc:`guide/models` — regions, boundaries, materials, damping, drives, meshing
* :doc:`guide/studies` — modes, frequency responses (with remeshing per wavelength), transients
* :doc:`guide/results` — fields, evaluations at points / arcs / lines, SPL, saving and loading
* :doc:`guide/postprocessing` — NumPy, pandas, CSV / MATLAB / ParaView exports
* :doc:`guide/plotting` — figures from the results, the loudpy style, 3D views
* :doc:`guide/scenes` — 3D enclosures with one or several drivers
