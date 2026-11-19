Studies
=======

A study takes a meshed model (or a 3D :doc:`scene <scenes>`) and returns results. All studies
share the same three steps:

.. code-block:: python

   study = lp.<Study>(model, ...)          # what to compute
   res = study.solve(...)                  # Results (FrequencyResults, ModeResults, TimeResults)
   study.save("Results/Files/name.h5", note="anything")   # loudpy result format 2 (+ metadata)

Modes
-----

.. code-block:: python

   modes = lp.Modes(model).solve(f_max=20000)                       # structure (in vacuo)
   fourier = lp.Modes(model).solve(f_max=10000, orders=[0, 1, 2, 3])  # azimuthal orders m
   fsi = lp.Modes(model, physics="coupled").solve(f_max=5000, dtn_center=(0, 0), dtn_half=0.3)

``physics`` is ``"structure"`` (default), ``"acoustic"`` or ``"coupled"`` (structure + air, with
the exact radiation condition: complex modes). The structural band is covered by shift-invert
Lanczos at several shifts (``method="multishift"``) or one ARPACK call (``method="arpack"``);
the start vector is fixed, so the solves are reproducible.

Frequency response
------------------

.. code-block:: python

   study = lp.FrequencyResponse(model, drive=lp.Force(0.1))
   res = study.solve(np.geomspace(20, 20000, 300))

With air, the problem is coupled automatically. For wide bands the air and PML meshes can follow
the wavelength:

.. code-block:: python

   remesh = lp.PerWavelength(every=30, air_elements=6, air_max_size=0.005,
                             pml_thickness=0.25, pml_elements=4, folder="Results/Files")
   res = study.solve(freqs, remesh=remesh)      # one result block per mesh

Transient
---------

.. code-block:: python

   study = lp.Transient(model, drive=lp.Force(signal, fs=48000), rho_inf=0.98)
   res = study.solve()                          # TimeResults: u, v, a over time

Generalised-α time integration (``rho_inf``: high-frequency damping), Newton–Raphson iterations
for the nonlinear coil (``nonlinear=True``).

Studies on a 3D scene
---------------------

``lp.FrequencyResponse(scene, drive=…)`` and ``lp.Modes(scene)`` accept an
:class:`lp.Scene <loudpy.Scene>`; see :doc:`scenes`.
