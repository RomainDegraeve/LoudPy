Conventions
===========

Units and signs
---------------

SI units throughout (m, kg, s, Pa, N, V). Harmonic quantities are complex amplitudes with the
time dependence :math:`e^{j\omega t}`. Sound pressure levels are RMS:
:math:`L_p = 20 \log_{10}(|p| / (\sqrt{2}\, p_\mathrm{ref}))`, :math:`p_\mathrm{ref} = 20\,\mu\mathrm{Pa}`.

Axisymmetric frame: ``r ≥ 0`` (radius), ``z`` along the axis (front +z). 3D scenes: the driver
axis is +z unless placed otherwise, ``y`` is vertical (the ground ``y = 0``).

Every physical input is checked where it enters the model (:mod:`loudpy.checks`): a value outside its
valid range raises :class:`~loudpy.errors.ModelError` (a negative density, ν ≥ 0.5 …); a valid value
outside its plausible SI range is kept with a :class:`~loudpy.errors.PlausibilityWarning` (a mesh size
of 2 for 2 mm, a Young's modulus in MPa …).

Errors and warnings
-------------------

Every error is a :class:`~loudpy.errors.LoudpyError`, and also the built-in exception it replaces
(``except ValueError`` keeps working):

=================================== ==========================================================
``ModelError`` (``MaterialError``)  invalid model description, settings or materials
``SelectionError``                  a name or selection matches nothing (also a ``KeyError``)
``MeshError``                       geometry or mesh that cannot be built or used
``StateError``                      call order: mesh before solving, solve before saving
``SolverError``                     a solver failed (``ConvergenceError`` when asked to be strict)
``ResultsError``                    the results do not contain what was asked for
``ResultsFormatError``              a result file that cannot be read
=================================== ==========================================================

Warnings: ``ConvergenceWarning`` (Newton–Raphson, GMRES or ARPACK did not reach their tolerance; the
result is kept) and ``PlausibilityWarning``. Turn them into errors with the standard ``warnings``
filters, e.g. ``warnings.simplefilter("error", lp.errors.ConvergenceWarning)``.

Messages
--------

loudpy is silent by default (it does not configure Python's logging). ``lp.set_verbosity("info")``
prints the progress of meshing, sweeps and mode searches; ``"debug"`` adds solver details (shifts,
filters, mode tables). The logger is ``"loudpy"``: configure it like any other.

Troubleshooting: OpenMP
-----------------------

loudpy does not set ``KMP_DUPLICATE_LIB_OK``. If a mixed environment (e.g. conda packages linked to
Intel MKL together with pip wheels) aborts with ``OMP: Error #15``, install the scientific stack from
one source (all pip, or all conda-forge with ``nomkl``) or select Numba's own thread pool with
``NUMBA_THREADING_LAYER=workqueue``; setting ``KMP_DUPLICATE_LIB_OK`` only hides the clash and can give
wrong results.

Physical-group names
--------------------

A group name of the ``.geo`` file is a sequence of tokens separated by ``_``:

.. code-block:: text

   [physics prefix] base tokens [modifiers]       submeca_glue_1, interface_constrained_rz_2,
                                                  subacou_rear, PML_r+_1_front

* physics prefix: ``submeca``, ``subacou`` (any first token starting with ``sub``)
* modifiers: an index (digits), a direction (``r``, ``z``, ``rz``, ``r+``, ``z-`` …), a side
  (``front``, ``rear``), ``Corner`` (PML corners built by the mesher)

In a ``.geo`` without surfaces, every closed curve group is a curve loop (a domain), whatever its name;
the other groups are curve groups (boundaries, PML base curves). The ``sub…`` prefix is only a
convention of the examples (it is skipped when matching labels: ``"membranne"`` selects
``submeca_membranne``).

A label (the name of a region or boundary, or an ``lp.Group`` label) **selects** a group when its
tokens appear whole, at the start of the group or right after the prefix, and every following token
is a modifier (other selectors, by position or topology: :doc:`models`):

=========================== =======================================================
``"membranne"``             ``submeca_membranne``
``"glue"``                  ``submeca_glue_1``, ``submeca_glue_2``
``"coil"``                  ``submeca_coil`` (not ``submeca_coil_former``)
``"interface_constrained"`` ``interface_constrained_r_1``, ``interface_constrained_rz_1``, …
``"PML"``                   every PML layer and corner
``"glue_*"``                glob on the name, with or without the prefix
=========================== =======================================================

**Names are labels.** The modifiers only matter for matching: nothing physical is read from a
name. Clamped components and the coil direction are parameters (``lp.Clamped(…, dofs="rz")`` by
default, ``lp.Coil(…, direction="z")``); the fluid–structure coupling uses the outward normal of the
adjacent air on every edge, computed from the mesh, so the direction in which the curves of the
geometry are drawn does not matter, and every air region has the same formulation (no front / rear
sign).

Saved meshes are reused while their signature (``<mesh>.msh.json``, version 2) matches: the
geometry files, the meshed domain groups with their kind, size and PML thickness, the kept boundary
groups, the mesh order and the grading (the deterministic flag is recorded but does not decide reuse:
a saved file is reproducible). The signature records groups, not
the names that selected them; a saved mesh with more boundary groups than needed is reused. Meshes
saved by earlier versions (signature version 1) are rebuilt once. A 3D scene's meshes (its cavity, the inside air
of its walls, each air band) are reused the same way; ``scene.mesh_files(folder)`` lists them with their state
(built, out of date, missing) without meshing.

File layout of an example
-------------------------

.. code-block:: text

   NN_example/
       README.md                 what it shows, how to run it, the key results
       run.py                    computation → Results/Files/
       plot.py                   figures     → Results/Figures/ (and CSV / netCDF in Results/Files/)
       plot_3d.py                3D views    → Results/Figures/
       <geometry>.py             dimensions and builders of a 3D example (or ../shared/enclosure.py)
       Results/Files/            meshes (+ .json signatures), result files (.h5)
       Results/Figures/          PDF figures
