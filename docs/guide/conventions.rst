Conventions
===========

Units and signs
---------------

SI units throughout (m, kg, s, Pa, N, V). Harmonic quantities are complex amplitudes with the
time dependence :math:`e^{j\omega t}`. Sound pressure levels are RMS:
:math:`L_p = 20 \log_{10}(|p| / (\sqrt{2}\, p_\mathrm{ref}))`, :math:`p_\mathrm{ref} = 20\,\mu\mathrm{Pa}`.

Axisymmetric frame: ``r ≥ 0`` (radius), ``z`` along the axis (front +z). 3D scenes: the driver
axis is +z unless placed otherwise, ``y`` is vertical (the ground ``y = 0``).

Physical-group names
--------------------

A group name of the ``.geo`` file is a sequence of tokens separated by ``_``:

.. code-block:: text

   [physics prefix] base tokens [modifiers]       submeca_glue_1, interface_constrained_rz_2,
                                                  subacou_rear, PML_r+_1_front

* physics prefix: ``submeca``, ``subacou`` (any first token starting with ``sub``)
* modifiers: an index (digits), a direction (``r``, ``z``, ``rz``, ``r+``, ``z-`` …), a side
  (``front``, ``rear``), ``Corner`` (PML corners built by the mesher)

A name given to a region or boundary **selects** a group when its tokens appear whole, at the
start of the group or right after the prefix, and every following token is a modifier:

=========================== =======================================================
``"membranne"``             ``submeca_membranne``
``"glue"``                  ``submeca_glue_1``, ``submeca_glue_2``
``"coil"``                  ``submeca_coil`` (not ``submeca_coil_former``)
``"interface_constrained"`` ``interface_constrained_r_1``, ``interface_constrained_rz_1``, …
``"PML"``                   every PML layer and corner
``"glue_*"``                glob on the name, with or without the prefix
=========================== =======================================================

File layout of an example
-------------------------

.. code-block:: text

   Example/
       Running_script.py         computation → Results/Files/
       Loading_script.py         figures     → Results/Figures/
       Plot_3D_*.py              3D views    → Results/Figures/
       <geometry>.py             dimensions and builders shared by the scripts (3D)
       Results/Files/            meshes (+ .json signatures), result files (.h5)
       Results/Figures/          PDF figures
