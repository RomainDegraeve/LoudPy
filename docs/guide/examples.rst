Examples
========

``LoudPy_exemples/`` holds ready-to-run scripts. Run them from the repository root. Each example
folder separates the computation from the figures:

* ``Running_*.py`` — builds the model, solves, writes ``Results/Files/``
* ``Loading_*.py``, ``Plot_*.py`` — reads the results (``lp.load``), writes ``Results/Figures/``
  (figures) and ``Results/Files/`` (the key curves as CSV, fields as VTU for ParaView)

The plotting scripts are short: they only use the result objects (``res.spl(at=…)``,
``res.field(…)``, ``.plot()``, ``.export()``) and ``lp.plot`` — read them as worked examples of
:doc:`results`, :doc:`postprocessing` and :doc:`plotting`.

2D (axisymmetric)
-----------------

================================ ===================================================================
``Eigen_study/Meca``             structural modes of the HPNEW driver in vacuo (20 Hz – 40 kHz)
``Eigen_study/Fourier``          azimuthal-order modes m = 0 … 4 (rocking, ovalisation); 3D views;
                                 validation on a clamped plate (Kirchhoff)
``Eigen_study/FSI_DtN``          coupled structure + air modes with the exact radiation condition
``Frequency_study/Meca``         structural harmonic sweep (force on the coil)
``Frequency_study/FSI``          coupled sweep 20 Hz – 20 kHz, air and PML remeshed per wavelength
``Frequency_study/FSI_elec``     the same, driven by a voltage (Re, Bl, semi-inductance): electrical
                                 impedance, coil current
``Time_study/Multitone``         nonlinear transient response to a multitone signal, spectra
``Sweep_study``                  parametric sweeps: a geometry sweep (plate radius × thickness from a
                                 ``.geo`` with ``DefineConstant``, against Kirchhoff) and a material sweep
                                 (surround × spider stiffness of the driver: excursion, fs), netCDF exports
================================ ===================================================================

3D (drivers in enclosures)
--------------------------

========================================== ===============================================================
``Coupling_3D/Driver_in_cavity``            HPNEW in a non-axisymmetric box: cavity modes (A), on a rigid
                                            ground with exterior radiation (B), vented box as one air
                                            domain (C); the sealed box
                                            with elastic MDF walls (``Running_walls.py``, shells);
                                            validations of the cavity FE and of the half-space DtN
``Coupling_3D/Labyrinth``                   folded transmission line ("labyrinth") on a rigid ground (C)
``Coupling_3D/Column_labyrinth``            three identical drivers in a column with a folded line: the
                                            multi-driver scene (C)
========================================== ===============================================================

The 3D examples need the driver modes first:

.. code-block:: bash

   python3 LoudPy_exemples/Eigen_study/Fourier/Running_script.py

Their geometry lives in one module per example (``enclosure.py``, ``labyrinth.py``,
``column.py``): the dimensions, the cabinet builders and ``driver()``. The meshes are built on the
first run and reused afterwards.

``LoudPy_exemples/README.md`` lists the run times; ``LoudPy_exemples/legacy/`` keeps former scripts (solver
benchmarks, the low-level hybrid model of the vented box).
