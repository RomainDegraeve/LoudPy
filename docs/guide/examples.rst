Examples
========

``examples/`` holds one folder per example, numbered in reading order, each with the same layout:
``README.md`` (what it shows, how to run it, the key results), ``run.py`` (builds the model, solves, writes
``Results/Files/``), ``plot.py`` (reads the results with ``lp.load``, writes ``Results/Figures/`` and the key curves as
CSV or netCDF) and, where there are 3D views, ``plot_3d.py``. Paths are relative to each script, so they run from
anywhere; a ``run.py`` is computed once, ``plot.py`` is re-run on the stored results.

The plotting scripts are short: they only use the result objects (``res.spl(at=…)``, ``res.field(…)``, ``.plot()``,
``.export()``) and ``lp.plot`` — read them as worked examples of :doc:`results`, :doc:`postprocessing` and
:doc:`plotting`.

========================================= ======================================================================
``01_structure_modes``                    modes of the driver in vacuo
``02_azimuthal_modes``                    non-axisymmetric modes (orders m = 0 … 4), 3D views; the driver modes
                                          used by examples 10–15
``03_coupled_modes_radiation``            structure + air modes with the exact radiation condition
``04_response_structure``                 frequency response of the structure (force on the coil)
``05_response_coupled``                   structure + air response 20 Hz – 20 kHz, air and PML remeshed per
                                          wavelength, directivity
``06_response_voltage``                   the same driven by a voltage: electrical impedance, coil current
``07_transient``                          nonlinear transient response to a multitone force, spectra; output
                                          control (``lp.Output``)
``08_sweep_geometry``                     parametric sweep of the geometry (plate radius × thickness from a ``.geo``
                                          with ``DefineConstant``), against Kirchhoff; netCDF export
``09_sweep_materials``                    parametric sweep of materials (surround × spider stiffness): excursion, fs
``10_box_cavity``                         the driver in a non-axisymmetric 3D box: cavity modes, coupled modes
``11_box_radiation``                      the box on a rigid ground with exterior radiation (exact half-space DtN)
``12_box_vented``                         a vented box as one air domain (inside + port + outside)
``13_box_elastic_walls``                  the box with elastic MDF walls (shells)
``14_labyrinth``                          a folded transmission line on a rigid ground
``15_column_multi_driver``                three identical drivers in a column with a folded line
``validation``                            clamped plate (Kirchhoff), rectangular cavity, radiating hemisphere
========================================= ======================================================================

The box of examples 10–13 is defined once in ``shared/enclosure.py`` (dimensions, cabinet builders, the driver and
the configured scenes); the geometry of 14 and 15 lives in their own module (``labyrinth.py``, ``column.py``).
``examples/README.md`` lists the run times; ``legacy/`` keeps former scripts (solver benchmarks, the low-level
hybrid model of the vented box).
