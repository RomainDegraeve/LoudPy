Results
=======

Every study returns a :class:`Results <loudpy.Results>` object; ``lp.load`` reads the same
object back from a file (or a folder, a glob or a list of files of one sweep).

.. code-block:: python

   res = lp.load("Results/Files/sweep.h5")
   res                                  # summary: kind, entries, fields, blocks
   res.frequencies                      # (FrequencyResults) or res.times, modes.frequencies
   res.names                            # fields: "u" (displacement), "p" (pressure), "v", "a"

Fields
------

.. code-block:: python

   view = res.field("p", f=1000)        # nearest frequency; or index=…
   view.values, view.mesh, view.coordinate
   view.restrict("front")               # the nodes of a group
   coords, u = view.at_nodes(res.boundary("interface_acou_meca_front"))

Probes and levels
-----------------

.. code-block:: python

   p = res.probe((0.0, 0.5), "p")       # complex pressure at the nearest node, over frequency
   L = res.spl(at=(0.0, 0.5))           # SPL [dB re 20 µPa RMS]
   L = lp.spl(p)                        # the same conversion on any complex amplitudes

SPL is **RMS** everywhere in loudpy (``|p| / √2``; ``rms=False`` for peak amplitudes).

Per-entry quantities
--------------------

``res.extra("zeta")`` (damping ratio of modes), ``res.extra("I_coil")`` (coil current with a
voltage drive), and for 3D scenes the driver modal amplitudes (see :doc:`scenes`).

Modes
-----

.. code-block:: python

   modes = lp.load("modes.h5")
   modes.zeta, modes.frequencies
   modes.mode(3)                        # FieldView of mode 3
   modes.order(1)                       # the modes of azimuthal order 1 (Fourier modes)
   modes.set("fsi")                     # one mode set of a file holding several

Saving
------

``study.save(path, **metadata)`` or ``res.save(path)`` write the
:doc:`loudpy result format 2 <file_format>`; the metadata come back in ``res.meta``. Large 3D
fields are read lazily (only the rows asked for are read from disk).
