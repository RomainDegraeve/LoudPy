Models
======

An :class:`lp.Model <loudpy.Model>` is an axisymmetric loudspeaker: a gmsh geometry of the
meridian half-plane, the **regions** that fill its surfaces, the **boundaries** on its curves,
and a material library. Everything is declared with small value objects passed to
:meth:`Model.add <loudpy.Model.add>`.

.. code-block:: python

   model = lp.Model("speaker.geo")                     # materials: loudpy's library by default
   model.add(
       lp.Solid("membranne", "Paper", mesh_size=5e-4),
       lp.Solid("surround", "Rubber", damping=lp.Hysteretic(eta=0.1)),
       lp.Air("subacou", mesh_size=5e-3),
       lp.PML("PML", order=2, strength=1.5),
       lp.Clamped("interface_constrained"),
       lp.Coil("interface_forced"),
       lp.FluidStructure("interface_acou_meca_front"),
       lp.FluidStructure("interface_acou_meca_rear"),
   )
   model.mesh("Results/Files/mesh.msh")

Regions
-------

========================================= ==================================================
:class:`lp.Solid <loudpy.Solid>`          elastic structure (P2 triangles, axisymmetric or with an azimuthal order *m*)
:class:`lp.Air <loudpy.Air>`              linear acoustic fluid
:class:`lp.PML <loudpy.PML>`              perfectly matched layer, built by the mesher outward from the ``PML`` curves
========================================= ==================================================

The first argument selects physical groups by name (see :doc:`conventions`): ``"glue"``
selects ``submeca_glue_1`` and ``submeca_glue_2``, ``"coil"`` selects ``submeca_coil`` but not
``submeca_coil_former``, and a glob (``"glue_*"``) is allowed. Material properties can be
overridden on the region (``lp.Solid("cone", "Paper", E=3e9)``).

Boundaries
----------

========================================================= =========================================
:class:`lp.Clamped <loudpy.Clamped>`                       fixed displacement (``dofs="r"``, ``"z"``, ``"rz"``; default from the group names)
:class:`lp.Coil <loudpy.Coil>`                             where the drive acts
:class:`lp.FluidStructure <loudpy.FluidStructure>`         structure ↔ air interface (``side="front"`` / ``"rear"``)
:class:`lp.Radiation <loudpy.Radiation>`                   exact DtN radiation condition on a spherical boundary
========================================================= =========================================

Materials and damping
---------------------

``lp.materials`` is the default library (``Paper``, ``Rubber``, ``Copper``, ``Kapton``,
``PhenolicCloth``, ``Polypropylene``, ``SolidGlue``, ``Air``). Load your own with
:func:`lp.load_materials <loudpy.load_materials>` (a JSON file ``{name: {rho, E, nu, eta, …}}``)
and pass it to the model, or give an :class:`lp.Material <loudpy.Material>` directly.

Damping of a solid: :class:`lp.Rayleigh <loudpy.Rayleigh>` (default, the material's
coefficients), :class:`lp.Hysteretic <loudpy.Hysteretic>` (loss factor) or
:class:`lp.Undamped <loudpy.Undamped>`.

Drives
------

* :class:`lp.Force(amplitude) <loudpy.Force>` — a force on the coil [N]: a complex amplitude for
  harmonic studies, or a sampled signal with ``fs`` for transients.
* :class:`lp.Voltage(amplitude, Re, Bl, Le, n) <loudpy.Voltage>` — an electro-mechanical coil
  (semi-inductance model ``Z_e = Re + (jω)^n Le``); the frequency results then hold the coil
  current ``I_coil`` per frequency.

Meshing
-------

:meth:`Model.mesh(path) <loudpy.Model.mesh>` calls gmsh (P2 triangles) and **reuses** the file
when the geometry, the regions' mesh sizes and the settings are unchanged (a signature is
stored next to the mesh). ``deterministic=True`` meshes on one thread (bit-reproducible meshes).
The meshed model is an :class:`lp.Mesh <loudpy.Mesh>` with named groups
(``model.groups()``, ``model.selection()`` show what each name selected).
