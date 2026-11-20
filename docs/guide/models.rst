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

A :class:`lp.Material <loudpy.Material>` holds SI values: a **solid** has ``rho`` [kg/m³], ``E``
[Pa] and ``nu``; a **fluid** has ``rho`` and ``c`` [m/s]. Each material also records the
``temperature`` [K] its values refer to and their ``source``. Values are checked on creation
(errors outside the valid range, a ``PlausibilityWarning`` outside the usual one).

``lp.materials`` is the default library (``Paper``, ``Rubber``, ``Copper``, ``Kapton``,
``PhenolicCloth``, ``Polypropylene``, ``SolidGlue``, ``Air``). The solid values are **typical
values, not verified against measurements**: give your own for quantitative work, in the script
(``lp.Material("Steel", rho=7850, E=2.1e11, nu=0.3)``, ``lp.materials["Paper"].with_(E=1.2e9)``)
or in a library loaded with :func:`lp.load_materials <loudpy.load_materials>` and passed to the
model (``lp.Model(geo, materials=lib)``):

.. code-block:: json

   {"format": "loudpy-materials/2",
    "materials": {
      "Paper": {"kind": "solid", "rho": 1200, "E": 1.0e9, "nu": 0.30,
                "damping": {"rayleigh": [18.0, 4.6e-6], "loss_factor": 0.03},
                "temperature": 293.15, "source": "datasheet XYZ"},
      "Air":   {"kind": "fluid", "model": "air", "temperature": 293.15, "pressure": 101325.0,
                "humidity": 0.0}}}

Libraries of the earlier flat format (``"type"``, ``"eta"``, ``"alpha_ray"``, ``"beta_ray"``) are
still read; ``MaterialLibrary.save`` writes the current format.

**Damping is not a material constant.** It is chosen on the region:
:class:`lp.Rayleigh <loudpy.Rayleigh>` (default; ``C = αM + βK``),
:class:`lp.Hysteretic <loudpy.Hysteretic>` (complex stiffness ``K(1 + iη)``) or
:class:`lp.Undamped <loudpy.Undamped>`. A coefficient that is not given comes from the
material's *default damping fit* (``Material.rayleigh = (α [1/s], β [s])``,
``Material.loss_factor = η``). PML settings belong to the region (``lp.PML(strength=3, order=1)``
by default), not to the air.

Air
~~~

The library's ``Air`` is computed by :func:`lp.air <loudpy.air>` at 20 °C, 101.325 kPa, dry:
ρ = 1.2041 kg/m³, c = 343.24 m/s. 2D models, 3D scenes (``lp.Scene(…, air=…)``) and the low-level
3D toolkit all use these values. Other conditions:

.. code-block:: python

   warm = lp.air(temperature=308.15, humidity=0.8)     # 35 °C, 80 % RH: ρ = 1.126, c = 354.45
   model.add(lp.Air("subacou", material=warm), lp.PML("PML", material=warm))
   scene = lp.Scene(cabinet, drivers, air=warm)

``lp.air`` treats air as an ideal-gas mixture of dry air and water vapour (saturation pressure
from the Buck equation). Its speed of sound is within 0.05 % of Cramer's correlation (1993) over
0–40 °C, 80–101 kPa and 0–100 % RH (checked in the verification tests).

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
