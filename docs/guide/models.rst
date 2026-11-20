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
       lp.Clamped("interface_constrained_rz"), lp.Clamped("interface_constrained_r", dofs="r"),
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
:class:`lp.PML <loudpy.PML>`              perfectly matched layer, grown by the mesher outward from its base curves
========================================= ==================================================

The first argument is the region's **name**, its label in the results. By default it also selects
the physical groups (see :doc:`conventions`): ``"glue"`` selects ``submeca_glue_1`` and
``submeca_glue_2``, ``"coil"`` selects ``submeca_coil`` but not ``submeca_coil_former``, and a glob
(``"glue_*"``) is allowed. ``on=`` selects them otherwise (see `Selections`_ below). Material
properties can be overridden on the region (``lp.Solid("cone", "Paper", E=3e9)``).

Boundaries
----------

========================================================= =========================================
:class:`lp.Clamped <loudpy.Clamped>`                       fixed displacement (``dofs="rz"`` by default, ``"r"`` or ``"z"``)
:class:`lp.Coil <loudpy.Coil>`                             where the drive acts (``direction="z"`` by default)
:class:`lp.FluidStructure <loudpy.FluidStructure>`         structure ↔ air interface (normal from the mesh)
:class:`lp.Radiation <loudpy.Radiation>`                   exact DtN radiation condition on a spherical boundary
========================================================= =========================================

As for the regions, the name is the label and selects named curve groups by default; ``on=``
selects edges of the mesh otherwise (an interface then needs no curve drawn in the geometry).

Selections
----------

Every region and boundary takes a selector as ``on=``:

========================================================= =================================================
:class:`lp.Group("membranne", "dustcap") <loudpy.Group>`   named groups, by label (naming rules, globs)
:class:`lp.Box(r=(0, 0.01), z=0.032) <loudpy.Box>`         what lies entirely inside a box; a number is a
                                                           line (``lp.Box(r=0)``: on the axis)
:class:`lp.Near((r, z)) <loudpy.Near>`                     the domain containing the point, or the
                                                           boundary curve nearest to it
:class:`lp.Interface(a, b) <loudpy.Interface>`             the edges shared by the domains of ``a`` and ``b``
:class:`lp.BoundaryOf(a) <loudpy.BoundaryOf>`              the edges around the domains of ``a``
``a | b``, ``a & b``, ``a - b``                            union, intersection, difference
========================================================= =================================================

A string stands for ``lp.Group(string)``; a region or boundary of the model stands for what it
covers. Python's precedence applies (``-`` before ``&`` before ``|``): use parentheses when mixing.

.. code-block:: python

   solids = lp.Group("membranne", "dustcap", "surround", "spider", "former", "coil", "glue")
   model.add(
       lp.Solid("cone", "Paper", on=lp.Group("membranne") | "dustcap"),
       lp.Clamped("spider_rim", on=lp.Near((0.0447, -0.00075))),
       lp.Clamped("surround_rim", on=lp.Box(z=0.032) & lp.BoundaryOf("surround")),
       lp.Clamped("axis", on=lp.BoundaryOf("dustcap") & lp.Box(r=0), dofs="r"),
       lp.Coil("drive", on=lp.Interface("coil", "former")),
       lp.FluidStructure("front", on=lp.Interface(solids, "subacou_front")),
       lp.FluidStructure("rear", on=lp.Interface(solids, "subacou_rear")),
   )

**Where they are evaluated.** A region covers whole domain groups: its selector is evaluated
before meshing, on the curve loops of the geometry. A PML grows from curves of the geometry: its
selector is evaluated before meshing, on the curves (see `PML layers`_). A boundary covers edges of the mesh: its selector is evaluated after meshing, on the
edges between two different domain groups, on the outside of the meshed domains and on the named
curve groups kept in the mesh. A boundary made of labels only (``lp.Group``) keeps one part per
named curve group, as before; any other selector gives one part named after the boundary.
``model.select(selector)`` returns the selected edges ``[corner, corner, mid]`` of the meshed model
(``kind="domain"``: the mesh groups), and ``field.restrict(selector)`` restricts a result field.

A "boundary curve" of :class:`lp.Near <loudpy.Near>` is a chain of edges that stops at corners
(a turn of more than 30°), at junctions and where the adjacent domains change (on the mesh), or a
curve of the ``.geo`` (for a PML).

PML layers
~~~~~~~~~~

``lp.PML(name, on=…)`` selects the **base curves** of the layers: straight lines r = const or
z = const, each bounding one air region. Each layer grows by ``thickness`` along the outward normal
of that air (found from the geometry, whatever the direction in which the curve is drawn), and a
corner layer fills the gap where two layers meet at a right angle. The absorption grows from the base
across the layer (along r, z, or both in a corner).

.. code-block:: python

   lp.PML("PML", thickness=0.1, tuned_to=1000.0)                      # the PML_* curve groups (labels)
   lp.PML("absorber", on=lp.Box(r=0.3) | lp.Box(z=0.532), thickness=0.1, tuned_to=1000.0)

The geometry of every layer (start, thickness, direction) is stored with the mesh
(``<mesh>.msh.json``, ``pml_layers``), not in the names of its surfaces. Layers built from named
curve groups are named after them (``PML_r+_1_front``, ``PML_r+_1_front_Corner``), the others after the
region (``absorber_1``, ``absorber_1_Corner`` …). Meshes written by earlier versions held the layer
geometry in their surface names (``PML_r+_1_front__r0=…__t=…``): they are still read.

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
``Material.loss_factor = η``). ``lp.Rayleigh.from_loss_factor(η, band=(f1, f2))`` gives the α, β
whose loss factor is η at f1 and f2 (``η(ω) = α/ω + βω``, lower in between); the library fits use
100 Hz – 1 kHz. PML settings belong to the region (``lp.PML(strength=3, order=1)`` by default), not
to the air.

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
(``model.groups()``, ``model.selection()`` show what each item selected in the geometry). A saved
mesh that keeps more named curve groups than the model needs is reused (they do not change the
triangulation).
